"""PipelineService — typed contract for Agentic Graph RAG pipeline.

All clients (FastAPI, MCP, Streamlit) use this service.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rag_core.models import PipelineTrace, QAResult

from agentic_graph_rag.agent.retrieval_agent import run as agent_run
from agentic_graph_rag.trace_store import TraceStore, create_trace_store

if TYPE_CHECKING:
    from neo4j import Driver
    from openai import OpenAI

    from agentic_graph_rag.reasoning.reasoning_engine import ReasoningEngine

logger = logging.getLogger(__name__)

# Canonical tool list — single source of truth for routes, MCP, and service.
TOOL_NAMES: tuple[str, ...] = (
    "vector_search",
    "cypher_traverse",
    "hybrid_search",
    "comprehensive_search",
    "temporal_query",
    "full_document_read",
)


class PipelineService:
    """Typed contract for the Agentic Graph RAG pipeline."""

    def __init__(
        self,
        driver: Driver,
        openai_client: OpenAI,
        reasoning: ReasoningEngine | None = None,
        trace_store: TraceStore | None = None,
    ):
        self._driver = driver
        self._client = openai_client
        self._reasoning = reasoning
        self._trace_store = trace_store or create_trace_store()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        mode: str = "agent_pattern",
    ) -> QAResult:
        """Full pipeline: route -> retrieve -> generate -> trace."""
        use_llm = mode == "agent_llm"
        reasoning = self._reasoning if mode == "agent_mangle" else None

        qa = agent_run(
            text,
            self._driver,
            openai_client=self._client,
            use_llm_router=use_llm,
            reasoning=reasoning,
        )

        if qa.trace:
            self._cache_trace(qa.trace)

        return qa

    def search(self, text: str, tool: str = "vector_search") -> list:
        """Run a specific retrieval tool directly (no agent routing)."""
        from agentic_graph_rag.agent import tools as t

        tool_map = {name: getattr(t, name) for name in TOOL_NAMES}
        fn = tool_map.get(tool)
        if fn is None:
            raise ValueError(f"Unknown tool: {tool}. Valid: {', '.join(TOOL_NAMES)}")
        return fn(text, self._driver, self._client)

    def get_trace(self, trace_id: str) -> PipelineTrace | None:
        """Retrieve trace from store."""
        return self._trace_store.get(trace_id)

    def health(self) -> dict:
        """Neo4j connectivity check."""
        try:
            with self._driver.session() as session:
                session.run("RETURN 1").single()
            return {"status": "ok", "neo4j": "connected"}
        except Exception as e:
            return {"status": "degraded", "neo4j": str(e)}

    def graph_stats(self) -> dict:
        """Node and edge counts."""
        try:
            with self._driver.session() as session:
                result = session.run(
                    "MATCH (n) RETURN count(n) AS nodes "
                    "UNION ALL "
                    "MATCH ()-[r]->() RETURN count(r) AS nodes"
                )
                counts = [r["nodes"] for r in result]
            return {"nodes": counts[0] if counts else 0, "edges": counts[1] if len(counts) > 1 else 0}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Geology domain methods
    # ------------------------------------------------------------------

    def search_geology_docs(
        self,
        query: str,
        language: str = "",
        software_product: str = "",
        geology_subdomain: str = "",
        source_type: str = "",
    ) -> list:
        """Search geology knowledge base with metadata filtering.

        Returns results with source attribution.
        """
        from rag_core.embedder import embed_query
        from rag_core.vector_store import VectorStore

        filters = {}
        if language:
            filters["language"] = language
        if software_product:
            filters["software_product"] = software_product
        if geology_subdomain:
            filters["geology_subdomain"] = geology_subdomain
        if source_type:
            filters["source_type"] = source_type

        embedding = embed_query(query)
        store = VectorStore(driver=self._driver)
        return store.search(embedding, filters=filters or None)

    def get_software_feature(
        self,
        feature_name: str,
        software_product: str = "Micromine",
    ) -> list:
        """Find how a specific software implements a feature."""
        from rag_core.embedder import embed_query
        from rag_core.vector_store import VectorStore

        embedding = embed_query(f"{software_product} {feature_name}")
        store = VectorStore(driver=self._driver)
        return store.search(
            embedding,
            filters={"software_product": software_product} if software_product else None,
        )

    def compare_implementations(
        self,
        feature: str,
        products: list[str] | None = None,
    ) -> dict[str, list]:
        """Compare how different GGIS products implement a feature."""
        from rag_core.embedder import embed_query
        from rag_core.vector_store import VectorStore

        if products is None:
            products = ["Micromine", "Surpac", "ГЕОМИКС"]

        embedding = embed_query(feature)
        store = VectorStore(driver=self._driver)

        results: dict[str, list] = {}
        for product in products:
            results[product] = store.search(
                embedding,
                top_k=5,
                filters={"software_product": product},
            )
        return results

    def get_standard(self, standard_name: str) -> list:
        """Find Russian regulatory standards (ГКЗ, ГОСТ, etc.)."""
        from rag_core.embedder import embed_query
        from rag_core.vector_store import VectorStore

        embedding = embed_query(standard_name)
        store = VectorStore(driver=self._driver)
        return store.search(
            embedding,
            filters={"source_type": "standard"},
        )

    def list_sources(
        self,
        source_type: str = "",
        software_product: str = "",
    ) -> list[dict]:
        """List available sources in the knowledge base."""
        with self._driver.session() as session:
            where_parts = []
            params: dict = {}
            if source_type:
                where_parts.append("c.source_type = $source_type")
                params["source_type"] = source_type
            if software_product:
                where_parts.append("c.software_product = $software_product")
                params["software_product"] = software_product

            where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

            result = session.run(
                f"""
                MATCH (c:RagChunk)
                {where_clause}
                RETURN DISTINCT
                    c.source_file AS source_file,
                    c.source_type AS source_type,
                    c.software_product AS software_product,
                    c.language AS language,
                    count(c) AS chunk_count
                ORDER BY c.software_product, c.source_file
                """,
                **params,
            )
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_trace(self, trace: PipelineTrace) -> None:
        """Add trace to store."""
        self._trace_store.put(trace)

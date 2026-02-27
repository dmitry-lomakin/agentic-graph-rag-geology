"""MCP server tools for Agentic Graph RAG — Geology Domain.

Provides 3 generic tools + 5 geology-specific tools.
Tools are functions that can be registered with FastMCP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_graph_rag.service import PipelineService


def create_mcp_tools(service: PipelineService) -> dict:
    """Create MCP tool functions bound to a PipelineService instance.

    Returns a dict of {tool_name: callable} for registration with FastMCP
    or for direct testing.
    """

    # --- Generic tools ---

    def resolve_intent(query: str, mode: str = "agent_pattern") -> dict:
        """Resolve user query via Agentic Graph RAG pipeline."""
        qa = service.query(query, mode=mode)
        return qa.model_dump()

    def search_graph(query: str, tool: str = "vector_search") -> dict:
        """Search the knowledge graph using a specific retrieval tool."""
        try:
            results = service.search(query, tool=tool)
        except ValueError as e:
            return {"error": str(e)}
        return {
            "results": [r.model_dump() for r in results],
        }

    def explain_trace(trace_id: str) -> dict:
        """Get provenance trace by ID."""
        trace = service.get_trace(trace_id)
        if trace is None:
            return {"error": f"Trace {trace_id} not found"}
        return trace.model_dump()

    # --- Geology domain tools ---

    def search_geology_docs(
        query: str,
        language: str = "",
        software_product: str = "",
        geology_subdomain: str = "",
        source_type: str = "",
    ) -> dict:
        """Search geology knowledge base with metadata filtering.

        Args:
            query: Search query in Russian or English.
            language: Filter by language: "ru" or "en".
            software_product: Filter by software: "Micromine", "Surpac", etc.
            geology_subdomain: Filter by subdomain: block_modeling, reserve_calc,
                geostatistics, mine_planning, drillhole_data, wireframe_modeling,
                regulatory, reporting.
            source_type: Filter by source: competitor_manual, standard, paper,
                video, forum.

        Returns search results with source attribution.
        """
        results = service.search_geology_docs(
            query,
            language=language,
            software_product=software_product,
            geology_subdomain=geology_subdomain,
            source_type=source_type,
        )
        return {"results": [r.model_dump() for r in results]}

    def get_software_feature(
        feature_name: str,
        software_product: str = "Micromine",
    ) -> dict:
        """Find how a GGIS software implements a specific feature.

        Args:
            feature_name: Feature to look up (e.g. "kriging", "block modeling").
            software_product: Target software (default: Micromine).
        """
        results = service.get_software_feature(feature_name, software_product)
        return {"results": [r.model_dump() for r in results]}

    def compare_implementations(
        feature: str,
        products: str = "Micromine,Surpac,ГЕОМИКС",
    ) -> dict:
        """Compare how different GGIS products implement a feature.

        Args:
            feature: Feature to compare (e.g. "block modeling", "pit optimization").
            products: Comma-separated list of software products to compare.
        """
        product_list = [p.strip() for p in products.split(",") if p.strip()]
        result_map = service.compare_implementations(feature, product_list)
        return {
            product: [r.model_dump() for r in results]
            for product, results in result_map.items()
        }

    def get_standard(standard_name: str) -> dict:
        """Find Russian regulatory standards (ГКЗ, ГОСТ, etc.).

        Args:
            standard_name: Standard to look up (e.g. "ГКЗ классификация запасов",
                "Закон о недрах", "Приказ МПР 278").
        """
        results = service.get_standard(standard_name)
        return {"results": [r.model_dump() for r in results]}

    def list_sources(
        source_type: str = "",
        software_product: str = "",
    ) -> dict:
        """List available sources in the geology knowledge base.

        Args:
            source_type: Filter by type: competitor_manual, standard, paper, video, forum.
            software_product: Filter by software product name.
        """
        sources = service.list_sources(source_type, software_product)
        return {"sources": sources}

    return {
        "resolve_intent": resolve_intent,
        "search_graph": search_graph,
        "explain_trace": explain_trace,
        "search_geology_docs": search_geology_docs,
        "get_software_feature": get_software_feature,
        "compare_implementations": compare_implementations,
        "get_standard": get_standard,
        "list_sources": list_sources,
    }


def mount_mcp(app, service: PipelineService):
    """Mount FastMCP server on a FastAPI/Starlette app.

    Uses SSE transport at /mcp/sse.
    """
    try:
        from fastmcp import FastMCP

        mcp = FastMCP("Geology Graph RAG")
        tools = create_mcp_tools(service)

        # --- Generic tools ---

        @mcp.tool()
        def resolve_intent(query: str, mode: str = "agent_pattern") -> dict:
            """Resolve user query via Agentic Graph RAG pipeline.
            Returns answer with full provenance trace."""
            return tools["resolve_intent"](query, mode)

        @mcp.tool()
        def search_graph(query: str, tool: str = "vector_search") -> dict:
            """Search the knowledge graph. Tools: vector_search, cypher_traverse,
            hybrid_search, comprehensive_search, temporal_query, full_document_read."""
            return tools["search_graph"](query, tool)

        @mcp.tool()
        def explain_trace(trace_id: str) -> dict:
            """Get full provenance trace by trace ID."""
            return tools["explain_trace"](trace_id)

        # --- Geology domain tools ---

        @mcp.tool()
        def search_geology_docs(
            query: str,
            language: str = "",
            software_product: str = "",
            geology_subdomain: str = "",
            source_type: str = "",
        ) -> dict:
            """Search geology knowledge base with metadata filtering.
            Supports filters: language (ru/en), software_product (Micromine/Surpac/etc),
            geology_subdomain (block_modeling/reserve_calc/geostatistics/etc),
            source_type (competitor_manual/standard/paper/video/forum)."""
            return tools["search_geology_docs"](
                query, language, software_product, geology_subdomain, source_type,
            )

        @mcp.tool()
        def get_software_feature(
            feature_name: str,
            software_product: str = "Micromine",
        ) -> dict:
            """Find how a GGIS software implements a specific feature.
            Default: Micromine. Other: Surpac, Datamine, Leapfrog, Vulcan, ГЕОМИКС."""
            return tools["get_software_feature"](feature_name, software_product)

        @mcp.tool()
        def compare_implementations(
            feature: str,
            products: str = "Micromine,Surpac,ГЕОМИКС",
        ) -> dict:
            """Compare how different GGIS products implement a feature.
            Products: comma-separated list (e.g. 'Micromine,Surpac')."""
            return tools["compare_implementations"](feature, products)

        @mcp.tool()
        def get_standard(standard_name: str) -> dict:
            """Find Russian regulatory standards (ГКЗ, ГОСТ, Закон о недрах, etc.)."""
            return tools["get_standard"](standard_name)

        @mcp.tool()
        def list_sources(
            source_type: str = "",
            software_product: str = "",
        ) -> dict:
            """List available sources in the geology knowledge base.
            Filter by source_type or software_product."""
            return tools["list_sources"](source_type, software_product)

        try:
            mcp.mount(app, path="/mcp")
        except TypeError:
            # Newer fastmcp versions use different mount API
            mcp.mount(app)

    except ImportError:
        import logging
        logging.getLogger(__name__).warning("fastmcp not installed — MCP server disabled")

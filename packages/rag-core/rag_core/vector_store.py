"""Neo4j Vector Index store for RAG.

From RAG 2.0 — CRUD operations on Neo4j vector index with cosine similarity search.
Extended with geology domain metadata as individual Neo4j properties.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from rag_core.config import get_settings
from rag_core.models import Chunk, SearchResult

if TYPE_CHECKING:
    from neo4j import Driver

logger = logging.getLogger(__name__)

INDEX_NAME = "rag_chunks_index"
FULLTEXT_INDEX_NAME = "rag_chunks_fulltext"
NODE_LABEL = "RagChunk"
EMBEDDING_PROPERTY = "embedding"

# Geology metadata fields stored as individual Neo4j properties
_GEOLOGY_FIELDS = (
    "source_file", "source_type", "source_url", "software_product",
    "geology_subdomain", "section_path", "language", "doc_date",
    "page_numbers", "has_figures", "chunk_index",
)

# Fields indexed for fast filtering
_INDEXED_FIELDS = ("source_type", "software_product", "geology_subdomain", "language")


class VectorStore:
    """Neo4j-backed vector store with cosine similarity search."""

    def __init__(self, driver: Driver | None = None) -> None:
        cfg = get_settings()
        if driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                cfg.neo4j.uri,
                auth=(cfg.neo4j.user, cfg.neo4j.password),
            )
        else:
            self._driver = driver

    def close(self) -> None:
        self._driver.close()

    def init_index(self) -> None:
        """Create vector index and property indexes in Neo4j if they don't exist."""
        cfg = get_settings()
        with self._driver.session() as session:
            session.run(
                f"""
                CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
                FOR (n:{NODE_LABEL})
                ON (n.{EMBEDDING_PROPERTY})
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: $dimensions,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """,
                dimensions=cfg.embedding.dimensions,
            )

            # Create property indexes for geology metadata filtering
            for field in _INDEXED_FIELDS:
                session.run(
                    f"CREATE INDEX idx_{NODE_LABEL}_{field} IF NOT EXISTS "
                    f"FOR (n:{NODE_LABEL}) ON (n.{field})"
                )

            # Create fulltext index for BM25 hybrid search
            session.run(
                f"""
                CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
                FOR (n:{NODE_LABEL})
                ON EACH [n.content]
                """
            )

        logger.info("Vector index '%s' + fulltext index '%s' initialized with %d property indexes",
                     INDEX_NAME, FULLTEXT_INDEX_NAME, len(_INDEXED_FIELDS))

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Store chunks as Neo4j nodes with embeddings and geology metadata.

        Returns count added.
        """
        if not chunks:
            return 0

        with self._driver.session() as session:
            for chunk in chunks:
                chunk_id = chunk.id or hashlib.md5(chunk.content.encode()).hexdigest()
                session.run(
                    f"""
                    MERGE (c:{NODE_LABEL} {{id: $id}})
                    SET c.content = $content,
                        c.context = $context,
                        c.enriched_content = $enriched_content,
                        c.{EMBEDDING_PROPERTY} = $embedding,
                        c.source_file = $source_file,
                        c.source_type = $source_type,
                        c.source_url = $source_url,
                        c.software_product = $software_product,
                        c.geology_subdomain = $geology_subdomain,
                        c.section_path = $section_path,
                        c.language = $language,
                        c.doc_date = $doc_date,
                        c.page_numbers = $page_numbers,
                        c.has_figures = $has_figures,
                        c.chunk_index = $chunk_index
                    """,
                    id=chunk_id,
                    content=chunk.content,
                    context=chunk.context,
                    enriched_content=chunk.enriched_content,
                    embedding=chunk.embedding,
                    source_file=chunk.source_file,
                    source_type=chunk.source_type,
                    source_url=chunk.source_url,
                    software_product=chunk.software_product,
                    geology_subdomain=chunk.geology_subdomain,
                    section_path=chunk.section_path,
                    language=chunk.language,
                    doc_date=chunk.doc_date,
                    page_numbers=chunk.page_numbers,
                    has_figures=chunk.has_figures,
                    chunk_index=chunk.chunk_index,
                )

        logger.info("Added %d chunks to vector store", len(chunks))
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """Search vector index by cosine similarity with optional metadata filters.

        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            filters: Optional dict of {field: value} for metadata filtering.
                     Supported fields: source_type, software_product,
                     geology_subdomain, language.
        """
        if top_k is None:
            top_k = get_settings().retrieval.top_k_vector

        # Build WHERE clause from filters
        where_parts: list[str] = []
        params: dict = {"top_k": top_k, "embedding": query_embedding}

        if filters:
            for field, value in filters.items():
                if field in _INDEXED_FIELDS and value:
                    param_name = f"filter_{field}"
                    where_parts.append(f"node.{field} = ${param_name}")
                    params[param_name] = value

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        query = f"""
            CALL db.index.vector.queryNodes(
                '{INDEX_NAME}', $top_k, $embedding
            )
            YIELD node, score
            {where_clause}
            RETURN node.id AS id,
                   node.content AS content,
                   node.context AS context,
                   node.source_file AS source_file,
                   node.source_type AS source_type,
                   node.source_url AS source_url,
                   node.software_product AS software_product,
                   node.geology_subdomain AS geology_subdomain,
                   node.section_path AS section_path,
                   node.language AS language,
                   score
            ORDER BY score DESC
        """

        with self._driver.session() as session:
            result = session.run(query, **params)

            results = []
            for i, record in enumerate(result):
                chunk = Chunk(
                    id=record["id"] or "",
                    content=record["content"] or "",
                    context=record["context"] or "",
                    source_file=record["source_file"] or "",
                    source_type=record["source_type"] or "",
                    source_url=record["source_url"],
                    software_product=record["software_product"],
                    geology_subdomain=record["geology_subdomain"] or "general",
                    section_path=record["section_path"] or "",
                    language=record["language"] or "",
                )
                results.append(
                    SearchResult(chunk=chunk, score=record["score"], rank=i + 1)
                )

        return results

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict[str, str] | None = None,
        vector_weight: float = 0.7,
        fulltext_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Hybrid search combining vector similarity + fulltext (BM25) via RRF fusion.

        Essential for geology terminology — pure vector search misses exact terms
        like "кригинг" or "variogram nugget", while fulltext catches them.

        Args:
            query_text: Raw query text for fulltext search.
            query_embedding: Query vector for vector search.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            vector_weight: Weight for vector results in RRF (default 0.7).
            fulltext_weight: Weight for fulltext results in RRF (default 0.3).
            rrf_k: RRF constant (default 60).
        """
        if top_k is None:
            top_k = get_settings().retrieval.top_k_vector

        # Run vector search
        vector_results = self.search(query_embedding, top_k=top_k * 2, filters=filters)

        # Run fulltext search
        fulltext_results = self._fulltext_search(query_text, top_k=top_k * 2, filters=filters)

        # RRF fusion
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, r in enumerate(vector_results, start=1):
            key = r.chunk.id or r.chunk.content[:80]
            scores[key] = scores.get(key, 0.0) + vector_weight / (rrf_k + rank)
            result_map[key] = r

        for rank, r in enumerate(fulltext_results, start=1):
            key = r.chunk.id or r.chunk.content[:80]
            scores[key] = scores.get(key, 0.0) + fulltext_weight / (rrf_k + rank)
            if key not in result_map:
                result_map[key] = r

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

        merged = []
        for i, key in enumerate(sorted_keys):
            result = result_map[key]
            merged.append(SearchResult(
                chunk=result.chunk,
                score=scores[key],
                rank=i + 1,
                source="hybrid",
            ))

        logger.info("Hybrid search: %d vector + %d fulltext → %d merged results",
                     len(vector_results), len(fulltext_results), len(merged))
        return merged

    def _fulltext_search(
        self,
        query_text: str,
        top_k: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """Search using Neo4j fulltext (Lucene) index."""
        # Escape Lucene special chars
        import re
        escaped = re.sub(r'[+\-&|!(){}[\]^"~*?:\\/]', " ", query_text)
        escaped = escaped.strip()
        if not escaped:
            return []

        where_parts: list[str] = []
        params: dict = {"query": escaped, "top_k": top_k}

        if filters:
            for field, value in filters.items():
                if field in _INDEXED_FIELDS and value:
                    param_name = f"ft_{field}"
                    where_parts.append(f"node.{field} = ${param_name}")
                    params[param_name] = value

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        cypher = f"""
            CALL db.index.fulltext.queryNodes(
                '{FULLTEXT_INDEX_NAME}', $query
            )
            YIELD node, score
            {where_clause}
            RETURN node.id AS id,
                   node.content AS content,
                   node.context AS context,
                   node.source_file AS source_file,
                   node.source_type AS source_type,
                   node.source_url AS source_url,
                   node.software_product AS software_product,
                   node.geology_subdomain AS geology_subdomain,
                   node.section_path AS section_path,
                   node.language AS language,
                   score
            ORDER BY score DESC
            LIMIT $top_k
        """

        with self._driver.session() as session:
            result = session.run(cypher, **params)

            results = []
            for i, record in enumerate(result):
                chunk = Chunk(
                    id=record["id"] or "",
                    content=record["content"] or "",
                    context=record["context"] or "",
                    source_file=record["source_file"] or "",
                    source_type=record["source_type"] or "",
                    source_url=record["source_url"],
                    software_product=record["software_product"],
                    geology_subdomain=record["geology_subdomain"] or "general",
                    section_path=record["section_path"] or "",
                    language=record["language"] or "",
                )
                results.append(
                    SearchResult(chunk=chunk, score=record["score"], rank=i + 1,
                                 source="fulltext")
                )

        return results

    def delete_all(self) -> int:
        """Delete all RagChunk nodes. Returns count deleted."""
        with self._driver.session() as session:
            result = session.run(
                f"""
                MATCH (c:{NODE_LABEL})
                WITH c, count(c) AS total
                DETACH DELETE c
                RETURN total
                """
            )
            record = result.single()
            count = record["total"] if record else 0

        logger.info("Deleted %d chunks from vector store", count)
        return count

    def count(self) -> int:
        """Return total number of chunks."""
        with self._driver.session() as session:
            result = session.run(
                f"MATCH (c:{NODE_LABEL}) RETURN count(c) AS total"
            )
            record = result.single()
            return record["total"] if record else 0

#!/usr/bin/env python3
"""Ingest documents into the Agentic Graph RAG pipeline.

Usage:
    python scripts/ingest.py <file_or_directory> [options]

Options:
    --skip-enrichment   Skip LLM contextual enrichment (faster, no OpenAI calls for enrichment)
    --skip-skeleton     Skip skeleton indexing (no entity extraction, just vector store)
    --use-gpu           Enable GPU acceleration for Docling document parsing
    --force             Re-ingest files even if already processed

Examples:
    python scripts/ingest.py data/sample_graph_rag.txt
    python scripts/ingest.py data/sample_graph_rag.txt --skip-enrichment
    python scripts/ingest.py ~/documents/ --use-gpu
    python scripts/ingest.py data/gkz --geology --source-type standard --language ru --skip-enrichment --skip-skeleton
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pymangle"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ingest")

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "manifests"
CHECKPOINT_FILE = CHECKPOINT_DIR / "ingest_checkpoint.json"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _file_key(file_path: str) -> str:
    """Create a unique key for a file based on path, size, and mtime."""
    st = os.stat(file_path)
    raw = f"{os.path.abspath(file_path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_checkpoint() -> dict:
    """Load checkpoint data from disk."""
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_checkpoint(data: dict) -> None:
    """Save checkpoint data to disk."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _mark_done(checkpoint: dict, file_path: str, chunks: int) -> None:
    """Mark a file as successfully ingested."""
    key = _file_key(file_path)
    checkpoint[key] = {
        "path": os.path.abspath(file_path),
        "name": os.path.basename(file_path),
        "size": os.path.getsize(file_path),
        "chunks": chunks,
    }
    _save_checkpoint(checkpoint)


def _is_done(checkpoint: dict, file_path: str) -> bool:
    """Check if a file has already been ingested."""
    return _file_key(file_path) in checkpoint


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str,
    *,
    skip_enrichment: bool = False,
    skip_skeleton: bool = False,
    use_gpu: bool = False,
    geology: bool = False,
    source_type: str = "",
    software_product: str = "",
    language: str = "ru",
) -> int:
    """Ingest a single file through the full pipeline. Returns chunk count."""
    from neo4j import GraphDatabase
    from rag_core.chunker import chunk_geology, chunk_text
    from rag_core.config import get_settings, make_openai_client
    from rag_core.embedder import embed_chunks
    from rag_core.enricher import enrich_chunks
    from rag_core.loader import load_file
    from rag_core.vector_store import VectorStore

    cfg = get_settings()
    if not cfg.openai.api_key and not cfg.openai.base_url:
        logger.error("OPENAI_API_KEY is not set. Please configure it in .env")
        sys.exit(1)

    # 1. Load document
    logger.info("Loading: %s (GPU=%s)", file_path, use_gpu)
    text = load_file(file_path, use_gpu=use_gpu)
    logger.info("Loaded %d characters", len(text))

    if not text.strip():
        logger.warning("Document is empty, skipping: %s", file_path)
        return 0

    # 2. Chunk
    if geology:
        chunks = chunk_geology(
            text,
            source_file=os.path.basename(file_path),
            source_type=source_type,
            software_product=software_product or None,
            language=language,
        )
    else:
        chunks = chunk_text(text)
    logger.info("Created %d chunks", len(chunks))

    # 3. Enrich (optional)
    if not skip_enrichment:
        logger.info("Enriching chunks with LLM context...")
        chunks = enrich_chunks(chunks)
        logger.info("Enrichment complete")
    else:
        logger.info("Skipping enrichment (--skip-enrichment)")

    # 4. Embed
    logger.info("Embedding %d chunks...", len(chunks))
    chunks = embed_chunks(chunks)
    logger.info("Embedding complete")

    # 5. Store in vector index
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    try:
        store = VectorStore(driver=driver)
        try:
            store.init_index()
        except Exception as exc:
            if "ServiceUnavailable" in type(exc).__name__ or "Connection refused" in str(exc):
                logger.error(
                    "Cannot connect to Neo4j at %s. "
                    "Is it running? Try: docker compose up -d",
                    cfg.neo4j.uri,
                )
                sys.exit(1)
            raise
        stored = store.add_chunks(chunks)
        logger.info("Stored %d chunks in Neo4j vector index", stored)

        # 6. Skeleton indexing (optional)
        if not skip_skeleton:
            from agentic_graph_rag.indexing.dual_node import build_dual_graph, embed_phrase_nodes
            from agentic_graph_rag.indexing.skeleton import build_skeleton_index

            openai_client = make_openai_client(cfg)
            embeddings = [c.embedding for c in chunks if c.embedding]

            logger.info("Building skeleton index...")
            entities, relationships, skeletal, peripheral = build_skeleton_index(
                chunks, embeddings, openai_client=openai_client,
            )
            logger.info(
                "Skeleton: %d entities, %d relationships (%d skeletal, %d peripheral)",
                len(entities), len(relationships), len(skeletal), len(peripheral),
            )

            if entities:
                logger.info("Building dual-node graph...")
                phrase_nodes, passage_nodes, link_count = build_dual_graph(
                    entities, chunks, driver, relationships=relationships,
                )
                logger.info(
                    "Dual graph: %d phrase nodes, %d passage nodes, %d links",
                    len(phrase_nodes), len(passage_nodes), link_count,
                )

                logger.info("Embedding phrase nodes...")
                updated = embed_phrase_nodes(phrase_nodes, driver, openai_client)
                logger.info("Updated %d phrase node embeddings", updated)
        else:
            logger.info("Skipping skeleton indexing (--skip-skeleton)")

    finally:
        driver.close()

    logger.info("Done: %s", file_path)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into Agentic Graph RAG",
    )
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip LLM enrichment")
    parser.add_argument("--skip-skeleton", action="store_true", help="Skip skeleton indexing")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU for Docling")
    parser.add_argument("--geology", action="store_true", help="Use geology-aware chunker (768 tokens, subdomain classification)")
    parser.add_argument("--source-type", default="", help="Source type: competitor_manual|standard|paper|video|forum")
    parser.add_argument("--software-product", default="", help="Software product name (e.g. Micromine, Surpac)")
    parser.add_argument("--language", default="ru", help="Document language: ru|en")
    parser.add_argument("--force", action="store_true", help="Re-ingest files even if already processed")
    args = parser.parse_args()

    target = os.path.abspath(args.path)
    if not os.path.exists(target):
        logger.error("Path does not exist: %s", target)
        sys.exit(1)

    files: list[str] = []
    if os.path.isfile(target):
        files = [target]
    elif os.path.isdir(target):
        for root, _dirs, names in os.walk(target):
            for name in sorted(names):
                full = os.path.join(root, name)
                if not name.startswith("."):
                    files.append(full)

    if not files:
        logger.error("No files found at: %s", target)
        sys.exit(1)

    # Filter out already-ingested files
    checkpoint = _load_checkpoint()
    if not args.force:
        todo = []
        skipped = 0
        for f in files:
            if _is_done(checkpoint, f):
                skipped += 1
            else:
                todo.append(f)
        if skipped:
            logger.info("Skipping %d already-ingested file(s) (use --force to re-ingest)", skipped)
        files = todo

    if not files:
        logger.info("All files already ingested. Nothing to do.")
        return

    logger.info("Ingesting %d file(s)...", len(files))
    succeeded = 0
    failed = 0
    total_chunks = 0
    for i, f in enumerate(files, 1):
        logger.info("[%d/%d] %s", i, len(files), os.path.basename(f))
        try:
            n_chunks = ingest_file(
                f,
                skip_enrichment=args.skip_enrichment,
                skip_skeleton=args.skip_skeleton,
                use_gpu=args.use_gpu,
                geology=args.geology,
                source_type=args.source_type,
                software_product=args.software_product,
                language=args.language,
            )
            _mark_done(checkpoint, f, n_chunks)
            succeeded += 1
            total_chunks += n_chunks
        except Exception as exc:
            logger.error("FAILED %s: %s", os.path.basename(f), exc)
            failed += 1

    logger.info(
        "Done. %d succeeded (%d chunks), %d failed out of %d files.",
        succeeded, total_chunks, failed, len(files),
    )


if __name__ == "__main__":
    main()

"""Bridge script: reads manifest CSV → parses → chunks → embeds → stores in Neo4j.

Connects the scraper/parser pipeline to the Graph RAG ingestion system.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logging_config import setup_logging
from scripts.utils.manifest import Manifest, ManifestEntry, now_iso

logger = logging.getLogger(__name__)


def _parse_entry(entry: ManifestEntry, project_root: Path) -> str | None:
    """Parse a single manifest entry, return markdown text or None."""
    from scripts.ingest.dispatcher import parse_document

    file_path = project_root / entry.download_path
    if not file_path.exists():
        logger.warning("File not found: %s", file_path)
        return None

    try:
        parsed = parse_document(file_path)
        if parsed.is_empty:
            logger.warning("Empty parse result: %s", file_path)
            return None
        return parsed.markdown
    except Exception as e:
        logger.error("Parse error for %s: %s", file_path, e)
        return None


def _chunk_and_embed(
    markdown: str,
    entry: ManifestEntry,
) -> list:
    """Chunk markdown and embed using the rag-core pipeline.

    Returns list of Chunk objects with embeddings.
    """
    from rag_core.chunker import chunk_geology
    from rag_core.embedder import embed_chunks

    chunks = chunk_geology(
        markdown=markdown,
        source_file=entry.download_path,
        source_type=entry.source_type or "",
        source_url=entry.source_url,
        software_product=entry.software_product,
        language=entry.language or "ru",
        section_path=entry.section_path or "",
        geology_subdomain=entry.geology_subdomain or "",
    )

    if not chunks:
        return []

    embed_chunks(chunks)
    return chunks


def _store_in_neo4j(chunks: list, driver) -> int:
    """Store chunks in Neo4j via VectorStore and dual-node graph."""
    from rag_core.vector_store import VectorStore

    store = VectorStore(driver=driver)
    return store.add_chunks(chunks)


@click.command()
@click.argument("manifest_path", type=click.Path(exists=True, path_type=Path))
@click.option("--limit", type=int, default=0, help="Process at most N entries (0=all)")
@click.option("--force", is_flag=True, help="Re-process already parsed entries")
@click.option("--dry-run", is_flag=True, help="Parse and chunk but don't store")
@click.option("--log-level", default="INFO", help="Logging level")
def main(
    manifest_path: Path,
    limit: int,
    force: bool,
    dry_run: bool,
    log_level: str,
):
    """Ingest scraped documents from a manifest CSV into Neo4j Graph RAG.

    Reads the manifest, parses each document, chunks with geology-aware
    chunker, embeds with multilingual-e5-large, and stores in Neo4j.
    """
    setup_logging(level=log_level)

    manifest = Manifest(manifest_path)
    entries = manifest.entries

    if not entries:
        logger.warning("No entries in manifest %s", manifest_path)
        return

    # Filter to unprocessed entries
    if not force:
        entries = [e for e in entries if e.embedded != "true"]

    if limit > 0:
        entries = entries[:limit]

    logger.info("Processing %d entries from %s", len(entries), manifest_path.name)

    if not dry_run:
        from neo4j import GraphDatabase
        from rag_core.config import get_settings
        from rag_core.vector_store import VectorStore

        cfg = get_settings()
        driver = GraphDatabase.driver(
            cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password),
        )
        store = VectorStore(driver=driver)
        store.init_index()
    else:
        driver = None

    total_chunks = 0
    processed = 0

    for entry in entries:
        logger.info("Processing: %s", entry.download_path)

        # Parse
        markdown = _parse_entry(entry, PROJECT_ROOT)
        if markdown is None:
            continue

        # Chunk and embed
        chunks = _chunk_and_embed(markdown, entry)
        if not chunks:
            logger.warning("No chunks produced for %s", entry.download_path)
            continue

        # Store
        if not dry_run and driver:
            stored = _store_in_neo4j(chunks, driver)
            logger.info("Stored %d chunks for %s", stored, entry.download_path)
        else:
            logger.info("[dry-run] Would store %d chunks for %s",
                        len(chunks), entry.download_path)

        total_chunks += len(chunks)
        processed += 1

        # Update manifest
        entry.parsed = "true"
        entry.parse_date = now_iso()
        entry.chunk_count = str(len(chunks))
        if not dry_run:
            entry.embedded = "true"

    # Save manifest
    manifest.save()

    if not dry_run and driver:
        driver.close()

    logger.info(
        "Done: %d/%d entries processed, %d total chunks",
        processed, len(entries), total_chunks,
    )


if __name__ == "__main__":
    main()

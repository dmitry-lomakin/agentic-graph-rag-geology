#!/usr/bin/env python3
"""CLI entry point for the document parsing, chunking, and embedding pipeline.

Usage:
    python scripts/ingest/run_parser.py parse -m gkz_rf --limit 30
    python scripts/ingest/run_parser.py parse -m geokniga
    python scripts/ingest/run_parser.py parse -m micromine_webhelp_25.5
    python scripts/ingest/run_parser.py parse --all
    python scripts/ingest/run_parser.py parse -m gkz_rf --format doc --limit 4
    python scripts/ingest/run_parser.py parse -m gkz_rf --dry-run
    python scripts/ingest/run_parser.py parse -m gkz_rf --force
    python scripts/ingest/run_parser.py parse -m gkz_rf --skip-scanned
    python scripts/ingest/run_parser.py figures -m gkz_rf
    python scripts/ingest/run_parser.py chunk -m gkz_rf
    python scripts/ingest/run_parser.py chunk --all
    python scripts/ingest/run_parser.py chunk -m gkz_rf --limit 10 --dry-run
    python scripts/ingest/run_parser.py chunk -m gkz_rf --force
    python scripts/ingest/run_parser.py chunk -m gkz_rf --chunk-size 512 --chunk-overlap 80
    python scripts/ingest/run_parser.py embed -m gkz_rf
    python scripts/ingest/run_parser.py embed --all
    python scripts/ingest/run_parser.py embed -m gkz_rf --limit 10 --dry-run
    python scripts/ingest/run_parser.py embed -m gkz_rf --force
    python scripts/ingest/run_parser.py embed --batch-size 64
    python scripts/ingest/run_parser.py status
"""

import logging
import sys
from pathlib import Path

import click

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logging_config import setup_logging
from scripts.utils.manifest import Manifest, ManifestEntry, now_iso

MANIFESTS_DIR = PROJECT_ROOT / "manifests"
PARSED_DOCS_DIR = PROJECT_ROOT / "parsed-docs"
CHUNKED_DOCS_DIR = PROJECT_ROOT / "chunked-docs"
FIGURES_DIR = PROJECT_ROOT / "figures"
RAW_DOCS_DIR = PROJECT_ROOT / "raw-docs"


def _discover_manifests() -> list[Path]:
    """Find all manifest CSV files."""
    if not MANIFESTS_DIR.exists():
        return []
    return sorted(MANIFESTS_DIR.glob("*.csv"))


def _load_manifest(name: str) -> Manifest:
    """Load a manifest by name (without .csv extension)."""
    path = MANIFESTS_DIR / f"{name}.csv"
    if not path.exists():
        raise click.ClickException(f"Manifest not found: {path}")
    return Manifest(path)


def _output_path_for(entry: ManifestEntry) -> Path:
    """Compute the parsed-docs output path mirroring the raw-docs structure.

    Given download_path like:
        raw-docs/standards/gkz/orders/filename.pdf
    Returns:
        parsed-docs/standards/gkz/orders/filename.md
    """
    download = Path(entry.download_path)

    # Strip the raw-docs/ prefix to get relative path
    try:
        rel = download.relative_to("raw-docs")
    except ValueError:
        # download_path might be absolute or differently rooted
        rel = download

    return PARSED_DOCS_DIR / rel.with_suffix(".md")


def _figures_dir_for(entry: ManifestEntry) -> Path:
    """Compute the figures output directory for a document."""
    download = Path(entry.download_path)
    try:
        rel = download.relative_to("raw-docs")
    except ValueError:
        rel = download
    return FIGURES_DIR / rel.parent


def _chunked_output_path_for(entry: ManifestEntry) -> Path:
    """Compute the chunked-docs output path mirroring the raw-docs structure.

    Given download_path like:
        raw-docs/standards/gkz/orders/filename.pdf
    Returns:
        chunked-docs/standards/gkz/orders/filename.jsonl
    """
    download = Path(entry.download_path)
    try:
        rel = download.relative_to("raw-docs")
    except ValueError:
        rel = download
    return CHUNKED_DOCS_DIR / rel.with_suffix(".jsonl")


def _resolve_download_path(entry: ManifestEntry) -> Path:
    """Resolve download_path to absolute path."""
    download = Path(entry.download_path)
    if download.is_absolute():
        return download
    return PROJECT_ROOT / download


def _filter_entries(
    manifest: Manifest,
    format_filter: str | None,
    force: bool,
    skip_scanned: bool,
) -> list[ManifestEntry]:
    """Filter manifest entries for parsing.

    Args:
        manifest: Loaded manifest.
        format_filter: File extension to filter (e.g. "doc", "pdf"), or None.
        force: If True, include already-parsed entries.
        skip_scanned: If True, skip scanned PDFs (detected via PyMuPDF).

    Returns:
        List of entries to parse.
    """
    entries = []

    for entry in manifest:
        # Skip already-parsed unless --force
        if not force and entry.parsed == "true":
            continue

        # Skip entries without download path
        if not entry.download_path:
            continue

        # Apply format filter
        if format_filter:
            ext = Path(entry.download_path).suffix.lower().lstrip(".")
            if ext != format_filter.lower().lstrip("."):
                continue

        # Skip entries whose files don't exist on disk
        abs_path = _resolve_download_path(entry)
        if not abs_path.exists():
            continue

        # Skip scanned PDFs if requested
        if skip_scanned and abs_path.suffix.lower() == ".pdf":
            from scripts.ingest.parse_pdf import is_scanned_pdf
            if is_scanned_pdf(abs_path):
                continue

        entries.append(entry)

    return entries


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Document parsing pipeline for geology knowledge base."""
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(name="geo-parser", level=level, log_dir=PROJECT_ROOT / "logs")


@cli.command()
@click.option(
    "-m", "--manifest", "manifest_name",
    help="Manifest name (without .csv). Required unless --all is used.",
)
@click.option("--all", "parse_all", is_flag=True, help="Parse all manifests")
@click.option("--format", "format_filter", help="Filter by file extension (e.g. doc, pdf, xlsx)")
@click.option("--limit", type=int, default=0, help="Max documents to parse (0 = unlimited)")
@click.option("--dry-run", is_flag=True, help="Show what would be parsed without parsing")
@click.option("--force", is_flag=True, help="Re-parse already-parsed documents")
@click.option("--skip-scanned", is_flag=True, help="Skip scanned PDFs (use for text-native only runs)")
def parse(
    manifest_name: str | None,
    parse_all: bool,
    format_filter: str | None,
    limit: int,
    dry_run: bool,
    force: bool,
    skip_scanned: bool,
) -> None:
    """Parse raw documents into markdown."""
    from scripts.ingest.dispatcher import parse_document

    logger = logging.getLogger("geo-parser")

    if not manifest_name and not parse_all:
        raise click.ClickException("Specify -m MANIFEST_NAME or --all")

    # Collect manifests to process
    if parse_all:
        manifest_paths = _discover_manifests()
        if not manifest_paths:
            raise click.ClickException(f"No manifests found in {MANIFESTS_DIR}")
        manifests = [(p.stem, Manifest(p)) for p in manifest_paths]
    else:
        manifests = [(manifest_name, _load_manifest(manifest_name))]

    total_parsed = 0
    total_skipped = 0
    total_errors = 0

    for mname, manifest in manifests:
        entries = _filter_entries(manifest, format_filter, force, skip_scanned)

        if limit > 0:
            entries = entries[:limit]

        if not entries:
            logger.info("Manifest %s: no documents to parse", mname)
            continue

        logger.info("Manifest %s: %d documents to parse", mname, len(entries))

        if dry_run:
            for entry in entries:
                ext = Path(entry.download_path).suffix
                click.echo(f"  [{ext}] {entry.download_path}")
            continue

        for i, entry in enumerate(entries, 1):
            abs_path = _resolve_download_path(entry)
            logger.info(
                "[%d/%d] Parsing: %s",
                i, len(entries), abs_path.name,
            )

            try:
                doc = parse_document(abs_path)
            except Exception:
                logger.error(
                    "Failed to parse: %s", abs_path.name, exc_info=True
                )
                total_errors += 1
                continue

            if doc.is_empty:
                logger.warning(
                    "Skipping empty result: %s (%d words)",
                    abs_path.name, doc.word_count,
                )
                total_skipped += 1
                continue

            # Write markdown to parsed-docs/
            output_path = _output_path_for(entry)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(doc.markdown, encoding="utf-8")

            # Update manifest
            entry.parsed = "true"
            entry.parse_date = now_iso()
            manifest.save()

            total_parsed += 1
            logger.info(
                "  -> %s (%d words, parser=%s)",
                output_path.relative_to(PROJECT_ROOT),
                doc.word_count,
                doc.parser_used,
            )

            if doc.parse_warnings:
                for w in doc.parse_warnings:
                    logger.warning("  ! %s", w)

    if not dry_run:
        logger.info(
            "Done: %d parsed, %d skipped (empty), %d errors",
            total_parsed, total_skipped, total_errors,
        )


@cli.command()
@click.option(
    "-m", "--manifest", "manifest_name", required=True,
    help="Manifest name (without .csv)",
)
@click.option("--limit", type=int, default=0, help="Max PDFs to process (0 = unlimited)")
def figures(manifest_name: str, limit: int) -> None:
    """Extract figures from PDF documents."""
    from scripts.ingest.extract_figures import extract_figures_from_pdf

    logger = logging.getLogger("geo-parser")
    manifest = _load_manifest(manifest_name)

    entries = [
        e for e in manifest
        if Path(e.download_path).suffix.lower() == ".pdf"
        and _resolve_download_path(e).exists()
    ]

    if limit > 0:
        entries = entries[:limit]

    if not entries:
        logger.info("No PDF files found in manifest %s", manifest_name)
        return

    logger.info("Extracting figures from %d PDFs", len(entries))
    total_figures = 0

    for i, entry in enumerate(entries, 1):
        abs_path = _resolve_download_path(entry)
        fig_dir = _figures_dir_for(entry)

        logger.info("[%d/%d] %s", i, len(entries), abs_path.name)

        try:
            paths = extract_figures_from_pdf(abs_path, fig_dir)
            total_figures += len(paths)
        except Exception:
            logger.error(
                "Failed to extract figures: %s", abs_path.name, exc_info=True
            )

    logger.info("Done: %d figures extracted from %d PDFs", total_figures, len(entries))


@cli.command()
@click.option(
    "-m", "--manifest", "manifest_name",
    help="Manifest name (without .csv). Required unless --all is used.",
)
@click.option("--all", "chunk_all", is_flag=True, help="Chunk all manifests")
@click.option("--limit", type=int, default=0, help="Max documents to chunk (0 = unlimited)")
@click.option("--dry-run", is_flag=True, help="Show what would be chunked without chunking")
@click.option("--force", is_flag=True, help="Re-chunk already-chunked documents")
@click.option("--chunk-size", type=int, default=768, help="Target chunk size in tokens")
@click.option("--chunk-overlap", type=int, default=100, help="Overlap between chunks in tokens")
def chunk(
    manifest_name: str | None,
    chunk_all: bool,
    limit: int,
    dry_run: bool,
    force: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Chunk parsed markdown documents into JSONL for embedding."""
    from scripts.ingest.chunk import chunk_document, write_chunks_jsonl

    logger = logging.getLogger("geo-parser")

    if not manifest_name and not chunk_all:
        raise click.ClickException("Specify -m MANIFEST_NAME or --all")

    # Collect manifests
    if chunk_all:
        manifest_paths = _discover_manifests()
        if not manifest_paths:
            raise click.ClickException(f"No manifests found in {MANIFESTS_DIR}")
        manifests = [(p.stem, Manifest(p)) for p in manifest_paths]
    else:
        manifests = [(manifest_name, _load_manifest(manifest_name))]

    total_chunked = 0
    total_chunks = 0
    total_skipped = 0
    total_errors = 0

    for mname, manifest in manifests:
        # Filter entries: must be parsed, not yet chunked (unless --force), md exists
        entries = []
        for entry in manifest:
            if entry.parsed != "true":
                continue
            if not force and entry.chunk_count not in ("", "0"):
                continue
            md_path = _output_path_for(entry)
            if not md_path.exists():
                continue
            entries.append(entry)

        if limit > 0:
            entries = entries[:limit]

        if not entries:
            logger.info("Manifest %s: no documents to chunk", mname)
            continue

        logger.info("Manifest %s: %d documents to chunk", mname, len(entries))

        if dry_run:
            for entry in entries:
                md_path = _output_path_for(entry)
                size_kb = md_path.stat().st_size / 1024
                click.echo(f"  {md_path.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")
            continue

        for i, entry in enumerate(entries, 1):
            md_path = _output_path_for(entry)
            logger.info(
                "[%d/%d] Chunking: %s",
                i, len(entries), md_path.name,
            )

            try:
                records = chunk_document(
                    md_path, entry,
                    target_chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except Exception:
                logger.error(
                    "Failed to chunk: %s", md_path.name, exc_info=True
                )
                total_errors += 1
                continue

            if not records:
                logger.warning("No chunks produced: %s", md_path.name)
                total_skipped += 1
                continue

            # Write JSONL
            jsonl_path = _chunked_output_path_for(entry)
            write_chunks_jsonl(records, jsonl_path)

            # Update manifest
            entry.chunk_count = str(len(records))
            manifest.save()

            total_chunked += 1
            total_chunks += len(records)
            logger.info(
                "  -> %s (%d chunks)",
                jsonl_path.relative_to(PROJECT_ROOT), len(records),
            )

    if not dry_run:
        logger.info(
            "Done: %d docs chunked (%d chunks total), %d skipped, %d errors",
            total_chunked, total_chunks, total_skipped, total_errors,
        )


@cli.command()
@click.option(
    "-m", "--manifest", "manifest_name",
    help="Manifest name (without .csv). Required unless --all is used.",
)
@click.option("--all", "embed_all", is_flag=True, help="Embed all manifests")
@click.option("--limit", type=int, default=0, help="Max documents to embed (0 = unlimited)")
@click.option("--dry-run", is_flag=True, help="Show what would be embedded without embedding")
@click.option("--force", is_flag=True, help="Re-embed already-embedded documents")
@click.option("--batch-size", type=int, default=64, help="Batch size for embedding and upsert")
def embed(
    manifest_name: str | None,
    embed_all: bool,
    limit: int,
    dry_run: bool,
    force: bool,
    batch_size: int,
) -> None:
    """Embed chunked documents into Qdrant vector database."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from scripts.ingest.embed import (
        embed_jsonl_file,
        ensure_collection,
        get_collection_stats,
    )
    from server.config import EMBEDDING_MODEL, QDRANT_PATH

    logger = logging.getLogger("geo-parser")

    if not manifest_name and not embed_all:
        raise click.ClickException("Specify -m MANIFEST_NAME or --all")

    # Collect manifests
    if embed_all:
        manifest_paths = _discover_manifests()
        if not manifest_paths:
            raise click.ClickException(f"No manifests found in {MANIFESTS_DIR}")
        manifests = [(p.stem, Manifest(p)) for p in manifest_paths]
    else:
        manifests = [(manifest_name, _load_manifest(manifest_name))]

    # Filter entries: must have chunks, not yet embedded (unless --force), JSONL exists
    all_work: list[tuple[str, Manifest, ManifestEntry, Path]] = []
    for mname, manifest in manifests:
        for entry in manifest:
            if entry.chunk_count in ("", "0"):
                continue
            if not force and entry.embedded == "true":
                continue
            jsonl_path = _chunked_output_path_for(entry)
            if not jsonl_path.exists():
                continue
            all_work.append((mname, manifest, entry, jsonl_path))

    if limit > 0:
        all_work = all_work[:limit]

    if not all_work:
        logger.info("No documents to embed")
        return

    logger.info("%d documents to embed", len(all_work))

    if dry_run:
        for mname, _, entry, jsonl_path in all_work:
            size_kb = jsonl_path.stat().st_size / 1024
            click.echo(
                f"  [{mname}] {jsonl_path.relative_to(PROJECT_ROOT)}"
                f" ({entry.chunk_count} chunks, {size_kb:.1f} KB)"
            )
        return

    # Initialize Qdrant + model (once for all files)
    logger.info("Initializing Qdrant at %s", QDRANT_PATH)
    client = QdrantClient(path=QDRANT_PATH)
    ensure_collection(client)

    logger.info("Loading model: %s (this may take a minute)", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded")

    total_embedded = 0
    total_points = 0
    total_errors = 0

    for i, (mname, manifest, entry, jsonl_path) in enumerate(all_work, 1):
        logger.info(
            "[%d/%d] Embedding: %s (%s chunks)",
            i, len(all_work), jsonl_path.name, entry.chunk_count,
        )

        try:
            count = embed_jsonl_file(jsonl_path, client, model, batch_size=batch_size)
        except Exception:
            logger.error(
                "Failed to embed: %s", jsonl_path.name, exc_info=True
            )
            total_errors += 1
            continue

        # Update manifest
        entry.embedded = "true"
        manifest.save()

        total_embedded += 1
        total_points += count
        logger.info("  -> %d points upserted", count)

    logger.info(
        "Done: %d docs embedded (%d points total), %d errors",
        total_embedded, total_points, total_errors,
    )

    # Show collection stats
    stats = get_collection_stats(client)
    logger.info(
        "Collection '%s': %s points",
        stats.get("collection", "?"),
        stats.get("points_count", "?"),
    )


@cli.command()
def status() -> None:
    """Show parsing and chunking progress across all manifests."""
    manifest_paths = _discover_manifests()

    if not manifest_paths:
        click.echo("No manifests found.")
        return

    total_all = 0
    parsed_all = 0
    chunked_all = 0
    embedded_all = 0

    click.echo(
        f"{'Manifest':<35} {'Total':>6} {'Parsed':>7} {'Chunked':>8}"
        f" {'Embedded':>9} {'Remaining':>10} {'Progress':>9}"
    )
    click.echo("-" * 90)

    for mpath in manifest_paths:
        manifest = Manifest(mpath)
        total = len(manifest)
        parsed = sum(1 for e in manifest if e.parsed == "true")
        chunked = sum(
            1 for e in manifest
            if e.chunk_count not in ("", "0")
        )
        embedded = sum(1 for e in manifest if e.embedded == "true")
        remaining = total - parsed
        pct = (parsed / total * 100) if total > 0 else 0

        click.echo(
            f"{mpath.stem:<35} {total:>6} {parsed:>7} {chunked:>8}"
            f" {embedded:>9} {remaining:>10} {pct:>8.1f}%"
        )

        total_all += total
        parsed_all += parsed
        chunked_all += chunked
        embedded_all += embedded

    click.echo("-" * 90)
    remaining_all = total_all - parsed_all
    pct_all = (parsed_all / total_all * 100) if total_all > 0 else 0
    click.echo(
        f"{'TOTAL':<35} {total_all:>6} {parsed_all:>7} {chunked_all:>8}"
        f" {embedded_all:>9} {remaining_all:>10} {pct_all:>8.1f}%"
    )

    # Format breakdown
    click.echo("")
    click.echo("Format breakdown (unparsed only):")

    format_counts: dict[str, int] = {}
    for mpath in manifest_paths:
        manifest = Manifest(mpath)
        for entry in manifest:
            if entry.parsed == "true":
                continue
            ext = Path(entry.download_path).suffix.lower() if entry.download_path else "unknown"
            format_counts[ext] = format_counts.get(ext, 0) + 1

    for ext, count in sorted(format_counts.items(), key=lambda x: -x[1]):
        click.echo(f"  {ext:<10} {count:>6}")


if __name__ == "__main__":
    cli()

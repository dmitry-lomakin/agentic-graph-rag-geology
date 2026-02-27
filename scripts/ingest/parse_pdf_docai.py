"""PDF → Markdown parser using Google Document AI.

Sends PDFs to Document AI Enterprise Document OCR for cloud-based parsing.
Handles both scanned and text-native PDFs with Russian + English language hints.

Online processing has a 15-page limit per request, so larger PDFs are split
into chunks with PyMuPDF, processed individually, and concatenated.

Configuration via environment variables:
    DOCAI_PROJECT_ID   — GCP project ID (default: kodename-nedra)
    DOCAI_LOCATION     — processor region (default: eu)
    DOCAI_PROCESSOR_ID — processor ID (required)

Authentication: set GOOGLE_APPLICATION_CREDENTIALS to the service account
JSON key path, or use Application Default Credentials.
"""

import logging
import os
import time
from pathlib import Path

from scripts.ingest.models import ParsedDocument

logger = logging.getLogger(__name__)

DOCAI_PROJECT_ID = os.environ.get("DOCAI_PROJECT_ID", "kodename-nedra")
DOCAI_LOCATION = os.environ.get("DOCAI_LOCATION", "eu")
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID", "")

MAX_ONLINE_PAGES = 15

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds


def _split_pdf(file_path: Path, max_pages: int = MAX_ONLINE_PAGES) -> list[bytes]:
    """Split a PDF into chunks of max_pages using PyMuPDF.

    Args:
        file_path: Path to the PDF file.
        max_pages: Maximum pages per chunk.

    Returns:
        List of PDF byte strings, each containing at most max_pages pages.
    """
    import fitz

    doc = fitz.open(str(file_path))
    total_pages = len(doc)

    if total_pages <= max_pages:
        pdf_bytes = file_path.read_bytes()
        doc.close()
        return [pdf_bytes]

    logger.info(
        "Splitting %s (%d pages) into %d-page chunks",
        file_path.name, total_pages, max_pages,
    )

    chunks = []
    for start in range(0, total_pages, max_pages):
        end = min(start + max_pages - 1, total_pages - 1)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end)
        chunks.append(chunk_doc.tobytes())
        chunk_doc.close()

    doc.close()
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def _get_processor_name() -> str:
    """Build the fully qualified Document AI processor resource name."""
    return (
        f"projects/{DOCAI_PROJECT_ID}"
        f"/locations/{DOCAI_LOCATION}"
        f"/processors/{DOCAI_PROCESSOR_ID}"
    )


def _call_docai(pdf_bytes: bytes) -> "google.cloud.documentai_v1.Document":
    """Send a PDF chunk to Document AI for processing.

    Retries on transient errors (quota exhaustion, service unavailable).

    Args:
        pdf_bytes: Raw PDF bytes (must be <= 15 pages).

    Returns:
        Document AI Document object with extracted text and layout.

    Raises:
        google.api_core.exceptions.GoogleAPICallError: On permanent API errors.
    """
    from google.api_core.exceptions import (
        DeadlineExceeded,
        ResourceExhausted,
        ServiceUnavailable,
    )
    from google.cloud import documentai_v1 as documentai

    client_options = {"api_endpoint": f"{DOCAI_LOCATION}-documentai.googleapis.com"}
    client = documentai.DocumentProcessorServiceClient(
        client_options=client_options,
    )

    request = documentai.ProcessRequest(
        name=_get_processor_name(),
        raw_document=documentai.RawDocument(
            content=pdf_bytes,
            mime_type="application/pdf",
        ),
        process_options=documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_native_pdf_parsing=True,
                hints=documentai.OcrConfig.Hints(
                    language_hints=["ru", "en"],
                ),
            ),
        ),
    )

    retryable_errors = (ResourceExhausted, ServiceUnavailable, DeadlineExceeded)

    for attempt in range(MAX_RETRIES):
        try:
            result = client.process_document(request=request)
            return result.document
        except retryable_errors as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "Document AI transient error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_RETRIES, delay, exc,
            )
            time.sleep(delay)

    # Unreachable, but satisfies type checker
    raise RuntimeError("Exhausted retries")


def _extract_text_from_layout(
    layout: "google.cloud.documentai_v1.Document.Page.Layout",
    full_text: str,
) -> str:
    """Extract text for a layout element using text_anchor offsets.

    Args:
        layout: A layout object with text_anchor.text_segments.
        full_text: The full document text (document.text).

    Returns:
        Extracted text string.
    """
    if not layout.text_anchor or not layout.text_anchor.text_segments:
        return ""

    parts = []
    for segment in layout.text_anchor.text_segments:
        start = int(segment.start_index) if segment.start_index else 0
        end = int(segment.end_index)
        parts.append(full_text[start:end])
    return "".join(parts).strip()


def _table_to_markdown(
    table: "google.cloud.documentai_v1.Document.Page.Table",
    full_text: str,
) -> str:
    """Convert a Document AI table to markdown format.

    Args:
        table: A Table object from document.pages[].tables.
        full_text: The full document text.

    Returns:
        Markdown-formatted table string.
    """
    def _row_cells(row) -> list[str]:
        cells = []
        for cell in row.cells:
            text = _extract_text_from_layout(cell.layout, full_text)
            # Replace newlines within cells with spaces for clean markdown
            text = text.replace("\n", " ").strip()
            cells.append(text)
        return cells

    rows = []

    # Header rows
    for row in table.header_rows:
        rows.append(_row_cells(row))

    # Body rows
    for row in table.body_rows:
        rows.append(_row_cells(row))

    if not rows:
        return ""

    # Determine column count from widest row
    col_count = max(len(r) for r in rows)

    # Pad short rows
    for row in rows:
        while len(row) < col_count:
            row.append("")

    lines = []

    # First row as header
    header = rows[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    # Remaining rows as body
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _docai_to_markdown(document: "google.cloud.documentai_v1.Document") -> str:
    """Convert a Document AI response to markdown.

    Processes pages sequentially, emitting paragraphs as text blocks and
    tables as markdown tables. Detects headings by font size.

    Args:
        document: Document AI Document object.

    Returns:
        Markdown string with extracted text and tables.
    """
    full_text = document.text
    if not full_text:
        return ""

    md_parts: list[str] = []

    for page_idx, page in enumerate(document.pages):
        if page_idx > 0:
            md_parts.append("\n---\n")

        # Collect all blocks with their vertical position for ordering
        blocks: list[tuple[float, str]] = []

        # Process paragraphs
        for paragraph in page.paragraphs:
            text = _extract_text_from_layout(paragraph.layout, full_text)
            if not text:
                continue

            # Detect headings by font size
            heading_level = _detect_heading_level(paragraph)
            if heading_level:
                text = "#" * heading_level + " " + text

            y_pos = _get_y_position(paragraph.layout)
            blocks.append((y_pos, text))

        # Process tables
        for table in page.tables:
            md_table = _table_to_markdown(table, full_text)
            if md_table:
                y_pos = _get_y_position(table.layout)
                blocks.append((y_pos, md_table))

        # Sort by vertical position and emit
        blocks.sort(key=lambda b: b[0])
        for _, text in blocks:
            md_parts.append(text)

    return "\n\n".join(md_parts)


def _detect_heading_level(
    paragraph: "google.cloud.documentai_v1.Document.Page.Paragraph",
) -> int | None:
    """Detect heading level from paragraph style/font size.

    Args:
        paragraph: A Paragraph object from a Document AI page.

    Returns:
        Heading level (1-4) or None if not a heading.
    """
    if not paragraph.layout or not paragraph.layout.text_anchor:
        return None

    # Check detected_languages for heading-like confidence
    # Check style info if available
    try:
        style = paragraph.layout.orientation
        # Document AI doesn't directly expose font size in all cases.
        # Use the bounding box height as a proxy for font size.
        if paragraph.layout.bounding_poly and paragraph.layout.bounding_poly.vertices:
            vertices = paragraph.layout.bounding_poly.vertices
            if len(vertices) >= 4:
                height = abs(vertices[2].y - vertices[0].y)
                # These thresholds work for typical A4 documents (page height ~1)
                # normalized_vertices are 0..1 range
                pass
        if (
            paragraph.layout.bounding_poly
            and paragraph.layout.bounding_poly.normalized_vertices
        ):
            nverts = paragraph.layout.bounding_poly.normalized_vertices
            if len(nverts) >= 4:
                height = abs(nverts[2].y - nverts[0].y)
                if height > 0.05:
                    return 1
                if height > 0.035:
                    return 2
                if height > 0.025:
                    return 3
    except (AttributeError, IndexError):
        pass

    return None


def _get_y_position(
    layout: "google.cloud.documentai_v1.Document.Page.Layout",
) -> float:
    """Get the top Y position of a layout element for vertical ordering.

    Args:
        layout: Layout object with bounding_poly.

    Returns:
        Top Y coordinate (0.0 if unavailable).
    """
    try:
        if layout.bounding_poly and layout.bounding_poly.normalized_vertices:
            return layout.bounding_poly.normalized_vertices[0].y
        if layout.bounding_poly and layout.bounding_poly.vertices:
            return float(layout.bounding_poly.vertices[0].y)
    except (AttributeError, IndexError):
        pass
    return 0.0


def parse_pdf_with_docai(file_path: Path) -> ParsedDocument:
    """Parse a PDF file using Google Document AI.

    Splits large PDFs into 15-page chunks, sends each to the Document AI
    Enterprise Document OCR processor, converts the response to markdown,
    and concatenates results.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ParsedDocument with extracted markdown content.

    Raises:
        ValueError: If DOCAI_PROCESSOR_ID is not configured.
        google.api_core.exceptions.GoogleAPICallError: On API errors.
    """
    if not DOCAI_PROCESSOR_ID:
        raise ValueError(
            "DOCAI_PROCESSOR_ID environment variable is required for Document AI parsing"
        )

    logger.info("Parsing with Document AI: %s", file_path.name)

    # Split PDF if needed
    chunks = _split_pdf(file_path)
    logger.info(
        "Processing %d chunk(s) for %s", len(chunks), file_path.name,
    )

    # Process each chunk
    md_parts: list[str] = []
    total_pages = 0

    for i, chunk_bytes in enumerate(chunks):
        logger.info(
            "  Chunk %d/%d (%d bytes)", i + 1, len(chunks), len(chunk_bytes),
        )
        document = _call_docai(chunk_bytes)
        chunk_md = _docai_to_markdown(document)
        if chunk_md:
            md_parts.append(chunk_md)
        total_pages += len(document.pages)

    markdown = "\n\n".join(md_parts)
    word_count = len(markdown.split())

    warnings: list[str] = []
    if word_count < 50:
        warnings.append("Very low word count -- possible Document AI extraction issue")

    logger.info(
        "Document AI done: %s -> %d pages, %d words",
        file_path.name, total_pages, word_count,
    )

    return ParsedDocument(
        title=file_path.stem,
        markdown=markdown,
        has_figures=False,
        word_count=word_count,
        page_count=total_pages,
        source_format="pdf",
        parser_used="docai",
        parse_warnings=warnings,
    )

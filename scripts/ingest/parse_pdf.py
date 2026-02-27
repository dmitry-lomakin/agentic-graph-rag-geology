"""PDF → Markdown parser with automatic scanned-document detection.

Uses PyMuPDF (fitz) to determine if a PDF is text-native or scanned:
  - Text-native (>100 chars/page): parsed with docling (faster, more accurate)
  - Scanned (<100 chars/page): parsed with marker-pdf (OCR with Russian support)

If docling fails on a text-native PDF, falls back to marker-pdf automatically.
"""

import logging
import os
from pathlib import Path

from scripts.ingest.models import ParsedDocument

logger = logging.getLogger(__name__)

# Below this threshold (chars per page), a PDF is considered scanned
SCANNED_THRESHOLD_CHARS_PER_PAGE = 100


def _get_pdf_info(file_path: Path) -> tuple[int, int]:
    """Get page count and total text length using PyMuPDF.

    Args:
        file_path: Path to PDF file.

    Returns:
        Tuple of (page_count, total_text_chars).
    """
    import fitz

    try:
        doc = fitz.open(str(file_path))
        page_count = len(doc)
        total_chars = 0
        for page in doc:
            total_chars += len(page.get_text().strip())
        doc.close()
        return page_count, total_chars
    except Exception:
        logger.warning("PyMuPDF failed to read %s", file_path.name, exc_info=True)
        return 1, 0


def is_scanned_pdf(file_path: Path) -> bool:
    """Detect whether a PDF is scanned (image-based) vs text-native.

    Uses PyMuPDF text extraction character count divided by page count.
    Scanned documents typically yield <100 chars/page while text-native
    yield >1000.

    Args:
        file_path: Path to PDF file.

    Returns:
        True if the PDF appears to be scanned.
    """
    page_count, text_length = _get_pdf_info(file_path)

    if page_count == 0:
        return True

    chars_per_page = text_length / page_count
    logger.debug(
        "%s: %d chars / %d pages = %.0f chars/page -> %s",
        file_path.name,
        text_length,
        page_count,
        chars_per_page,
        "scanned" if chars_per_page < SCANNED_THRESHOLD_CHARS_PER_PAGE else "text-native",
    )
    return chars_per_page < SCANNED_THRESHOLD_CHARS_PER_PAGE


def _get_torch_device() -> str:
    """Detect best available torch device (cuda > mps > cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _ensure_torch_device() -> str:
    """Set TORCH_DEVICE env var if not already set. Must be called before
    importing marker modules, as marker reads this at import time."""
    device = _get_torch_device()
    if "TORCH_DEVICE" not in os.environ:
        os.environ["TORCH_DEVICE"] = device
        logger.info("Set TORCH_DEVICE=%s", device)
    return os.environ["TORCH_DEVICE"]


def _parse_with_docling(file_path: Path) -> ParsedDocument:
    """Parse a text-native PDF using docling (IBM).

    Args:
        file_path: Path to PDF file.

    Returns:
        ParsedDocument with extracted markdown.

    Raises:
        Exception: If docling fails to convert the document.
    """
    from docling.document_converter import DocumentConverter

    logger.info("Parsing with docling: %s", file_path.name)
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    markdown = result.document.export_to_markdown()

    page_count, _ = _get_pdf_info(file_path)
    word_count = len(markdown.split())

    warnings: list[str] = []
    if word_count < 50:
        warnings.append("Very low word count -- possible extraction issue")

    return ParsedDocument(
        title=file_path.stem,
        markdown=markdown,
        has_figures=False,  # Figure detection done separately in extract_figures
        word_count=word_count,
        page_count=page_count,
        source_format="pdf",
        parser_used="docling",
        parse_warnings=warnings,
    )


def _parse_with_marker(file_path: Path) -> ParsedDocument:
    """Parse a scanned PDF using marker-pdf (OCR).

    Args:
        file_path: Path to PDF file.

    Returns:
        ParsedDocument with OCR-extracted markdown.
    """
    # Set TORCH_DEVICE BEFORE importing marker — it reads env at import time
    device = _ensure_torch_device()

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    logger.info("Parsing with marker-pdf (OCR) on %s: %s", device, file_path.name)
    models = create_model_dict()
    converter = PdfConverter(artifact_dict=models)
    rendered = converter(str(file_path))
    markdown = rendered.markdown

    page_count, _ = _get_pdf_info(file_path)
    word_count = len(markdown.split())

    warnings: list[str] = []
    if word_count < 50:
        warnings.append("Very low word count -- possible OCR issue")

    return ParsedDocument(
        title=file_path.stem,
        markdown=markdown,
        has_figures=False,
        word_count=word_count,
        page_count=page_count,
        source_format="pdf",
        parser_used="marker",
        parse_warnings=warnings,
    )


def parse_pdf_file(file_path: Path) -> ParsedDocument:
    """Parse a PDF file into structured markdown.

    If DOCAI_PROCESSOR_ID is set, uses Google Document AI for all PDFs
    (handles both scanned and text-native). Falls back to local parsers
    (docling/marker-pdf) if Document AI is not configured or fails.

    Args:
        file_path: Path to a PDF file on disk.

    Returns:
        ParsedDocument with extracted content.
    """
    # Try Document AI first (if configured)
    if os.environ.get("DOCAI_PROCESSOR_ID"):
        try:
            from scripts.ingest.parse_pdf_docai import parse_pdf_with_docai

            return parse_pdf_with_docai(file_path)
        except Exception:
            logger.warning(
                "Document AI failed for %s, falling back to local parser",
                file_path.name,
                exc_info=True,
            )

    # Fall back to local parsing
    scanned = is_scanned_pdf(file_path)

    if scanned:
        return _parse_with_marker(file_path)

    # Text-native: try docling first, fall back to marker
    try:
        doc = _parse_with_docling(file_path)
        if doc.is_empty:
            logger.warning(
                "docling returned near-empty result for %s, falling back to marker",
                file_path.name,
            )
            return _parse_with_marker(file_path)
        return doc
    except Exception:
        logger.warning(
            "docling failed for %s, falling back to marker-pdf",
            file_path.name,
            exc_info=True,
        )
        return _parse_with_marker(file_path)

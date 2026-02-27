"""Unified document parser dispatcher.

Routes files to the appropriate parser based on file extension. Uses lazy
imports to avoid loading heavy libraries (docling, marker-pdf) unless needed.
"""

import logging
from pathlib import Path

from scripts.ingest.models import ParsedDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".htm": "html",
    ".html": "html",
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".xlsx": "xlsx",
    ".xls": "xls",
}


def parse_document(file_path: Path) -> ParsedDocument:
    """Parse a document file into structured markdown.

    Routes to the appropriate parser based on file extension. Heavy libraries
    (docling, marker-pdf, python-docx, openpyxl) are imported lazily.

    Args:
        file_path: Path to the source document.

    Returns:
        ParsedDocument with extracted content.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    format_name = SUPPORTED_EXTENSIONS.get(ext)

    if format_name is None:
        raise ValueError(
            f"Unsupported file extension: {ext} "
            f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    logger.debug("Dispatching %s -> %s parser", file_path.name, format_name)

    if format_name == "html":
        return _parse_html(file_path)
    elif format_name == "pdf":
        return _parse_pdf(file_path)
    elif format_name == "docx":
        return _parse_docx(file_path)
    elif format_name == "doc":
        return _parse_doc(file_path)
    elif format_name == "xlsx":
        return _parse_xlsx(file_path)
    elif format_name == "xls":
        return _parse_xls(file_path)
    else:
        raise ValueError(f"No parser implemented for format: {format_name}")


def _parse_html(file_path: Path) -> ParsedDocument:
    """Parse HTML via parse_html.py and wrap into ParsedDocument."""
    from scripts.ingest.parse_html import parse_html_file

    page = parse_html_file(file_path)
    return ParsedDocument(
        title=page.title,
        markdown=page.markdown,
        has_figures=page.has_figures,
        word_count=page.word_count,
        source_format="html",
        parser_used="trafilatura",
    )


def _parse_pdf(file_path: Path) -> ParsedDocument:
    """Parse PDF via parse_pdf.py (docling or marker-pdf)."""
    from scripts.ingest.parse_pdf import parse_pdf_file

    return parse_pdf_file(file_path)


def _parse_docx(file_path: Path) -> ParsedDocument:
    """Parse DOCX via parse_docx.py (python-docx)."""
    from scripts.ingest.parse_docx import parse_docx_file

    return parse_docx_file(file_path)


def _parse_doc(file_path: Path) -> ParsedDocument:
    """Parse legacy DOC via parse_docx.py (libreoffice + python-docx)."""
    from scripts.ingest.parse_docx import parse_doc_file

    return parse_doc_file(file_path)


def _parse_xlsx(file_path: Path) -> ParsedDocument:
    """Parse XLSX via parse_xlsx.py (openpyxl)."""
    from scripts.ingest.parse_xlsx import parse_xlsx_file

    return parse_xlsx_file(file_path)


def _parse_xls(file_path: Path) -> ParsedDocument:
    """Parse legacy XLS via parse_xlsx.py (libreoffice + openpyxl)."""
    from scripts.ingest.parse_xlsx import parse_xls_file

    return parse_xls_file(file_path)

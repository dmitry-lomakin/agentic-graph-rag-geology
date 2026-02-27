"""Shared data models for the document parsing pipeline.

ParsedDocument is the unified output format for all parsers (HTML, PDF, DOCX,
XLSX). The dispatcher converts parser-specific results into this common type.
"""

from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """Result of parsing a single source document into markdown."""

    title: str
    markdown: str
    has_figures: bool
    word_count: int
    page_count: int = 0
    source_format: str = ""         # "html" | "pdf" | "doc" | "docx" | "xlsx"
    parser_used: str = ""           # "trafilatura" | "docling" | "marker" | "python-docx" | "openpyxl"
    figure_paths: list[str] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if the document has too little content to be useful."""
        return self.word_count < 10

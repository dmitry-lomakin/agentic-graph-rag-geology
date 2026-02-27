"""HTML → Markdown parser for scraped documentation pages.

Uses trafilatura for main content extraction with BeautifulSoup fallback
for pages where trafilatura returns insufficient content (common with
MadCap Flare-generated help pages that have heavy boilerplate).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup, Tag

MIN_CONTENT_LENGTH = 100  # Minimum chars before falling back to BS4


@dataclass
class ParsedPage:
    """Result of parsing an HTML page."""

    title: str
    markdown: str
    has_figures: bool
    word_count: int


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract page title from HTML."""
    # MadCap Flare uses <title> or <h1 class="Heading1">
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        title = title_tag.string.strip()
        # Strip " - Micromine Help" suffix if present
        title = re.sub(r"\s*[-|]\s*Micromine.*$", "", title)
        if title:
            return title

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return ""


def _has_figures(soup: BeautifulSoup) -> bool:
    """Check if the page contains images/figures."""
    return bool(soup.find_all("img"))


def _bs4_extract(html: str) -> str:
    """Fallback extraction using BeautifulSoup for Flare pages.

    Targets the main content area and converts to clean markdown-like text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # MadCap Flare puts content in <div class="body-container"> or <body>
    content = (
        soup.find("div", class_="body-container")
        or soup.find("div", {"id": "mc-main-content"})
        or soup.find("div", class_="topic-content")
        or soup.body
    )

    if content is None:
        return ""

    # Remove navigation, sidebar, breadcrumbs, scripts, styles
    for tag in content.find_all(
        ["nav", "script", "style", "header", "footer", "aside"]
    ):
        tag.decompose()
    for cls in [
        "breadcrumbs",
        "nav-container",
        "side-nav",
        "search-container",
        "MCBreadcrumbsBox",
    ]:
        for tag in content.find_all(class_=cls):
            tag.decompose()

    lines: list[str] = []
    for elem in content.descendants:
        if isinstance(elem, Tag):
            tag_name = elem.name
            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag_name[1])
                text = elem.get_text(strip=True)
                if text:
                    lines.append(f"\n{'#' * level} {text}\n")
            elif tag_name == "li":
                text = elem.get_text(strip=True)
                if text:
                    lines.append(f"- {text}")
            elif tag_name == "tr":
                cells = [td.get_text(strip=True) for td in elem.find_all(["td", "th"])]
                if any(cells):
                    lines.append("| " + " | ".join(cells) + " |")
            elif tag_name == "p":
                text = elem.get_text(strip=True)
                if text:
                    lines.append(text)
            elif tag_name == "pre" or tag_name == "code":
                text = elem.get_text()
                if text.strip():
                    lines.append(f"```\n{text}\n```")
            elif tag_name == "img":
                alt = elem.get("alt", "")
                src = elem.get("src", "")
                if alt or src:
                    lines.append(f"[Image: {alt or src}]")

    return "\n\n".join(lines)


def parse_html_file(file_path: Path) -> ParsedPage:
    """Parse a single HTML file into structured markdown.

    Args:
        file_path: Path to an HTML file on disk.

    Returns:
        ParsedPage with extracted content.
    """
    html = file_path.read_text(encoding="utf-8", errors="replace")
    return parse_html_string(html)


def parse_html_string(html: str) -> ParsedPage:
    """Parse an HTML string into structured markdown.

    Uses trafilatura first; falls back to BeautifulSoup if trafilatura
    returns too little content.

    Args:
        html: Raw HTML string.

    Returns:
        ParsedPage with extracted content.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = _extract_title(soup)
    has_figs = _has_figures(soup)

    # Try trafilatura first
    markdown = trafilatura.extract(
        html,
        output_format="txt",
        include_tables=True,
        include_links=False,
        include_images=False,
        favor_recall=True,
    )

    if not markdown or len(markdown) < MIN_CONTENT_LENGTH:
        markdown = _bs4_extract(html)

    if not markdown:
        markdown = ""

    # Clean up excessive whitespace
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()

    word_count = len(markdown.split())

    return ParsedPage(
        title=title,
        markdown=markdown,
        has_figures=has_figs,
        word_count=word_count,
    )

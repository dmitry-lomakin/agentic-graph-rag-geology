"""Scraper for forum.web.ru MSU geology forum (phpBB).

Covers geological software (Геософт), general geology questions, mineralogy,
drilling, and geological literature discussions. All content is in Russian.

Usage:
    python scripts/scrapers/run_scrapers.py forumwebru [--rate 0.5] [--force]
"""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlencode, parse_qs, urlparse

import aiohttp
from bs4 import BeautifulSoup

from scripts.scrapers.base_scraper import BaseScraper, DiscoveredItem, PROJECT_ROOT

BASE_URL = "https://forum.web.ru"
VIEWFORUM_URL = f"{BASE_URL}/viewforum.php"
VIEWTOPIC_URL = f"{BASE_URL}/viewtopic.php"

POSTS_PER_PAGE = 20  # phpBB default


@dataclass
class ForumSection:
    """A phpBB forum section."""

    forum_id: int
    name: str
    slug: str
    subdomain: str


SECTIONS: list[ForumSection] = [
    ForumSection(4, "Вопросы Геологии", "voprosy-geologii", "general"),
    ForumSection(14, "Геософт", "geosoft", "geological_software"),
    ForumSection(25, "Определить минерал", "opredelit-mineral", "mineralogy"),
    ForumSection(30, "Геологическая книга", "geologicheskaya-kniga", "general"),
    ForumSection(31, "Проблемы бурения", "problemy-bureniya", "drilling"),
    ForumSection(15, "Полезные ссылки", "poleznye-ssylki", "general"),
]

# Subdomain fallback classification from thread title keywords
_SUBDOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("geological_software", re.compile(
        r"(Micromine|Surpac|Datamine|Leapfrog|ГЕОМИКС|Vulcan|программ|софт)",
        re.IGNORECASE,
    )),
    ("geostatistics", re.compile(
        r"(кригинг|вариограм|геостатист|интерполяц)",
        re.IGNORECASE,
    )),
    ("reserve_calc", re.compile(
        r"(подсчёт|подсчет|запас[оыа]в|ГКЗ|кондици)",
        re.IGNORECASE,
    )),
    ("block_modeling", re.compile(
        r"(блочн[оа][еймг]\s+модел|block\s*model)",
        re.IGNORECASE,
    )),
    ("drilling", re.compile(
        r"(бурен|скважин|колонк|керн)",
        re.IGNORECASE,
    )),
    ("mineralogy", re.compile(
        r"(минерал|определи|кристалл|порода|петрограф)",
        re.IGNORECASE,
    )),
    ("mine_planning", re.compile(
        r"(карьер|горн[ыоа][еймх]\s+работ|рудник|шахт)",
        re.IGNORECASE,
    )),
    ("regulatory", re.compile(
        r"(ГОСТ|СНиП|норматив|стандарт|классификац)",
        re.IGNORECASE,
    )),
]


def _classify_subdomain(title: str, section_subdomain: str) -> str:
    """Classify thread subdomain from section default + title keywords."""
    for subdomain, pattern in _SUBDOMAIN_PATTERNS:
        if pattern.search(title):
            return subdomain
    return section_subdomain


def _safe_filename(title: str, max_len: int = 80) -> str:
    """Create a filesystem-safe version of a thread title."""
    safe = re.sub(r"[^\w\s\-.]", "", title, flags=re.UNICODE)
    safe = safe.strip()[:max_len].strip()
    return safe.replace(" ", "_") or "untitled"


def _clean_topic_url(href: str) -> tuple[str, str, str] | None:
    """Parse a phpBB topic link, return (clean_url, forum_id, topic_id) or None.

    Strips session IDs (&sid=...) and start offsets.
    """
    parsed = urlparse(href)
    params = parse_qs(parsed.query)

    topic_id = params.get("t", [None])[0]
    forum_id = params.get("f", [None])[0]

    if not topic_id:
        return None

    # Build clean URL without sid or start params
    clean_url = f"{VIEWTOPIC_URL}?t={topic_id}"
    if forum_id:
        clean_url = f"{VIEWTOPIC_URL}?f={forum_id}&t={topic_id}"

    return clean_url, forum_id or "0", topic_id


class ForumWebRuScraper(BaseScraper):
    """Scrape forum threads from forum.web.ru (phpBB)."""

    def __init__(self, rate: float = 0.5, max_concurrent: int = 2) -> None:
        manifest_path = PROJECT_ROOT / "manifests" / "forum_web_ru.csv"
        super().__init__(
            manifest_path=manifest_path,
            source_type="forum",
            software_product="",
            rate=rate,
            max_concurrent=max_concurrent,
        )

    async def _get_forum_topic_count(
        self, session: aiohttp.ClientSession, section: ForumSection
    ) -> int:
        """Detect total number of topics in a forum section from pagination."""
        url = f"{VIEWFORUM_URL}?f={section.forum_id}"
        async with self.rate_limiter:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return 0
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # phpBB pagination: look for total topic count or last page link
        # Common pattern: "X topics" text or pagination div
        pagination = soup.find("div", class_="pagination")
        if not pagination:
            return POSTS_PER_PAGE  # single page

        # Look for last page start offset in pagination links
        max_start = 0
        for a_tag in pagination.find_all("a", href=True):
            href = a_tag["href"]
            params = parse_qs(urlparse(href).query)
            start_vals = params.get("start", [])
            for val in start_vals:
                if val.isdigit():
                    max_start = max(max_start, int(val))

        return max_start + POSTS_PER_PAGE  # approximate total

    async def _fetch_forum_page(
        self,
        session: aiohttp.ClientSession,
        section: ForumSection,
        start: int,
    ) -> list[DiscoveredItem]:
        """Fetch one listing page and extract topic links."""
        url = f"{VIEWFORUM_URL}?f={section.forum_id}&start={start}"
        items: list[DiscoveredItem] = []

        async with self.rate_limiter:
            async with session.get(url) as resp:
                if resp.status != 200:
                    self.logger.warning(
                        "Forum f=%d start=%d returned status %d",
                        section.forum_id, start, resp.status,
                    )
                    return items
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        seen_urls: set[str] = set()

        # phpBB topic links contain viewtopic.php?f=...&t=...
        for a_tag in soup.find_all("a", class_="topictitle", href=True):
            href = a_tag["href"]
            # Resolve relative URLs
            abs_href = urljoin(url, href)
            parsed = _clean_topic_url(abs_href)
            if not parsed:
                continue

            clean_url, forum_id, topic_id = parsed

            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)

            title = a_tag.get_text(strip=True)
            subdomain = _classify_subdomain(title, section.subdomain)

            items.append(DiscoveredItem(
                url=clean_url,
                title=title,
                section_path=f"forum.web.ru > {section.name} > {title[:80]}",
                geology_subdomain=subdomain,
                language="ru",
                extra={
                    "topic_id": topic_id,
                    "forum_id": forum_id,
                    "section_slug": section.slug,
                    "section_name": section.name,
                },
            ))

        return items

    async def discover(
        self, session: aiohttp.ClientSession
    ) -> list[DiscoveredItem]:
        """Discover all forum topics across all sections."""
        all_items: list[DiscoveredItem] = []
        seen_urls: set[str] = set()

        for section in SECTIONS:
            self.logger.info("Discovering section: %s (f=%d)", section.name, section.forum_id)
            approx_total = await self._get_forum_topic_count(session, section)
            self.logger.info(
                "Section %s: ~%d topics estimated", section.name, approx_total
            )

            start = 0
            while start < approx_total:
                page_items = await self._fetch_forum_page(session, section, start)
                if not page_items:
                    break  # no more topics

                for item in page_items:
                    if item.url not in seen_urls:
                        seen_urls.add(item.url)
                        all_items.append(item)

                # phpBB uses topic-count-based pagination (typically 15 per page for forums)
                start += 15

            self.logger.info(
                "Section %s: %d unique topics so far",
                section.name, len(all_items),
            )

        self.logger.info("Discovery complete: %d unique topics", len(all_items))
        return all_items

    async def _fetch_thread_page(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[str, int]:
        """Fetch a single thread page, return (posts_html, max_start_offset)."""
        async with self.rate_limiter:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return "", 0
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Detect max pagination offset
        max_start = 0
        pagination = soup.find("div", class_="pagination")
        if pagination:
            for a_tag in pagination.find_all("a", href=True):
                params = parse_qs(urlparse(a_tag["href"]).query)
                start_vals = params.get("start", [])
                for val in start_vals:
                    if val.isdigit():
                        max_start = max(max_start, int(val))

        # Extract post content divs
        posts = soup.find_all("div", class_="postbody")
        if not posts:
            # Fallback: try content divs
            posts = soup.find_all("div", class_="content")

        content_parts = []
        for post in posts:
            content_parts.append(str(post))

        return "\n<hr/>\n".join(content_parts), max_start

    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download all pages of a thread and assemble into a single HTML document."""
        first_page_html, max_start = await self._fetch_thread_page(session, item.url)

        if not first_page_html:
            return b"", 404

        all_content = [first_page_html]

        # Fetch remaining pages using start offsets
        offset = POSTS_PER_PAGE
        while offset <= max_start:
            sep = "&" if "?" in item.url else "?"
            page_url = f"{item.url}{sep}start={offset}"
            try:
                page_html, _ = await self._fetch_thread_page(session, page_url)
                if page_html:
                    all_content.append(page_html)
            except Exception as e:
                self.logger.warning(
                    "Failed to fetch offset %d of %s: %s",
                    offset, item.url, e,
                )
            offset += POSTS_PER_PAGE

        body_html = "\n<hr class='page-break'/>\n".join(all_content)
        title = item.title

        html = (
            "<!DOCTYPE html>\n"
            '<html lang="ru">\n'
            f'<head><meta charset="utf-8"><title>{title}</title></head>\n'
            "<body>\n"
            f"<h1>{title}</h1>\n"
            f'<p class="source-url">{item.url}</p>\n'
            f"{body_html}\n"
            "</body>\n"
            "</html>"
        )

        return html.encode("utf-8"), 200

    def local_path(self, item: DiscoveredItem) -> Path:
        """Return local path for saving thread HTML."""
        section_slug = (item.extra or {}).get("section_slug", "unknown")
        topic_id = (item.extra or {}).get("topic_id", "0")
        safe_title = _safe_filename(item.title)
        filename = f"t{topic_id}_{safe_title}.html"
        return (
            PROJECT_ROOT
            / "raw-docs"
            / "forums"
            / "forum_web_ru"
            / section_slug
            / filename
        )

    async def run(self, force: bool = False) -> None:
        """Override run() with longer timeout for forum threads."""
        timeout = aiohttp.ClientTimeout(total=120)
        connector = aiohttp.TCPConnector(limit=20, ssl=False)
        headers = {"User-Agent": "MCP-GeoKnowledge-Bot/1.0"}

        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector, headers=headers
        ) as session:
            self.logger.info("Starting discovery...")
            items = await self.discover(session)
            self.logger.info("Discovered %d items", len(items))

            if not force:
                items = [i for i in items if not self.manifest.has_url(i.url)]
                self.logger.info(
                    "%d items remaining after filtering already-downloaded",
                    len(items),
                )

            if not items:
                self.logger.info("Nothing to download.")
                return

            import asyncio as _asyncio

            tasks = [self._process_item(session, item) for item in items]

            success = 0
            failed = 0
            for i, coro in enumerate(_asyncio.as_completed(tasks), 1):
                result = await coro
                if result:
                    success += 1
                else:
                    failed += 1
                if i % 50 == 0 or i == len(tasks):
                    self.logger.info(
                        "Progress: %d/%d (success=%d, failed=%d)",
                        i, len(tasks), success, failed,
                    )

            self.logger.info(
                "Done. Downloaded %d, failed %d, total in manifest: %d",
                success, failed, len(self.manifest),
            )

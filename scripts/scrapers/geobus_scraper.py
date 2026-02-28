"""Scraper for geobus.ru engineering geology forum (Invision Community).

Covers lab work, drilling, geomechanics, hydrogeology, standards discussions.
All content is in Russian.

Usage:
    python scripts/scrapers/run_scrapers.py geobus [--rate 0.5] [--force]
"""

import re
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

from scripts.scrapers.base_scraper import BaseScraper, DiscoveredItem, PROJECT_ROOT

BASE_URL = "https://geobus.ru/forum"


@dataclass
class ForumSection:
    """A forum section with its ID, slug, display name, and subdomain."""

    section_id: int
    slug: str
    name: str
    subdomain: str


# Key sections on geobus.ru relevant to geology domain
SECTIONS: list[ForumSection] = [
    ForumSection(2, "laboratornye-raboty", "Лабораторные работы", "laboratory"),
    ForumSection(3, "burovye-raboty", "Буровые работы", "drilling"),
    ForumSection(4, "geomekhanika", "Геомеханика", "geomechanics"),
    ForumSection(5, "gidrogeologiya", "Гидрогеология", "hydrogeology"),
    ForumSection(6, "normativnye-dokumenty", "Нормативные документы", "regulatory"),
    ForumSection(7, "inzhenernaya-geologiya", "Инженерная геология", "general"),
    ForumSection(8, "obshchie-voprosy", "Общие вопросы", "general"),
    ForumSection(9, "programmnoe-obespechenie", "Программное обеспечение", "geological_software"),
    ForumSection(10, "izyskaniya", "Изыскания", "general"),
    ForumSection(11, "geofizika", "Геофизика", "geophysics"),
]

# Subdomain fallback classification from thread title keywords
_SUBDOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("drilling", re.compile(r"(бурен|скважин|колонк)", re.IGNORECASE)),
    ("geomechanics", re.compile(r"(геомехан|деформац|прочност|сдвиг)", re.IGNORECASE)),
    ("hydrogeology", re.compile(r"(гидрогеолог|водоносн|грунтов[ыа][хемй]\s+вод)", re.IGNORECASE)),
    ("regulatory", re.compile(r"(ГОСТ|СНиП|СП\s+\d|норматив|стандарт)", re.IGNORECASE)),
    ("laboratory", re.compile(r"(лаборатор|испытан|опыт\b|образец)", re.IGNORECASE)),
    ("geological_software", re.compile(r"(программ|софт|PLAXIS|GeoStab|Rocscience)", re.IGNORECASE)),
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


class GeobusScraper(BaseScraper):
    """Scrape forum threads from geobus.ru (Invision Community)."""

    _session_timeout = 120

    def __init__(
        self,
        rate: float = 0.5,
        max_concurrent: int = 2,
        proxies: list[str] | None = None,
    ) -> None:
        manifest_path = PROJECT_ROOT / "manifests" / "geobus_ru.csv"
        super().__init__(
            manifest_path=manifest_path,
            source_type="forum",
            software_product="",
            rate=rate,
            max_concurrent=max_concurrent,
            proxies=proxies,
        )

    async def _fetch_section_page(
        self,
        session: aiohttp.ClientSession,
        section: ForumSection,
        page: int,
    ) -> list[DiscoveredItem]:
        """Fetch one listing page of a forum section and extract topic links."""
        url = f"{BASE_URL}/{section.section_id}-{section.slug}/"
        if page > 1:
            url += f"?page={page}"

        items: list[DiscoveredItem] = []
        resp = await self._fetch(session, url)
        async with resp:
            if resp.status != 200:
                self.logger.warning(
                    "Section %s page %d returned status %d",
                    section.slug, page, resp.status,
                )
                return items
            html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Invision Community topic links: /topic/{id}-{slug}/
        topic_pattern = re.compile(r"/topic/(\d+)-([^/]+)/?")
        seen_urls: set[str] = set()

        for link in soup.find_all("a", href=topic_pattern):
            href = link["href"]
            match = topic_pattern.search(href)
            if not match:
                continue

            topic_id = match.group(1)
            topic_slug = match.group(2)
            topic_url = f"{BASE_URL}/topic/{topic_id}-{topic_slug}/"

            if topic_url in seen_urls:
                continue
            seen_urls.add(topic_url)

            title = link.get_text(strip=True) or topic_slug.replace("-", " ")
            subdomain = _classify_subdomain(title, section.subdomain)

            items.append(DiscoveredItem(
                url=topic_url,
                title=title,
                section_path=f"geobus.ru > {section.name} > {title[:80]}",
                geology_subdomain=subdomain,
                language="ru",
                extra={
                    "topic_id": topic_id,
                    "section_slug": section.slug,
                    "section_name": section.name,
                },
            ))

        return items

    async def _get_section_page_count(
        self, session: aiohttp.ClientSession, section: ForumSection
    ) -> int:
        """Detect total number of listing pages in a section."""
        url = f"{BASE_URL}/{section.section_id}-{section.slug}/"
        resp = await self._fetch(session, url)
        async with resp:
            if resp.status != 200:
                return 1
            html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Invision uses ipsPagination elements
        pagination = soup.find("ul", class_="ipsPagination")
        if not pagination:
            return 1

        # Look for last page number in pagination links
        page_links = pagination.find_all("li", class_="ipsPagination_page")
        max_page = 1
        for li in page_links:
            a_tag = li.find("a")
            text = a_tag.get_text(strip=True) if a_tag else li.get_text(strip=True)
            if text.isdigit():
                max_page = max(max_page, int(text))

        return max_page

    async def discover(
        self, session: aiohttp.ClientSession
    ) -> list[DiscoveredItem]:
        """Discover all forum topics across all sections."""
        all_items: list[DiscoveredItem] = []
        seen_urls: set[str] = set()

        for section in SECTIONS:
            self.logger.info("Discovering section: %s", section.name)
            total_pages = await self._get_section_page_count(session, section)
            self.logger.info(
                "Section %s has %d listing pages", section.name, total_pages
            )

            for page in range(1, total_pages + 1):
                page_items = await self._fetch_section_page(session, section, page)
                for item in page_items:
                    if item.url not in seen_urls:
                        seen_urls.add(item.url)
                        all_items.append(item)

            self.logger.info(
                "Section %s: %d unique topics so far",
                section.name, len(all_items),
            )

        self.logger.info("Discovery complete: %d unique topics", len(all_items))
        return all_items

    async def _fetch_thread_page(
        self, session: aiohttp.ClientSession, url: str
    ) -> tuple[str, int]:
        """Fetch a single thread page, return (html_text, page_count)."""
        resp = await self._fetch(session, url)
        async with resp:
            if resp.status != 200:
                return "", 0
            html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Detect total pages from pagination
        page_count = 1
        pagination = soup.find("ul", class_="ipsPagination")
        if pagination:
            page_links = pagination.find_all("li", class_="ipsPagination_page")
            for li in page_links:
                a_tag = li.find("a")
                text = a_tag.get_text(strip=True) if a_tag else li.get_text(strip=True)
                if text.isdigit():
                    page_count = max(page_count, int(text))

        # Extract post content
        posts = soup.find_all("div", attrs={"data-role": "commentContent"})
        if not posts:
            # Fallback: try generic post content divs
            posts = soup.find_all("div", class_="cPost_contentWrap")

        content_parts = []
        for post in posts:
            content_parts.append(str(post))

        return "\n<hr/>\n".join(content_parts), page_count

    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download all pages of a thread and assemble into a single HTML document."""
        first_page_html, page_count = await self._fetch_thread_page(session, item.url)

        if not first_page_html:
            return b"", 404

        all_content = [first_page_html]

        # Fetch remaining pages
        for page_num in range(2, page_count + 1):
            page_url = f"{item.url}page/{page_num}/"
            try:
                page_html, _ = await self._fetch_thread_page(session, page_url)
                if page_html:
                    all_content.append(page_html)
            except Exception as e:
                self.logger.warning(
                    "Failed to fetch page %d of %s: %s",
                    page_num, item.url, e,
                )

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
        filename = f"{topic_id}_{safe_title}.html"
        return (
            PROJECT_ROOT
            / "raw-docs"
            / "forums"
            / "geobus"
            / section_slug
            / filename
        )


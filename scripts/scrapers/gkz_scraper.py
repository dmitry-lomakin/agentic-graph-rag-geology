"""Scraper for ГКЗ regulatory documents from gkz-rf.ru.

Downloads laws, government decrees, ministry orders, and methodological
recommendations from the ГКЗ documents section. These are the core Russian
regulatory standards for mineral resource estimation (подсчёт запасов ТПИ).

Usage:
    python scripts/scrapers/run_scrapers.py gkz [--rate 1.0] [--force]
"""

import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import aiohttp
from bs4 import BeautifulSoup

from scripts.scrapers.base_scraper import BaseScraper, DiscoveredItem, PROJECT_ROOT

BASE_URL = "https://gkz-rf.ru/dokumenty"
TOTAL_PAGES = 6  # pages 0..5

# --- Document category classification ---

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("laws", re.compile(
        r"(федеральный закон|закон российской|ФЗ\b|закон о недрах)",
        re.IGNORECASE,
    )),
    ("decrees", re.compile(
        r"(постановление правительства|распоряжение правительства)",
        re.IGNORECASE,
    )),
    ("orders", re.compile(
        r"(приказ\b|утвержд[её]н приказом)",
        re.IGNORECASE,
    )),
    ("methodology", re.compile(
        r"(методическ[иоа][еймх]\s+рекомендаци|методика\b|руководство\b)",
        re.IGNORECASE,
    )),
    ("templates", re.compile(
        r"(макет\b|образец\b|форма\b|шаблон\b)",
        re.IGNORECASE,
    )),
    ("protocols", re.compile(
        r"(протокол\b|заключение\b)",
        re.IGNORECASE,
    )),
]

# --- Geology subdomain classification ---

_SUBDOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("reserve_calc", re.compile(
        r"(методическ[иоа][еймх]\s+рекомендаци[иймх]\s+по\s+применению|"
        r"ТЭО\s+кондиций|подсчёт[уе]?\s+запасов|"
        r"оценк[аеиу]\s+запасов|классификаци[яию]\s+запасов|"
        r"кондици[ийям])",
        re.IGNORECASE,
    )),
    ("block_modeling", re.compile(
        r"(блочн[оа][еймг]\s+модел|геостатистик)",
        re.IGNORECASE,
    )),
    ("geostatistics", re.compile(
        r"(кригинг|вариограм|интерполяци)",
        re.IGNORECASE,
    )),
    ("mine_planning", re.compile(
        r"(горн[оыа][еймг]\s+работ|проектировани[ея]\s+карьер|"
        r"открыт[ыоа][еймх]\s+горн|подземн[ыоа][еймх]\s+горн|"
        r"план\s+горн|разработк[аеиу]\s+месторожден)",
        re.IGNORECASE,
    )),
]


def _classify_category(title: str, url: str = "") -> str:
    """Determine document category subdirectory from title and URL."""
    # Mineral-specific methodology guides have tpi_ prefix in filename
    filename = Path(urlparse(url).path).stem.lower() if url else ""
    if filename.startswith("tpi_"):
        return "methodology"
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(title):
            return category
    return "orders"  # most GKZ documents are ministry orders


def _classify_subdomain(title: str, url: str = "") -> str:
    """Determine geology_subdomain from title keywords and URL."""
    # Mineral-specific methodology guides → reserve_calc
    filename = Path(urlparse(url).path).stem.lower() if url else ""
    if filename.startswith("tpi_"):
        return "reserve_calc"
    for subdomain, pattern in _SUBDOMAIN_PATTERNS:
        if pattern.search(title):
            return subdomain
    return "regulatory"


def _safe_filename(url: str) -> str:
    """Extract a safe filename from a download URL."""
    parsed = urlparse(url)
    filename = unquote(Path(parsed.path).name)
    # Replace problematic characters but keep Cyrillic
    filename = re.sub(r'[<>:"|?*]', "_", filename)
    return filename


class GkzScraper(BaseScraper):
    """Scrape ГКЗ regulatory documents from gkz-rf.ru/dokumenty."""

    def __init__(self, rate: float = 1.0, max_concurrent: int = 3) -> None:
        manifest_path = PROJECT_ROOT / "manifests" / "gkz_rf.csv"
        super().__init__(
            manifest_path=manifest_path,
            source_type="standard",
            software_product="",
            rate=rate,
            max_concurrent=max_concurrent,
        )

    async def _fetch_page(
        self, session: aiohttp.ClientSession, page: int
    ) -> list[DiscoveredItem]:
        """Fetch one page of the documents listing and extract file links."""
        url = f"{BASE_URL}?page={page}" if page > 0 else BASE_URL
        items: list[DiscoveredItem] = []

        async with self.rate_limiter:
            async with session.get(url) as resp:
                if resp.status != 200:
                    self.logger.warning(
                        "Page %d returned status %d", page, resp.status
                    )
                    return items
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Site uses docs__item divs with structured spans inside
        for doc_item in soup.find_all("div", class_="docs__item"):
            link = doc_item.find("a", href=True)
            if not link:
                continue
            href = link["href"]
            if "/sites/default/files/" not in href:
                continue

            abs_url = urljoin(BASE_URL, href)

            # Skip non-document files (images, etc.)
            ext = Path(urlparse(abs_url).path).suffix.lower()
            if ext not in (".pdf", ".docx", ".doc", ".xlsx", ".xls", ".rtf"):
                continue

            # Extract clean title from docs__title span (excludes mime/size info)
            title_span = doc_item.find("span", class_="docs__title")
            if title_span:
                title = title_span.get_text(strip=True)
            else:
                title = link.get_text(strip=True)

            # Clean up title
            title = re.sub(r"\s+", " ", title).strip()

            category = _classify_category(title, abs_url)
            subdomain = _classify_subdomain(title, abs_url)

            # Enrich mineral-specific methodology titles for better searchability
            filename_stem = Path(urlparse(abs_url).path).stem.lower()
            if filename_stem.startswith("tpi_") and not title.lower().startswith("метод"):
                title = f"Методические рекомендации по применению Классификации запасов: {title}"

            items.append(DiscoveredItem(
                url=abs_url,
                title=title,
                section_path=f"ГКЗ > {category} > {title[:80]}",
                geology_subdomain=subdomain,
                language="ru",
                extra={"category": category, "file_ext": ext},
            ))

        self.logger.info("Page %d: found %d documents", page, len(items))
        return items

    async def discover(
        self, session: aiohttp.ClientSession
    ) -> list[DiscoveredItem]:
        """Fetch all paginated pages and collect document links."""
        all_items: list[DiscoveredItem] = []
        seen_urls: set[str] = set()

        for page in range(TOTAL_PAGES):
            page_items = await self._fetch_page(session, page)
            for item in page_items:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    all_items.append(item)

        self.logger.info(
            "Discovery complete: %d unique documents across %d pages",
            len(all_items),
            TOTAL_PAGES,
        )
        return all_items

    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download a single document file."""
        async with session.get(item.url) as resp:
            data = await resp.read()
            return data, resp.status

    def local_path(self, item: DiscoveredItem) -> Path:
        """Map item to local path: raw-docs/standards/gkz/{category}/{filename}."""
        category = (item.extra or {}).get("category", "orders")
        filename = _safe_filename(item.url)
        return (
            PROJECT_ROOT
            / "raw-docs"
            / "standards"
            / "gkz"
            / category
            / filename
        )

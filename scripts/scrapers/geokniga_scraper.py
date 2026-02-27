"""Scraper for geological books from geokniga.org.

Downloads books by specified authors from the GeoKniga geological literature
portal. Initially configured for Закревский К.Е. — a foundational Russian-
language author on 3D geological modeling.

Each book page may list multiple file variants (different scan qualities).
The scraper picks the best PDF: largest file under 100 MB, or the smallest
if all exceed 100 MB.

Usage:
    python scripts/scrapers/run_scrapers.py geokniga [--rate 0.5] [--force]
"""

import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import aiohttp
from bs4 import BeautifulSoup

from scripts.scrapers.base_scraper import BaseScraper, DiscoveredItem, PROJECT_ROOT

GEOKNIGA_BASE = "https://www.geokniga.org"

# Author pages to scrape (extensible)
AUTHOR_PAGES = [
    {
        "url": f"{GEOKNIGA_BASE}/authors/5573",
        "name": "Закревский К.Е.",
        "dir_name": "zakrevsky",
    },
]

# --- Subdomain classification by title keywords ---

_SUBDOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("block_modeling", re.compile(
        r"(блочн[оа][еймг]\s+модел|геостатистик|кригинг|вариограм)",
        re.IGNORECASE,
    )),
    ("3d_modeling", re.compile(
        r"(3[dд]\s+модел|трёхмерн|геологическ[оа][еймг]\s+модел|"
        r"цифров[оа][еймг]\s+модел|компьютерн[оа][еймг]\s+модел|"
        r"построени[ея]\s+модел)",
        re.IGNORECASE,
    )),
    ("reserve_calc", re.compile(
        r"(подсчёт[уе]?\s+запасов|оценк[аеиу]\s+запасов|"
        r"классификаци[яию]\s+запасов)",
        re.IGNORECASE,
    )),
    ("geostatistics", re.compile(
        r"(математическ[оа][еймг]\s+модел|статистическ)",
        re.IGNORECASE,
    )),
]

# Max file size to prefer (bytes). Among PDFs under this limit, pick largest.
_PREFERRED_MAX_SIZE_MB = 100


def _classify_subdomain(title: str) -> str:
    """Determine geology_subdomain from book title."""
    for subdomain, pattern in _SUBDOMAIN_PATTERNS:
        if pattern.search(title):
            return subdomain
    return "general"


def _parse_file_size(context_text: str) -> float:
    """Parse file size from GeoKniga format like 'filename.pdf(30.16M)' into MB."""
    # Look for size in parentheses: (30.16M), (5.28M), (275.8M), etc.
    match = re.search(r"\((\d+(?:[.,]\d+)?)\s*(M|MB|G|GB|K|KB|КБ|МБ|ГБ)\)", context_text, re.IGNORECASE)
    if not match:
        return 0.0
    raw = match.group(1).replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    unit = match.group(2).upper()
    if unit in ("M", "MB", "МБ"):
        return value
    if unit in ("G", "GB", "ГБ"):
        return value * 1024
    if unit in ("K", "KB", "КБ"):
        return value / 1024
    return value


def _safe_filename(url: str) -> str:
    """Extract a safe filename from a download URL."""
    parsed = urlparse(url)
    filename = unquote(Path(parsed.path).name)
    filename = re.sub(r'[<>:"|?*]', "_", filename)
    return filename


class GeoknigaScraper(BaseScraper):
    """Scrape geological books from geokniga.org by author."""

    def __init__(
        self,
        rate: float = 0.5,
        max_concurrent: int = 2,
    ) -> None:
        manifest_path = PROJECT_ROOT / "manifests" / "geokniga.csv"
        super().__init__(
            manifest_path=manifest_path,
            source_type="paper",
            software_product="",
            rate=rate,
            max_concurrent=max_concurrent,
        )

    async def _fetch_author_books(
        self, session: aiohttp.ClientSession, author: dict
    ) -> list[str]:
        """Fetch an author page and return book page URLs."""
        async with self.rate_limiter:
            async with session.get(author["url"]) as resp:
                if resp.status != 200:
                    self.logger.warning(
                        "Author page %s returned %d", author["url"], resp.status
                    )
                    return []
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        book_urls: list[str] = []

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if re.match(r"/books/\d+$", href):
                abs_url = urljoin(GEOKNIGA_BASE, href)
                if abs_url not in book_urls:
                    book_urls.append(abs_url)

        self.logger.info(
            "Author '%s': found %d books", author["name"], len(book_urls)
        )
        return book_urls

    async def _fetch_book_files(
        self, session: aiohttp.ClientSession, book_url: str, author: dict
    ) -> list[DiscoveredItem]:
        """Fetch a book page and return DiscoveredItems for downloadable files."""
        async with self.rate_limiter:
            async with session.get(book_url) as resp:
                if resp.status != 200:
                    self.logger.warning(
                        "Book page %s returned %d", book_url, resp.status
                    )
                    return []
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        # Extract book metadata
        title_tag = soup.find("h1") or soup.find("title")
        book_title = title_tag.get_text(strip=True) if title_tag else ""
        book_title = re.sub(r"\s+", " ", book_title).strip()
        # Remove site name suffix like "| GeoKniga"
        book_title = re.sub(r"\s*\|\s*.*$", "", book_title).strip()

        # Extract year from page text
        year = ""
        year_match = re.search(r"Год издания[:\s]*(\d{4})", soup.get_text())
        if not year_match:
            # Try finding year in metadata table
            for text in soup.stripped_strings:
                m = re.match(r"^(\d{4})$", text)
                if m and 1990 <= int(m.group(1)) <= 2030:
                    year = m.group(1)
                    break
        else:
            year = year_match.group(1)

        # Collect all file download links
        file_candidates: list[tuple[str, float, str]] = []  # (url, size_mb, ext)

        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/bookfiles/" not in href and "../../bookfiles/" not in href:
                continue

            # Resolve the URL (handles ../../bookfiles/ pattern)
            abs_url = urljoin(book_url, href)
            # Also handle the /sites/geokniga/../../bookfiles/ pattern
            if "/sites/geokniga/" in abs_url:
                filename = abs_url.split("/bookfiles/")[-1]
                abs_url = f"{GEOKNIGA_BASE}/bookfiles/{filename}"

            ext = Path(urlparse(abs_url).path).suffix.lower()
            if ext not in (".pdf", ".djvu"):
                continue

            # Parse size from parent text — format is "filename.pdf(30.16M)"
            parent = link.find_parent()
            context_text = parent.get_text(strip=True) if parent else ""
            size_mb = _parse_file_size(context_text)

            file_candidates.append((abs_url, size_mb, ext))

        if not file_candidates:
            self.logger.warning("No files found for book: %s", book_title)
            return []

        # Select best file: prefer PDF, then pick best size
        pdf_candidates = [(u, s, e) for u, s, e in file_candidates if e == ".pdf"]
        candidates = pdf_candidates if pdf_candidates else file_candidates

        # Among candidates: pick largest under 100MB, or smallest if all > 100MB
        under_limit = [(u, s, e) for u, s, e in candidates if s <= _PREFERRED_MAX_SIZE_MB]
        if under_limit:
            best_url, best_size, best_ext = max(under_limit, key=lambda x: x[1])
        else:
            best_url, best_size, best_ext = min(candidates, key=lambda x: x[1])

        self.logger.info(
            "Book '%s': selected %.1f MB %s (from %d variants)",
            book_title[:60],
            best_size,
            best_ext,
            len(file_candidates),
        )

        subdomain = _classify_subdomain(book_title)
        section = f"{author['name']} > {book_title}"

        return [DiscoveredItem(
            url=best_url,
            title=book_title,
            section_path=section[:200],
            geology_subdomain=subdomain,
            language="ru",
            extra={
                "author": author["name"],
                "author_dir": author["dir_name"],
                "year": year,
                "book_page": book_url,
                "file_ext": best_ext,
                "file_size_mb": best_size,
            },
        )]

    async def discover(
        self, session: aiohttp.ClientSession
    ) -> list[DiscoveredItem]:
        """Discover all books from configured authors."""
        all_items: list[DiscoveredItem] = []

        for author in AUTHOR_PAGES:
            book_urls = await self._fetch_author_books(session, author)
            for book_url in book_urls:
                items = await self._fetch_book_files(session, book_url, author)
                all_items.extend(items)

        self.logger.info("Discovery complete: %d books total", len(all_items))
        return all_items

    async def run(self, force: bool = False) -> None:
        """Override to use longer timeout for large book files."""
        timeout = aiohttp.ClientTimeout(total=600)  # 10 min for 100+ MB files
        connector = aiohttp.TCPConnector(limit=20, ssl=False)
        from scripts.scrapers.base_scraper import USER_AGENT
        headers = {"User-Agent": USER_AGENT}

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

            # Download sequentially — these are large files, be gentle
            success = 0
            failed = 0
            for i, item in enumerate(items, 1):
                result = await self._process_item(session, item)
                if result:
                    success += 1
                else:
                    failed += 1
                self.logger.info(
                    "Progress: %d/%d (success=%d, failed=%d)",
                    i, len(items), success, failed,
                )

            self.logger.info(
                "Done. Downloaded %d, failed %d, total in manifest: %d",
                success, failed, len(self.manifest),
            )

    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download a single book file."""
        size_mb = (item.extra or {}).get("file_size_mb", 0)
        self.logger.info(
            "Downloading %s (%.1f MB)...", _safe_filename(item.url), size_mb
        )
        async with session.get(item.url) as resp:
            data = await resp.read()
            return data, resp.status

    def local_path(self, item: DiscoveredItem) -> Path:
        """Map item to local path: raw-docs/scientific/geokniga/{author_dir}/{filename}."""
        author_dir = (item.extra or {}).get("author_dir", "unknown")
        filename = _safe_filename(item.url)
        return (
            PROJECT_ROOT
            / "raw-docs"
            / "scientific"
            / "geokniga"
            / author_dir
            / filename
        )

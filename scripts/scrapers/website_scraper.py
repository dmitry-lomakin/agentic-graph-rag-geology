"""Scraper for micromine.ru / micromine.kz marketing websites.

Performs a depth-limited crawl starting from seed URLs, collecting HTML pages
and linked PDFs (brochures, presentations). Handles SSL certificate issues
common with these sites.
"""

import asyncio
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from scripts.scrapers.base_scraper import BaseScraper, DiscoveredItem, PROJECT_ROOT
from scripts.utils.manifest import Manifest

# Seed URLs for crawl
SEED_URLS = {
    "micromine.ru": [
        "https://micromine.ru/",
        "https://micromine.ru/products/",
        "https://micromine.ru/mining-geology/",
    ],
    "micromine.kz": [
        "https://micromine.kz/",
        "https://micromine.kz/products/",
    ],
}

MAX_DEPTH = 2  # How many link hops from seed URLs

# File extensions to skip during crawl
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".bmp", ".webp",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv",
    ".zip", ".rar", ".gz", ".tar",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
}


def _is_same_domain(url: str, domain: str) -> bool:
    """Check if URL belongs to the given domain."""
    parsed = urlparse(url)
    return parsed.netloc == domain or parsed.netloc == f"www.{domain}"


def _should_skip(url: str) -> bool:
    """Check if URL should be skipped based on extension."""
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    return ext in SKIP_EXTENSIONS


def _is_pdf(url: str) -> bool:
    """Check if URL points to a PDF file."""
    return urlparse(url).path.lower().endswith(".pdf")


class WebsiteScraper(BaseScraper):
    """Crawl micromine.ru/.kz for product pages and linked PDFs."""

    def __init__(
        self,
        domain: str = "micromine.ru",
        rate: float = 1.0,
        max_concurrent: int = 3,
    ) -> None:
        self.domain = domain
        safe_domain = domain.replace(".", "_")
        manifest_path = PROJECT_ROOT / "manifests" / f"micromine_{safe_domain}.csv"
        super().__init__(
            manifest_path=manifest_path,
            source_type="competitor_manual",
            software_product="Micromine",
            rate=rate,
            max_concurrent=max_concurrent,
        )
        self._seed_urls = SEED_URLS.get(domain, [f"https://{domain}/"])

    async def _crawl_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
        depth: int,
        visited: set[str],
        found: list[DiscoveredItem],
    ) -> None:
        """Recursively crawl a page, collecting URLs up to MAX_DEPTH."""
        if url in visited or depth > MAX_DEPTH:
            return
        visited.add(url)

        if _should_skip(url):
            return

        # PDFs are collected but not crawled further
        if _is_pdf(url):
            found.append(
                DiscoveredItem(
                    url=url,
                    title=Path(urlparse(url).path).stem,
                    section_path=f"Micromine > {self.domain} > PDF",
                    geology_subdomain="general",
                    language="ru",
                    extra={"type": "pdf"},
                )
            )
            return

        try:
            async with self.rate_limiter:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        return
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        return
                    html = await resp.text()
        except Exception as e:
            self.logger.debug("Failed to fetch %s: %s", url, e)
            return

        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        found.append(
            DiscoveredItem(
                url=url,
                title=title,
                section_path=f"Micromine > {self.domain} > {title}",
                geology_subdomain="general",
                language="ru",
                extra={"type": "html"},
            )
        )

        # Extract links for deeper crawl
        if depth < MAX_DEPTH:
            for link in soup.find_all("a", href=True):
                href = link["href"]
                abs_url = urljoin(url, href)
                # Strip fragment
                abs_url = abs_url.split("#")[0]
                # Only follow links on the same domain
                if _is_same_domain(abs_url, self.domain) and abs_url not in visited:
                    await self._crawl_page(session, abs_url, depth + 1, visited, found)

    async def discover(self, session: aiohttp.ClientSession) -> list[DiscoveredItem]:
        """Crawl seed URLs up to MAX_DEPTH and collect all pages + PDFs."""
        visited: set[str] = set()
        found: list[DiscoveredItem] = []

        for seed_url in self._seed_urls:
            self.logger.info("Crawling from seed: %s", seed_url)
            await self._crawl_page(session, seed_url, depth=0, visited=visited, found=found)

        self.logger.info(
            "Discovery complete: %d pages from %s", len(found), self.domain
        )
        return found

    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download a single page or PDF."""
        async with session.get(item.url) as resp:
            data = await resp.read()
            return data, resp.status

    def local_path(self, item: DiscoveredItem) -> Path:
        """Map URL to local file path."""
        parsed = urlparse(item.url)
        # Build path from URL structure
        path_parts = parsed.path.strip("/").split("/")
        if not path_parts or path_parts == [""]:
            path_parts = ["index"]

        # Add extension if missing
        filename = path_parts[-1]
        if not Path(filename).suffix:
            if item.extra and item.extra.get("type") == "pdf":
                filename += ".pdf"
            else:
                filename += ".html"
            path_parts[-1] = filename

        return (
            PROJECT_ROOT
            / "raw-docs"
            / "competitor-manuals"
            / "micromine"
            / "website"
            / self.domain
            / Path(*path_parts)
        )

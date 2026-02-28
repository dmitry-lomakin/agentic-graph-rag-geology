"""Scraper for Micromine Alastri documentation on Atlassian Confluence.

Uses the Confluence REST API to fetch pages from publicly accessible wiki
spaces. No authentication required as of 2026-02.

Available spaces:
    - 2MA1: Russian documentation (~424 pages)
    - 2MAD: English documentation (~444 pages)

Usage:
    python scripts/scrapers/run_scrapers.py confluence --space 2MA1
    python scripts/scrapers/run_scrapers.py confluence --space 2MAD --rate 3.0
"""

import logging
import re
from pathlib import Path
from typing import Any

import aiohttp

from scripts.scrapers.base_scraper import BaseScraper, DiscoveredItem, PROJECT_ROOT

CONFLUENCE_BASE = "https://micromine.atlassian.net/wiki"
REST_API = f"{CONFLUENCE_BASE}/rest/api"

# Known spaces with their default language
SPACE_INFO: dict[str, dict[str, str]] = {
    "2MA1": {"name": "Micromine Alastri Docs (RU)", "language": "ru"},
    "2MAD": {"name": "Micromine Alastri Docs (EN)", "language": "en"},
}

# Map title prefixes to geology subdomains.
# Micromine Alastri is a mine-planning suite; product prefixes in page titles
# identify which module the page belongs to.
TITLE_PREFIX_MAP: dict[str, str] = {
    "RR.": "reserve_calc",
    "HI.": "mine_planning",
    "TS.": "mine_planning",
    "PS.": "mine_planning",
    "SC.": "mine_surveying",
    "UI.": "general",
}

# Keywords in ancestor page titles → subdomain (checked if prefix match fails)
ANCESTOR_KEYWORD_MAP: dict[str, str] = {
    "rapid reserver": "reserve_calc",
    "haul infinity": "mine_planning",
    "tactical scheduler": "mine_planning",
    "production scheduler": "mine_planning",
    "spatial conformetrics": "mine_surveying",
    "block model": "block_modeling",
    "блочная модель": "block_modeling",
    "установка": "general",
    "installation": "general",
    "лицензирование": "general",
    "licensing": "general",
    "пользовательского интерфейса": "general",
    "user interface": "general",
    "упражнения": "general",
    "exercises": "general",
}


def _classify_subdomain(title: str, ancestors: list[dict[str, Any]]) -> str:
    """Classify a page into a geology subdomain."""
    # 1. Check title prefix (most reliable)
    for prefix, subdomain in TITLE_PREFIX_MAP.items():
        if title.startswith(prefix):
            return subdomain

    # 2. Check ancestor page titles
    ancestor_text = " ".join(a.get("title", "") for a in ancestors).lower()
    for keyword, subdomain in ANCESTOR_KEYWORD_MAP.items():
        if keyword in ancestor_text:
            return subdomain

    # 3. Check title keywords
    lower_title = title.lower()
    for keyword, subdomain in ANCESTOR_KEYWORD_MAP.items():
        if keyword in lower_title:
            return subdomain

    # Default: Alastri is a mine-planning suite
    return "mine_planning"


def _build_section_path(title: str, ancestors: list[dict[str, Any]]) -> str:
    """Build a human-readable breadcrumb from ancestor pages."""
    parts = ["Micromine Alastri"]
    # Skip the root space page (first ancestor is the space homepage)
    for ancestor in ancestors[1:]:
        parts.append(ancestor.get("title", ""))
    parts.append(title)
    return " > ".join(parts)


def _safe_filename(title: str, max_len: int = 80) -> str:
    """Create a filesystem-safe version of a page title."""
    safe = re.sub(r"[^\w\s\-.]", "", title, flags=re.UNICODE)
    safe = safe.strip()[:max_len].strip()
    return safe.replace(" ", "_")


class ConfluenceScraper(BaseScraper):
    """Scraper for Micromine Alastri Confluence wiki pages."""

    def __init__(
        self,
        space_key: str = "2MA1",
        rate: float = 2.0,
        max_concurrent: int = 5,
    ) -> None:
        self.space_key = space_key.upper()
        space_info = SPACE_INFO.get(self.space_key, {"language": "en"})
        self._default_language = space_info["language"]

        manifest_path = (
            PROJECT_ROOT
            / "manifests"
            / f"micromine_confluence_{self.space_key.lower()}.csv"
        )
        super().__init__(
            manifest_path=manifest_path,
            source_type="competitor_manual",
            software_product="Micromine Alastri",
            rate=rate,
            max_concurrent=max_concurrent,
        )

    async def discover(
        self, session: aiohttp.ClientSession
    ) -> list[DiscoveredItem]:
        """Fetch all pages in the Confluence space via paginated REST API."""
        items: list[DiscoveredItem] = []
        start = 0
        limit = 25  # keep responses small; Confluence default

        while True:
            url = (
                f"{REST_API}/content"
                f"?spaceKey={self.space_key}"
                f"&type=page"
                f"&limit={limit}"
                f"&start={start}"
                f"&expand=ancestors,version"
            )

            try:
                async with self.rate_limiter:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            self.logger.warning(
                                "Content API returned %d at start=%d",
                                resp.status,
                                start,
                            )
                            break
                        data = await resp.json()
            except Exception as e:
                self.logger.error(
                    "Failed to fetch page list at start=%d: %s", start, e
                )
                break

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                page_id = page["id"]
                title = page.get("title", "")
                ancestors = page.get("ancestors", [])
                version = page.get("version", {}).get("number", 1)

                page_url = (
                    f"{CONFLUENCE_BASE}/spaces/{self.space_key}"
                    f"/pages/{page_id}"
                )
                subdomain = _classify_subdomain(title, ancestors)
                section_path = _build_section_path(title, ancestors)

                items.append(
                    DiscoveredItem(
                        url=page_url,
                        title=title,
                        section_path=section_path,
                        geology_subdomain=subdomain,
                        language=self._default_language,
                        extra={
                            "page_id": page_id,
                            "version": version,
                            "ancestors": [
                                {"id": a["id"], "title": a.get("title", "")}
                                for a in ancestors
                            ],
                        },
                    )
                )

            # Check if there are more pages
            size = data.get("size", len(results))
            if size < limit:
                break
            start += limit
            self.logger.debug(
                "Fetched %d pages so far (start=%d)", len(items), start
            )

        self.logger.info(
            "Discovered %d Confluence pages in space %s",
            len(items),
            self.space_key,
        )
        return items

    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download page content as rendered HTML via export_view."""
        page_id = item.extra["page_id"] if item.extra else ""
        api_url = f"{REST_API}/content/{page_id}?expand=body.export_view"

        async with session.get(api_url) as resp:
            if resp.status != 200:
                return b"", resp.status
            data = await resp.json()

        body_html = (
            data.get("body", {}).get("export_view", {}).get("value", "")
        )
        title = data.get("title", item.title)

        # Wrap in a full HTML document for parse_html.py compatibility
        html = (
            "<!DOCTYPE html>\n"
            f'<html lang="{self._default_language}">\n'
            f"<head><meta charset=\"utf-8\"><title>{title}</title></head>\n"
            "<body>\n"
            f"<h1>{title}</h1>\n"
            f"{body_html}\n"
            "</body>\n"
            "</html>"
        )

        return html.encode("utf-8"), resp.status

    def local_path(self, item: DiscoveredItem) -> Path:
        """Return path for saving page as HTML."""
        page_id = item.extra["page_id"] if item.extra else "unknown"
        safe_title = _safe_filename(item.title)
        filename = f"{page_id}_{safe_title}.html"
        return (
            PROJECT_ROOT
            / "raw-docs"
            / "competitor-manuals"
            / "micromine"
            / "confluence"
            / self.space_key.lower()
            / filename
        )

"""Scraper for Micromine Zendesk knowledge base.

Uses the Zendesk Help Center API to fetch articles. Note: as of 2026-02,
the Micromine Zendesk API returns empty results and the help center returns
403. This scraper is ready for when access becomes available (e.g., via
API token authentication or if the KB is reopened).

Usage:
    Set ZENDESK_API_TOKEN env var if authentication is required.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from urllib.parse import urljoin

import aiohttp

from scripts.scrapers.base_scraper import BaseScraper, DiscoveredItem, PROJECT_ROOT
from scripts.utils.manifest import Manifest

ZENDESK_BASE = "https://micromine.zendesk.com"
API_PREFIX = f"{ZENDESK_BASE}/api/v2/help_center/en-us"

# Map Zendesk section names to geology subdomains
SECTION_SUBDOMAIN_MAP = {
    "block model": "block_modeling",
    "drill hole": "drillhole_data",
    "drillhole": "drillhole_data",
    "geostatist": "geostatistics",
    "kriging": "geostatistics",
    "variogram": "geostatistics",
    "wireframe": "wireframe_modeling",
    "implicit": "implicit_modeling",
    "pit optim": "mine_planning",
    "mine design": "mine_planning",
    "scheduling": "mine_planning",
    "survey": "mine_surveying",
    "report": "reporting",
    "python": "python_api",
    "install": "general",
    "license": "general",
    "getting started": "general",
}


def _classify_section(section_name: str) -> str:
    """Map a Zendesk section name to a geology subdomain."""
    lower = section_name.lower()
    for keyword, subdomain in SECTION_SUBDOMAIN_MAP.items():
        if keyword in lower:
            return subdomain
    return "general"


class ZendeskScraper(BaseScraper):
    """Scraper for Micromine Zendesk Help Center articles."""

    def __init__(
        self,
        rate: float = 5.0,
        max_concurrent: int = 5,
    ) -> None:
        manifest_path = PROJECT_ROOT / "manifests" / "micromine_zendesk.csv"
        super().__init__(
            manifest_path=manifest_path,
            source_type="competitor_manual",
            software_product="Micromine",
            rate=rate,
            max_concurrent=max_concurrent,
        )
        self._sections: dict[int, str] = {}
        self._api_token = os.environ.get("ZENDESK_API_TOKEN")

    def _auth_headers(self) -> dict[str, str]:
        """Return auth headers if API token is available."""
        if self._api_token:
            return {"Authorization": f"Bearer {self._api_token}"}
        return {}

    async def _fetch_sections(self, session: aiohttp.ClientSession) -> None:
        """Fetch all sections to map section_id → name."""
        url = f"{API_PREFIX}/sections.json?per_page=100"
        headers = self._auth_headers()
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    self.logger.warning("Sections API returned %d", resp.status)
                    return
                data = await resp.json()
                for section in data.get("sections", []):
                    self._sections[section["id"]] = section.get("name", "")
        except Exception as e:
            self.logger.warning("Failed to fetch sections: %s", e)

    async def discover(self, session: aiohttp.ClientSession) -> list[DiscoveredItem]:
        """Fetch all articles from Zendesk Help Center API (paginated)."""
        await self._fetch_sections(session)

        items: list[DiscoveredItem] = []
        url: str | None = f"{API_PREFIX}/articles.json?per_page=100"
        headers = self._auth_headers()

        while url:
            try:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        self.logger.warning(
                            "Articles API returned %d for %s", resp.status, url
                        )
                        break
                    data = await resp.json()
            except Exception as e:
                self.logger.error("Failed to fetch articles: %s", e)
                break

            for article in data.get("articles", []):
                article_id = article["id"]
                title = article.get("title", "")
                section_id = article.get("section_id", 0)
                section_name = self._sections.get(section_id, "")
                html_url = article.get("html_url", "")
                subdomain = _classify_section(section_name)

                items.append(
                    DiscoveredItem(
                        url=html_url or f"{API_PREFIX}/articles/{article_id}.json",
                        title=title,
                        section_path=f"Micromine > Zendesk > {section_name} > {title}",
                        geology_subdomain=subdomain,
                        language=article.get("locale", "en-us")[:2],
                        extra={
                            "article_id": article_id,
                            "section_name": section_name,
                        },
                    )
                )

            url = data.get("next_page")

        self.logger.info("Discovered %d Zendesk articles", len(items))
        return items

    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download article content via API."""
        article_id = item.extra["article_id"] if item.extra else ""
        api_url = f"{API_PREFIX}/articles/{article_id}.json"
        headers = self._auth_headers()

        async with session.get(api_url, headers=headers) as resp:
            if resp.status != 200:
                return b"", resp.status
            data = await resp.json()

        article = data.get("article", {})
        # Save both the metadata and the HTML body
        result = {
            "id": article.get("id"),
            "title": article.get("title"),
            "body": article.get("body", ""),
            "section_id": article.get("section_id"),
            "created_at": article.get("created_at"),
            "updated_at": article.get("updated_at"),
            "html_url": article.get("html_url"),
            "label_names": article.get("label_names", []),
        }
        content = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
        return content, resp.status

    def local_path(self, item: DiscoveredItem) -> Path:
        """Save articles as JSON files."""
        article_id = item.extra["article_id"] if item.extra else "unknown"
        safe_title = "".join(
            c if c.isalnum() or c in "-_ " else "" for c in item.title
        )[:80].strip()
        filename = f"{article_id}_{safe_title}.json"
        return (
            PROJECT_ROOT
            / "raw-docs"
            / "competitor-manuals"
            / "micromine"
            / "zendesk"
            / filename
        )

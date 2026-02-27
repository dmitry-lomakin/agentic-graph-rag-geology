"""Scraper for Micromine webhelp (MadCap Flare documentation).

Discovers all pages via SearchTopic_Chunk*.js index files, then downloads
HTML content. This is the highest-priority source (~3,100+ pages of product
documentation).
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import aiohttp

from scripts.scrapers.base_scraper import BaseScraper, DiscoveredItem, PROJECT_ROOT
from scripts.utils.manifest import Manifest

BASE_URL = "https://webhelp.micromine.com"
NUM_TOPIC_CHUNKS = 46  # Chunks 0..45 as discovered

# Map Micromine module path prefixes to geology subdomains.
# Checked against both directory names and filename prefixes (mm* and idh_*).
MODULE_SUBDOMAIN_MAP = {
    # Directory / mm-prefix based
    "mmmodel": "block_modeling",
    "mmdhole": "drillhole_data",
    "mmplicit": "implicit_modeling",
    "mmgeostat": "geostatistics",
    "mmpit": "mine_planning",
    "mmdesign": "mine_planning",
    "mmschedule": "mine_planning",
    "mmblast": "drill_blast",
    "mmsurvey": "mine_surveying",
    "mmwireframe": "wireframe_modeling",
    "mmvisualization": "visualization",
    "mmstratigraphy": "stratigraphy",
    "mmstring": "wireframe_modeling",
    "mmsurface": "surface_modeling",
    "mmsection": "cross_sections",
    "mmreport": "reporting",
    "mmcharging": "drill_blast",
    "mmstockpile": "mine_planning",
    "mmresource": "reserve_calc",
    "mmpy": "python_api",
    "mmpython": "python_api",
    "mmproject": "general",
    "mmvizex": "visualization",
    "mmfile": "general",
    "mmdata": "drillhole_data",
    "mmforms": "general",
    "mmtools": "general",
    "whatsnew": "general",
    "home": "general",
    "getting": "general",
}

# Map IDH_* filename prefixes to geology subdomains (many Flare files use this convention)
IDH_SUBDOMAIN_MAP = {
    "idh_wirefr": "wireframe_modeling",
    "idh_grade": "block_modeling",
    "idh_block": "block_modeling",
    "idh_krig": "geostatistics",
    "idh_vario": "geostatistics",
    "idh_stats": "geostatistics",
    "idh_optimi": "mine_planning",
    "idh_pit": "mine_planning",
    "idh_ring": "mine_planning",
    "idh_mining": "mine_planning",
    "idh_stope": "mine_planning",
    "idh_sc_def": "mine_planning",
    "idh_schedu": "mine_planning",
    "idh_import": "data_management",
    "idh_export": "data_management",
    "idh_macro": "scripting",
    "idh_script": "scripting",
    "idh_plt": "visualization",
    "idh_chart": "visualization",
    "idh_colour": "visualization",
    "idh_vx_": "visualization",
    "idh_rep_": "reporting",
    "idh_workfl": "general",
    "idh_licenc": "general",
    "idh_projec": "general",
    "idh_form_": "general",
    "idh_fields": "general",
    "idh_numeri": "general",
    "idh_lookup": "general",
    "idh_units": "general",
    "idh_assign": "general",
    "idh_option": "general",
    "idh_opt_se": "general",
}


def _classify_subdomain(url_path: str) -> str:
    """Classify a URL path into a geology subdomain based on module prefix."""
    parts = url_path.lower().replace("\\", "/").split("/")
    filename = parts[-1] if parts else ""

    # 1. Check mm* prefix on filename
    for prefix, subdomain in MODULE_SUBDOMAIN_MAP.items():
        if filename.startswith(prefix):
            return subdomain

    # 2. Check idh_* prefix on filename
    for prefix, subdomain in IDH_SUBDOMAIN_MAP.items():
        if filename.startswith(prefix):
            return subdomain

    # 3. Fall back to directory-based classification
    for part in parts:
        for prefix, subdomain in MODULE_SUBDOMAIN_MAP.items():
            if part.startswith(prefix):
                return subdomain

    return "general"


def _extract_section_path(url_path: str, title: str) -> str:
    """Build a human-readable section path like 'Micromine > Block Modeling > Kriging'."""
    parts = url_path.replace("\\", "/").split("/")
    # Find Content/ and take path after it
    try:
        idx = parts.index("Content")
        relevant = parts[idx + 1 :]
    except ValueError:
        relevant = parts

    if not relevant:
        return f"Micromine > {title}"

    # Use directory names + title
    dirs = [p for p in relevant[:-1] if p]
    return " > ".join(["Micromine"] + dirs + [title])


def _parse_topic_chunk_js(js_text: str) -> list[dict[str, Any]]:
    """Parse a SearchTopic_Chunk*.js file into a list of topic entries.

    Format: define({"0": {y:0, u:"../Content/...", l:-1, t:"Title", i:0.001, a:"desc"}, ...})
    """
    # Strip the define() wrapper
    match = re.search(r"define\s*\(\s*(\{.*\})\s*\)", js_text, re.DOTALL)
    if not match:
        return []

    obj_text = match.group(1)

    # Convert JS object notation to valid JSON:
    # - Add quotes around property names (y, u, l, t, i, a, m)
    obj_text = re.sub(r'(?<=[{,])\s*([a-z])\s*:', r'"\1":', obj_text)

    try:
        data = json.loads(obj_text)
    except json.JSONDecodeError:
        # Fallback: more aggressive cleanup
        # Handle trailing commas
        obj_text = re.sub(r",\s*}", "}", obj_text)
        obj_text = re.sub(r",\s*]", "]", obj_text)
        try:
            data = json.loads(obj_text)
        except json.JSONDecodeError:
            return []

    entries = []
    for key, val in data.items():
        if isinstance(val, dict) and "u" in val:
            entries.append(
                {
                    "id": key,
                    "url": val["u"],
                    "title": val.get("t", ""),
                    "abstract": val.get("a", ""),
                    "type": val.get("y", 0),
                }
            )
    return entries


class WebhelpScraper(BaseScraper):
    """Scraper for Micromine webhelp.micromine.com documentation."""

    def __init__(
        self,
        version: str = "25.5",
        rate: float = 2.0,
        max_concurrent: int = 5,
    ) -> None:
        self.version = version
        manifest_path = PROJECT_ROOT / "manifests" / f"micromine_webhelp_{version}.csv"
        super().__init__(
            manifest_path=manifest_path,
            source_type="competitor_manual",
            software_product="Micromine",
            rate=rate,
            max_concurrent=max_concurrent,
        )
        self.data_base_url = f"{BASE_URL}/mm/{version}/English/Data"
        self.content_base_url = f"{BASE_URL}/mm/{version}/English"

    async def discover(self, session: aiohttp.ClientSession) -> list[DiscoveredItem]:
        """Fetch all SearchTopic_Chunk*.js files and extract page URLs."""
        all_items: list[DiscoveredItem] = []
        seen_urls: set[str] = set()

        for chunk_idx in range(NUM_TOPIC_CHUNKS):
            chunk_url = f"{self.data_base_url}/SearchTopic_Chunk{chunk_idx}.js"
            try:
                async with session.get(chunk_url) as resp:
                    if resp.status != 200:
                        self.logger.warning(
                            "Chunk %d returned status %d, stopping discovery",
                            chunk_idx,
                            resp.status,
                        )
                        break
                    js_text = await resp.text()
            except Exception as e:
                self.logger.warning("Failed to fetch chunk %d: %s", chunk_idx, e)
                break

            entries = _parse_topic_chunk_js(js_text)
            self.logger.debug(
                "Chunk %d: parsed %d entries", chunk_idx, len(entries)
            )

            for entry in entries:
                # Skip micro-content entries (y=1) — they're snippets, not full pages
                if entry.get("type") == 1:
                    continue

                # Resolve relative URL (../Content/...) to absolute
                rel_url = entry["url"]
                if rel_url.startswith("../"):
                    rel_url = rel_url[3:]  # Strip leading ../
                full_url = f"{self.content_base_url}/{rel_url}"

                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                title = entry.get("title", "")
                subdomain = _classify_subdomain(rel_url)
                section_path = _extract_section_path(rel_url, title)

                all_items.append(
                    DiscoveredItem(
                        url=full_url,
                        title=title,
                        section_path=section_path,
                        geology_subdomain=subdomain,
                        language="en",
                        extra={"abstract": entry.get("abstract", "")},
                    )
                )

        self.logger.info(
            "Discovery complete: %d unique pages across %d chunks",
            len(all_items),
            NUM_TOPIC_CHUNKS,
        )
        return all_items

    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download a single HTML page."""
        async with session.get(item.url) as resp:
            data = await resp.read()
            return data, resp.status

    def local_path(self, item: DiscoveredItem) -> Path:
        """Map URL to local file path preserving directory structure.

        Example: .../Content/MMModel/BlockModel/KrigingParams.htm
              → raw-docs/competitor-manuals/micromine/webhelp/mm-25.5/MMModel/BlockModel/KrigingParams.htm
        """
        # Extract path after /Content/
        url = item.url
        content_marker = "/Content/"
        idx = url.find(content_marker)
        if idx != -1:
            rel = url[idx + len(content_marker) :]
        else:
            # Fallback: use everything after the version/English/
            marker = f"/mm/{self.version}/English/"
            idx = url.find(marker)
            rel = url[idx + len(marker) :] if idx != -1 else url.split("/")[-1]

        return (
            PROJECT_ROOT
            / "raw-docs"
            / "competitor-manuals"
            / "micromine"
            / "webhelp"
            / f"mm-{self.version}"
            / rel
        )

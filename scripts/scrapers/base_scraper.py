"""Abstract base class for all scrapers in the pipeline."""

import abc
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from scripts.utils.manifest import Manifest, ManifestEntry, compute_sha256, now_iso
from scripts.utils.rate_limiter import RateLimiter

USER_AGENT = "MCP-GeoKnowledge-Bot/1.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class DiscoveredItem:
    """An item discovered during the discovery phase."""

    url: str
    title: str = ""
    section_path: str = ""
    geology_subdomain: str = ""
    language: str = "en"
    extra: dict[str, Any] | None = None


class BaseScraper(abc.ABC):
    """Abstract scraper with manifest tracking, retry, and rate limiting.

    Subclasses must implement:
        - discover() → list of items to download
        - download_one(session, item) → (bytes, http_status)
        - local_path(item) → Path where the file should be saved
    """

    def __init__(
        self,
        manifest_path: Path,
        source_type: str,
        software_product: str,
        rate: float = 2.0,
        max_concurrent: int = 5,
    ) -> None:
        self.manifest = Manifest(manifest_path)
        self.source_type = source_type
        self.software_product = software_product
        self.rate_limiter = RateLimiter(rate=rate)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    async def discover(self, session: aiohttp.ClientSession) -> list[DiscoveredItem]:
        """Return all items that should be downloaded."""
        ...

    @abc.abstractmethod
    async def download_one(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download a single item. Return (content_bytes, http_status)."""
        ...

    @abc.abstractmethod
    def local_path(self, item: DiscoveredItem) -> Path:
        """Return the local file path where this item should be saved."""
        ...

    async def _download_with_retry(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> tuple[bytes, int]:
        """Download with retry and rate limiting."""

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            before_sleep=lambda rs: self.logger.warning(
                "Retry %d for %s", rs.attempt_number, item.url
            ),
        )
        async def _do() -> tuple[bytes, int]:
            async with self.rate_limiter:
                return await self.download_one(session, item)

        return await _do()

    async def _process_item(
        self, session: aiohttp.ClientSession, item: DiscoveredItem
    ) -> bool:
        """Download one item, save to disk, update manifest. Returns True on success."""
        async with self.semaphore:
            try:
                data, status = await self._download_with_retry(session, item)
            except Exception as e:
                self.logger.error("Failed to download %s: %s", item.url, e)
                return False

            dest = self.local_path(item)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

            entry = ManifestEntry(
                source_id=dest.stem,
                source_url=item.url,
                source_type=self.source_type,
                software_product=self.software_product,
                download_path=str(dest.relative_to(PROJECT_ROOT)),
                download_date=now_iso(),
                http_status=str(status),
                content_hash_sha256=compute_sha256(data),
                file_size_bytes=str(len(data)),
                language=item.language,
                section_path=item.section_path,
                geology_subdomain=item.geology_subdomain,
                title=item.title,
            )
            self.manifest.add_entry(entry)
            self.manifest.save()
            return True

    async def run(self, force: bool = False) -> None:
        """Main entry point: discover items, skip already-downloaded, download the rest."""
        timeout = aiohttp.ClientTimeout(total=60)
        connector = aiohttp.TCPConnector(limit=20, ssl=False)
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
                    "%d items remaining after filtering already-downloaded", len(items)
                )

            if not items:
                self.logger.info("Nothing to download.")
                return

            tasks = [self._process_item(session, item) for item in items]

            success = 0
            failed = 0
            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                result = await coro
                if result:
                    success += 1
                else:
                    failed += 1
                if i % 100 == 0 or i == len(tasks):
                    self.logger.info(
                        "Progress: %d/%d (success=%d, failed=%d)",
                        i,
                        len(tasks),
                        success,
                        failed,
                    )

            self.logger.info(
                "Done. Downloaded %d, failed %d, total in manifest: %d",
                success,
                failed,
                len(self.manifest),
            )

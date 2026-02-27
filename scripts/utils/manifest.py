"""CSV-based manifest for tracking downloaded and parsed sources.

Each scraper maintains its own manifest CSV. The manifest acts as a
checkpoint: re-running a scraper skips URLs already present (unless --force).
"""

import csv
import hashlib
import shutil
import tempfile
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


MANIFEST_COLUMNS = [
    "source_id",
    "source_url",
    "source_type",
    "software_product",
    "download_path",
    "download_date",
    "http_status",
    "content_hash_sha256",
    "file_size_bytes",
    "language",
    "section_path",
    "geology_subdomain",
    "title",
    "parsed",
    "parse_date",
    "chunk_count",
    "embedded",
]


@dataclass
class ManifestEntry:
    source_id: str = ""
    source_url: str = ""
    source_type: str = ""
    software_product: str = ""
    download_path: str = ""
    download_date: str = ""
    http_status: str = ""
    content_hash_sha256: str = ""
    file_size_bytes: str = ""
    language: str = ""
    section_path: str = ""
    geology_subdomain: str = ""
    title: str = ""
    parsed: str = "false"
    parse_date: str = ""
    chunk_count: str = "0"
    embedded: str = "false"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class Manifest:
    """Thread-safe CSV manifest with atomic writes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, ManifestEntry] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = ManifestEntry(**{k: row.get(k, "") for k in MANIFEST_COLUMNS})
                self._entries[entry.source_url] = entry

    def save(self) -> None:
        """Atomically write manifest to disk (write to temp, then rename)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".csv.tmp"
        )
        try:
            with open(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                for entry in self._entries.values():
                    writer.writerow(entry.to_dict())
            shutil.move(tmp_path, self._path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def has_url(self, url: str) -> bool:
        return url in self._entries

    def get(self, url: str) -> ManifestEntry | None:
        return self._entries.get(url)

    def add_entry(self, entry: ManifestEntry) -> None:
        self._entries[entry.source_url] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[ManifestEntry]:
        return iter(self._entries.values())

    @property
    def path(self) -> Path:
        return self._path


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

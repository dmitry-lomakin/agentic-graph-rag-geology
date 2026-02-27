"""Cross-platform discovery of external tools (LibreOffice, poppler-utils).

On Linux these are typically on PATH. On Windows they're installed to
well-known locations under Program Files. This module finds the correct
executable path for the current platform.
"""

import os
import shutil
import sys
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"

# Well-known Windows installation paths for LibreOffice
_LIBREOFFICE_WIN_CANDIDATES = [
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "LibreOffice" / "program" / "soffice.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "LibreOffice" / "program" / "soffice.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LibreOffice" / "program" / "soffice.exe",
]


def _find_on_path(name: str) -> str | None:
    """Find an executable on system PATH."""
    return shutil.which(name)


def find_libreoffice() -> str:
    """Find the LibreOffice executable.

    Returns:
        Path to LibreOffice/soffice executable.

    Raises:
        FileNotFoundError: If LibreOffice is not found.
    """
    # Try PATH first (works on Linux, and Windows if user added it)
    for cmd in ("libreoffice", "soffice"):
        found = _find_on_path(cmd)
        if found:
            return found

    # Windows: check well-known locations
    if _IS_WINDOWS:
        for candidate in _LIBREOFFICE_WIN_CANDIDATES:
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(
        "LibreOffice not found. Install it from https://www.libreoffice.org/ "
        "or add its program directory to PATH."
    )

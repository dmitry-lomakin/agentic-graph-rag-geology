"""Extract embedded images from PDF files using PyMuPDF (fitz).

Extracts images that are at least 100x100 pixels (filtering out icons and
decorative elements). Output is saved to the figures/ directory, mirroring
the raw-docs/ structure.

This runs as a separate pass from text parsing -- the `figures` CLI command
in run_parser.py calls this module.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_WIDTH = 100   # pixels
MIN_HEIGHT = 100  # pixels


def extract_figures_from_pdf(
    file_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Extract images from a PDF and save them as PNG files.

    Args:
        file_path: Path to the PDF file.
        output_dir: Base output directory (e.g. PROJECT_ROOT / "figures").
            Images are saved in a subdirectory matching the PDF stem.

    Returns:
        List of paths to extracted image files.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(file_path))
    extracted: list[Path] = []
    fig_dir = output_dir / file_path.stem
    fig_num = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception:
                logger.debug(
                    "Failed to extract image xref=%d from page %d of %s",
                    xref,
                    page_num + 1,
                    file_path.name,
                )
                continue

            width = base_image["width"]
            height = base_image["height"]

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                continue

            fig_num += 1
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / f"fig_{fig_num:03d}.png"

            # PyMuPDF may return JPEG or other formats; save as-is with .png extension
            # if it's already PNG, or convert via pixmap for other formats
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            if ext == "png":
                fig_path.write_bytes(image_bytes)
            else:
                # Convert to PNG via pixmap
                try:
                    pix = fitz.Pixmap(image_bytes)
                    if pix.n > 4:  # CMYK or other color space
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(str(fig_path))
                except Exception:
                    # Fall back: save with original extension
                    fig_path = fig_path.with_suffix(f".{ext}")
                    fig_path.write_bytes(image_bytes)

            extracted.append(fig_path)

    doc.close()

    if extracted:
        logger.info(
            "Extracted %d figures from %s -> %s",
            len(extracted),
            file_path.name,
            fig_dir,
        )
    else:
        logger.debug("No figures found in %s", file_path.name)

    return extracted

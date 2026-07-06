"""
image_utils.py — Image discovery, loading, and encoding utilities.

Central module for all image I/O so that no pipeline stage has to
re-implement file discovery or format validation.

Usage:
    from annotation_pipeline.common.utils.image_utils import (
        discover_images,
        load_image,
        encode_image_base64,
    )
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


# ===========================================================================
# Image Discovery
# ===========================================================================


def discover_images(
    directory: Path | str,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
) -> list[Path]:
    """
    Find all image files in *directory* matching the given extensions.

    Args:
        directory: Path to search for images (non-recursive).
        extensions: Tuple of allowed file suffixes (case-insensitive).

    Returns:
        Sorted list of image paths.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no images are found.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    ext_lower = {e.lower() for e in extensions}
    images = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in ext_lower
    )

    if not images:
        raise ValueError(
            f"No images found in {directory} with extensions {extensions}"
        )

    logger.info(
        "Discovered {} image(s) in '{}'.",
        len(images),
        directory,
    )
    return images


# ===========================================================================
# Image Loading
# ===========================================================================


def load_image(
    path: Path | str,
    color_mode: int = cv2.IMREAD_COLOR,
) -> np.ndarray:
    """
    Load an image from disk with validation.

    Args:
        path: Path to the image file.
        color_mode: OpenCV imread flag.  Defaults to BGR color.

    Returns:
        NumPy array (H, W, C) in BGR format.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV fails to decode the image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(path), color_mode)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    logger.debug(
        "Loaded image: {}  shape={}  dtype={}",
        path.name,
        image.shape,
        image.dtype,
    )
    return image


def load_image_rgb(path: Path | str) -> np.ndarray:
    """Load an image and convert BGR → RGB."""
    bgr = load_image(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_image_dimensions(path: Path | str) -> tuple[int, int]:
    """
    Return (width, height) of an image without fully decoding it.

    Uses OpenCV's imread with IMREAD_UNCHANGED for efficiency.
    """
    img = load_image(path)
    h, w = img.shape[:2]
    return w, h


# ===========================================================================
# Base64 Encoding
# ===========================================================================


def encode_image_base64(path: Path | str) -> str:
    """
    Read an image file and return its base64-encoded string.

    Used for Vision Language Model API payloads that accept
    inline base64 images.

    Args:
        path: Path to the image file.

    Returns:
        Base64-encoded string of the raw file bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    logger.debug(
        "Base64-encoded image: {} ({} chars)",
        path.name,
        len(encoded),
    )
    return encoded


def get_image_mime_type(
    path: Path | str,
) -> str:
    """
    Determine the MIME type of an image from its file extension.

    Parameters
    ----------
    path
        Path to the image file.

    Returns
    -------
    str
        MIME type corresponding to the file extension.
    """

    suffix = Path(path).suffix.lower()

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    return mime_map.get(
        suffix,
        "application/octet-stream",
    )

# ===========================================================================
    # Test Function
    # ===========================================================================

def test_image_utils() -> None:
    """
    Verify image discovery, loading, and encoding utilities.

    This standalone test validates the image utility functions using the
    configured input image directory.

    Run
    ---
    python -m annotation_pipeline.common.utils.image_utils
    """
    from annotation_pipeline.common.configs.settings import load_config

    logger.info("=" * 70)
    logger.info("Image Utilities Test")
    logger.info("=" * 70)

    config = load_config()

    # ------------------------------------------------------------------
    # Image Discovery
    # ------------------------------------------------------------------

    images = discover_images(
        config.paths.random_frames,
        config.pipeline.supported_image_extensions,
    )

    logger.info(
        "Discovered {} image(s) in '{}'.",
        len(images),
         config.paths.random_frames,
    )

    logger.info(
        "First image: '{}'.",
        images[0].name,
    )

    logger.info(
        "Last image: '{}'.",
        images[-1].name,
    )

    # ------------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------------

    img = load_image(images[0])

    logger.info(
        "Loaded image '{}' (shape={}, dtype={}).",
        images[0].name,
        img.shape,
        img.dtype,
    )

    img_rgb = load_image_rgb(images[0])

    logger.info(
        "RGB image shape: {}.",
        img_rgb.shape,
    )

    # ------------------------------------------------------------------
    # Image Dimensions
    # ------------------------------------------------------------------

    width, height = get_image_dimensions(images[0])

    logger.info(
        "Image dimensions: {} x {}.",
         width,
         height,
    )

    # ------------------------------------------------------------------
    # Base64 Encoding
    # ------------------------------------------------------------------

    encoded = encode_image_base64(images[0])

    logger.info(
        "Base64 length: {} characters.",
            len(encoded),
    )

    # ------------------------------------------------------------------
    # MIME Type
    # ------------------------------------------------------------------

    mime_type = get_image_mime_type(images[0])

    logger.info(
         "MIME type: '{}'.",
         mime_type,
    )

    # ------------------------------------------------------------------
    # Validate All Images
    # ------------------------------------------------------------------

    error_count = 0

    for image_path in images:

        try:
            load_image(image_path)

        except (FileNotFoundError, ValueError) as exc:

            logger.error(
                 "Failed to load '{}': {}",
                image_path.name,
                exc,
            )

            error_count += 1

    if error_count == 0:

        logger.info(
            "Successfully validated all {} image(s).",
            len(images),
        )

        logger.info("Image utilities test PASSED.")

    else:

        logger.error(
            "{} of {} image(s) failed validation.",
            error_count,
            len(images),
        )

        logger.error("Image utilities test FAILED.")

    logger.info("=" * 70)

# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":

    test_image_utils()
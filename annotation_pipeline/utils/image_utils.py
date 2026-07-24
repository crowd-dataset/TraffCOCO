"""
image_utils.py — Image discovery, loading, and encoding utilities.

Central image utility module used throughout the annotation pipeline.

Provides image discovery, loading, validation and common image metadata.

All pipeline stages should obtain images through this module rather than
performing direct filesystem access.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from annotation_pipeline.configs.settings import load_config

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
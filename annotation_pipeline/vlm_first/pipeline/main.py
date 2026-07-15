"""
main.py

Main entry point for the VLM-First pipeline.

Current Pipeline

Random Frame Download (if required)
        │
        ▼
Image Discovery
        │
        ▼
Scene Understanding
        │
        ▼
Pipeline Cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from custom_logger import CustomLogger
from logmod import logs

from annotation_pipeline.common.cache.pipeline_cache import PipelineCache
from annotation_pipeline.common.configs.settings import load_config
from annotation_pipeline.common.models.vlm.model_loader import (
    create_scene_model,
)
from annotation_pipeline.common.utils.image_utils import (
    discover_images,
)

from annotation_pipeline.vlm_first.pipeline.scene_understanding import (
    SceneUnderstandingEngine,
)

# Random frame downloader
from random_frame_sampler import get_random_frames_from_common_config

logger = CustomLogger(__name__)

# ============================================================================
# Random Frame Preparation
# ============================================================================


def has_images(
    image_dir: Path,
    extensions: tuple[str, ...],
) -> bool:
    """
    Check whether the configured random frame directory already
    contains supported images.
    """

    if not image_dir.exists():
        return False

    try:

        image_paths = discover_images(
            image_dir,
            extensions,
        )

        return len(image_paths) > 0

    except ValueError:

        return False


def prepare_random_frames(
    config,
) -> None:
    """
    Download random frames only if the random_frames directory is empty.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("Preparing Random Frames")
    logger.info("=" * 80)

    if has_images(
        config.paths.random_frames,
        config.pipeline.supported_image_extensions,
    ):

        logger.info(
            "Random frame directory already contains images."
        )

        logger.info(
            "Skipping download."
        )

        return

    logger.info(
        "No random frames found."
    )

    logger.info(
        "Downloading random frames..."
    )

    try:

        records, manifest = (
            get_random_frames_from_common_config()
        )

        logger.info(
            "Downloaded {} image(s).",
            len(records),
        )

        if manifest is not None:

            logger.info(
                "Manifest written to '{}'.",
                manifest,
            )

    except Exception as e:

        logger.error(
            "Random frame download failed: {}",
            e,
        )

        raise

# ============================================================================
# Main Pipeline
# ============================================================================


def main() -> None:
    """
    Execute the VLM-First pipeline.

    Current Pipeline

        Random Frame Download (if required)
                │
                ▼
        Image Discovery
                │
                ▼
        Scene Understanding
                │
                ▼
        Pipeline Cache
    """

    # ------------------------------------------------------------------
    # Command-line Arguments
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="VLM-First Pipeline",
    )

    parser.add_argument(
        "--images",
        type=int,
        default=1,
        help="Number of images to process (0 = all).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load Configuration
    # ------------------------------------------------------------------

    config = load_config()

    config.paths.ensure_output_dirs()

    logs(
        show_level=config.logging.level.lower(),
        save_level=config.logging.level.lower(),
        program_name="vlm_first_pipeline",
        path=config.paths.logs,
    )

    logger.info("=" * 80)
    logger.info("VLM-First Pipeline")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Prepare Random Frames
    # ------------------------------------------------------------------

    if config.pipeline.download_random_frames:

        prepare_random_frames(config)

    # ------------------------------------------------------------------
    # Discover Images
    # ------------------------------------------------------------------

    logger.info(
        "Searching for images in '{}'.",
        config.paths.random_frames,
    )

    image_paths = discover_images(
        config.paths.random_frames,
        config.pipeline.supported_image_extensions,
    )

    if not image_paths:

        logger.error(
            "No supported images found in '{}'.",
            config.paths.random_frames,
        )

        sys.exit(1)

    if args.images > 0:

        image_paths = image_paths[: args.images]

    logger.info(
        "Discovered {} image(s).",
        len(image_paths),
    )

    # ------------------------------------------------------------------
    # Initialize Pipeline Cache
    # ------------------------------------------------------------------

    logger.info(
        "Initializing Pipeline Cache."
    )

    cache = PipelineCache()

    # ------------------------------------------------------------------
    # Create Scene Understanding Model
    # ------------------------------------------------------------------

    logger.info(
        "Creating Scene Understanding model."
    )

    try:

        model = create_scene_model(config)

    except Exception:

        logger.error(
            "Failed to create Scene Understanding model."
        )

        raise

    # ------------------------------------------------------------------
    # Load Model
    # ------------------------------------------------------------------

    logger.info(
        "Loading model '{}'.",
        model.model_name,
    )

    model.load()

    logger.info(
        "Model successfully loaded."
    )

    # ------------------------------------------------------------------
    # Scene Understanding
    # ------------------------------------------------------------------

    failed_images = []

    if config.pipeline.run_scene_understanding:

        # --------------------------------------------------------------
        # Create Scene Understanding Model
        # --------------------------------------------------------------

        logger.info(
            "Creating Scene Understanding model."
        )

        try:

            model = create_scene_model(config)

        except Exception:

            logger.error(
                "Failed to create Scene Understanding model."
            )

            raise

        # --------------------------------------------------------------
        # Load Model
        # --------------------------------------------------------------

        logger.info(
            "Loading model '{}'.",
            model.model_name,
        )

        model.load()

        logger.info(
            "Model successfully loaded."
        )

        # --------------------------------------------------------------
        # Create Scene Understanding Engine
        # --------------------------------------------------------------

        engine = SceneUnderstandingEngine(
            config=config,
            model=model,
        )

        # --------------------------------------------------------------
        # Process Images
        # --------------------------------------------------------------

        try:

            for index, image_path in enumerate(
                image_paths,
                start=1,
            ):

                logger.info("")
                logger.info("=" * 80)
                logger.info(
                    "Processing image {}/{} : {}",
                    index,
                    len(image_paths),
                    image_path.name,
                )
                logger.info("=" * 80)

                try:

                    objects = engine.process_image(
                        image_path=image_path,
                        cache=cache,
                    )

                    logger.info(
                        "Detected {} traffic object(s).",
                        len(objects),
                    )

                    logger.debug(
                        "Parsed Scene Understanding JSON:\n{}",
                        json.dumps(
                            objects,
                            indent=4,
                            ensure_ascii=False,
                        ),
                    )

                except Exception as e:

                    logger.error(
                        "Failed to process '{}': {}",
                        image_path.name,
                        e,
                    )

                    failed_images.append(
                        image_path.name,
                    )

                    continue

        finally:

            logger.info(
                "Unloading Scene Understanding model."
            )

            model.unload()

    else:

        logger.info(
            "Skipping Scene Understanding."
        )

    # ------------------------------------------------------------------
    # Pipeline Complete
    # ------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 80)
    logger.info("VLM-First Pipeline Complete")
    logger.info("=" * 80)

    if failed_images:

        logger.warning(
            "{} image(s) failed during processing.",
            len(failed_images),
        )

        for image_name in failed_images:

            logger.warning(
                "    {}",
                image_name,
            )

    else:

        logger.info(
            "All images processed successfully."
        )


if __name__ == "__main__":

    main()
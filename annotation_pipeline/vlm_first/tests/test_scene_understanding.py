"""
test_scene_understanding.py

Development pipeline for incremental testing.

Currently tests the pipeline up to the Scene Understanding stage.

Pipeline

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

from custom_logger import CustomLogger
from logmod import logs

from annotation_pipeline.common.cache.pipeline_cache import PipelineCache
from annotation_pipeline.common.configs.settings import load_config
from annotation_pipeline.common.models.vlm.model_loader import (
    create_scene_model,
)
from annotation_pipeline.common.utils.image_utils import discover_images
from annotation_pipeline.vlm_first.pipeline.scene_understanding import (
    SceneUnderstandingEngine,
)

logger = CustomLogger(__name__)


def main() -> None:
    """
    Execute the Scene Understanding test pipeline.

    The current development pipeline performs:

        Image Discovery
                │
                ▼
        Scene Understanding
                │
                ▼
        Pipeline Cache

    This script is intended for incremental testing while new pipeline stages
    are being developed.
    """

    # ------------------------------------------------------------------
    # Command-line Arguments
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Scene Understanding Test Pipeline",
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
        program_name="scene_understanding",
        path=config.paths.logs,
    )

    logger.info("=" * 80)
    logger.info("Scene Understanding Test Pipeline")
    logger.info("=" * 80)

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
    # Create Scene Understanding Engine
    # ------------------------------------------------------------------

    engine = SceneUnderstandingEngine(
        config=config,
        model=model,
    )

    # ------------------------------------------------------------------
    # Process Images
    # ------------------------------------------------------------------

    failed_images = []

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

                failed_images.append(image_path.name)

                continue
    finally:

        logger.info(
            "Unloading Scene Understanding model."
        )

        model.unload()

    # ------------------------------------------------------------------
    # Pipeline Complete
    # ------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 80)
    logger.info("Scene Understanding Test Complete")
    logger.info("=" * 80)


if __name__ == "__main__":

    main()

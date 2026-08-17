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
import torch
import time

from custom_logger import CustomLogger
from logmod import logs

from annotation_pipeline.cache.pipeline_cache import PipelineCache
from annotation_pipeline.configs.settings import load_config
from annotation_pipeline.models.vlm.model_loader import (
    create_scene_model,
)
from annotation_pipeline.utils.image_utils import (
    discover_images,
)
from annotation_pipeline.utils.gpu_monitor import GPUMonitor

from annotation_pipeline.pipeline.scene_understanding import (
    SceneUnderstandingEngine,
)

from annotation_pipeline.pipeline.ontology import (
    OntologyEngine,
)

from annotation_pipeline.pipeline.locate_anything import (
    LocateAnythingEngine,
)

# Random frame downloader
from annotation_pipeline.pipeline.random_frame_sampler import get_random_frames_from_common_config

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
                │
                ▼
        Ontology Reasoning
                │
                ▼
        Pipeline Cache
    """

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
    config = load_config()

    pipeline_start = time.perf_counter()

    config.paths.ensure_output_dirs()

    logger.info("=" * 80)
    logger.info("VLM-First Pipeline")
    logger.info("=" * 80)


    # ------------------------------------------------------------------
    # Command-line Arguments
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="VLM-First Pipeline",
    )

    parser.add_argument(
        "--images",
        type=int,
        default=config.pipeline.num_images,
        help="Number of images to process (0 = all).",
    )

    args = parser.parse_args()

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

    monitor = GPUMonitor(
        output_dir=config.paths.outputs,
    )

    monitor.start()

    total_generation_time = 0.0

    total_generated_tokens = 0

    total_images = 0

    total_objects = 0
    total_reasoning_time = 0.0

    total_grounding_time = 0.0
    total_grounding_objects = 0


    # ------------------------------------------------------------------
    # Scene Understanding
    # ------------------------------------------------------------------

    failed_images: list[tuple[str, str, str]] = []

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

        BATCH_SIZE = config.pipeline.batch_size

        try:

            for batch_start in range(0, len(image_paths), BATCH_SIZE):

                batch = image_paths[
                    batch_start : batch_start + BATCH_SIZE
                ]

                logger.info("")
                logger.info("=" * 80)
                logger.info(
                    "Processing batch {}-{} of {} ({} image(s))",
                    batch_start + 1,
                    min(batch_start + len(batch), len(image_paths)),
                    len(image_paths),
                    len(batch),
                )
                logger.info(
                    "GPU BEFORE BATCH"
                )

                logger.info(
                    "Allocated : {} GB",
                    round(torch.cuda.memory_allocated() / 1024**3, 2),
                )

                logger.info(
                    "Reserved : {} GB",
                    round(torch.cuda.memory_reserved() / 1024**3, 2),
                )
                logger.info("=" * 80)

                try:

                    objects_per_image = engine.process_images(
                        image_paths=batch,
                        cache=cache,
                    )

                    total_generation_time += engine.last_generation_time

                    total_generated_tokens += engine.last_generated_tokens

                    total_images += len(batch)

                    for image_path, objects in zip(
                        batch,
                        objects_per_image,
                    ):

                        logger.info(
                            "Detected {} traffic object(s) in '{}'.",
                            len(objects),
                            image_path.name,
                        )

                        logger.debug(
                            "Parsed Scene Understanding JSON for '{}':\n{}",
                            image_path.name,
                            json.dumps(
                                objects,
                                indent=4,
                                ensure_ascii=False,
                            ),
                        )

                except Exception as e:

                    logger.error(
                        "Batch failed: {}",
                        e,
                    )

                    for image_path in batch:

                        failed_images.append(
                            (
                                image_path.name,
                                "Scene Understanding",
                                str(e),
                            )
                        )

        finally:

            logger.info(
                "Unloading Scene Understanding model."
            )

            

            model.unload()

            logger.info("")
            logger.info("=" * 80)
            logger.info("SCENE UNDERSTANDING SUMMARY")
            logger.info("=" * 80)

            logger.info(
                "Images Processed      : {}",
                total_images,
            )

            logger.info(
                "Total Generation Time : {:.2f} s",
                total_generation_time,
            )

            if total_images:

                logger.info(
                    "Average/Image        : {:.2f} s",
                    total_generation_time / total_images,
                )

                logger.info(
                    "Images/Hour          : {:.2f}",
                    (3600 * total_images)
                    / total_generation_time,
                )

            logger.info(
                "Generated Tokens      : {}",
                total_generated_tokens,
            )

            if total_generation_time:

                logger.info(
                    "Tokens/sec           : {:.2f}",
                    total_generated_tokens
                    / total_generation_time,
                )

            else:

                logger.info(
                    "Skipping Scene Understanding."
                )

    # ------------------------------------------------------------------
    # Ontology Reasoning
    # ------------------------------------------------------------------

    if config.pipeline.run_ontology_reasoning:

        logger.info("")
        logger.info("=" * 80)
        logger.info("ONTOLOGY REASONING")
        logger.info("=" * 80)

        ontology_engine = OntologyEngine(
            config=config,
        )

        total_reasoning_time = 0.0

        total_objects = 0

        for image_path in image_paths:

            try:

                results = ontology_engine.process_images(
                    image_name=image_path.name,
                    cache=cache,
                )

                total_reasoning_time += (
                    ontology_engine.last_reasoning_time
                )

                total_objects += (
                    ontology_engine.last_objects_processed
                )

                logger.info(
                    "Ontology completed for '{}'.",
                    image_path.name,
                )

                logger.info(
                    "Retrieved {} ontology prediction(s).",
                    len(results),
                )

            except Exception as e:

                logger.error(
                    "Ontology failed for '{}': {}",
                    image_path.name,
                    e,
                )

                failed_images.append(
                    (
                        image_path.name,
                        "Ontology Reasoning",
                        str(e),
                    )
                )

        logger.info("")
        logger.info("=" * 80)
        logger.info("ONTOLOGY SUMMARY")
        logger.info("=" * 80)

        logger.info(
            "Objects Processed : {}",
            total_objects,
        )

        logger.info(
            "Total Time : {:.2f} s",
            total_reasoning_time,
        )

        if total_objects:

            logger.info(
                "Average/Object : {:.4f} s",
                total_reasoning_time / total_objects,
            )

    # ------------------------------------------------------------------
    # Locate Anything Grounding
    # ------------------------------------------------------------------

    if config.pipeline.run_grounding:

        logger.info("")
        logger.info("=" * 80)
        logger.info("LOCATE ANYTHING GROUNDING")
        logger.info("=" * 80)

        try:

            logger.info(
                "Creating LocateAnythingEngine."
            )

            grounding_engine = LocateAnythingEngine(
                config=config,
            )

            logger.info(
                "LocateAnythingEngine successfully initialized."
            )

            grounding_start = time.perf_counter()

            grounding_results = grounding_engine.process_images(
                image_paths=image_paths,
                cache=cache,
            )

            total_grounding_time = (
                time.perf_counter()
                - grounding_start
            )

            total_grounding_objects = (
                grounding_engine.last_objects_processed
            )

            logger.info("")
            logger.info(
                "Locate Anything completed successfully."
            )

            logger.info(
                "Images Processed : {}",
                grounding_engine.last_images_processed,
            )

            logger.info(
                "Objects Processed : {}",
                grounding_engine.last_objects_processed,
            )

            logger.info(
                "Grounding Time : {:.2f} s",
                total_grounding_time,
            )

            if grounding_engine.last_images_processed:

                logger.info(
                    "Average/Image : {:.2f} s",
                    total_grounding_time
                    / grounding_engine.last_images_processed,
                )

        except Exception as e:

            logger.error(
                "Locate Anything grounding failed: {}",
                e,
            )

            failed_images.extend(
                (
                    image_path.name,
                    "Locate Anything",
                    str(e),
                )
                for image_path in image_paths
            )

        logger.info("")
        logger.info("=" * 80)
        logger.info("LOCATE ANYTHING SUMMARY")
        logger.info("=" * 80)

        logger.info(
            "Objects Processed : {}",
            total_grounding_objects,
        )

        logger.info(
            "Total Time : {:.2f} s",
            total_grounding_time,
        )

        if total_grounding_objects:

            logger.info(
                "Average/Object : {:.4f} s",
                total_grounding_time
                / total_grounding_objects,
            )

    # ------------------------------------------------------------------
    # Pipeline Complete
    # ------------------------------------------------------------------
    monitor.stop()
    
    monitor.save()
    pipeline_time = time.perf_counter() - pipeline_start

    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)

    logger.info(
        "Images Processed      : {}",
        total_images,
    )

    logger.info(
        "Objects Processed     : {}",
        total_objects,
    )

    logger.info(
        "Scene Time            : {:.2f} s",
        total_generation_time,
    )

    logger.info(
        "Ontology Time         : {:.2f} s",
        total_reasoning_time,
    )

    logger.info(
        "Total Pipeline Time   : {:.2f} s",
        pipeline_time,
    )

    if total_images:

        logger.info(
            "Average/Image        : {:.2f} s",
            pipeline_time / total_images,
        )

        logger.info(
            "Images/Hour          : {:.2f}",
            3600 * total_images / pipeline_time,
        )
    logger.info("")
    logger.info("=" * 80)
    logger.info("VLM-First Pipeline Complete")
    logger.info("=" * 80)

    if failed_images:

        logger.warning(
        "{} image(s) failed during processing.",
        len(failed_images),
    )

        for image_name, stage, reason in failed_images:

            logger.warning(
                "    {} | Stage: {} | Reason: {}",
                image_name,
                stage,
                reason,
            )

    else:

        logger.info(
            "All images processed successfully."
        )


if __name__ == "__main__":

    main()
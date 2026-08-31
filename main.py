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

from annotation_pipeline.cache.pipeline_cache import PipelineCache, PipelineStage
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
from annotation_pipeline.pipeline.annotation import AnnotationEngine
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

def run_annotation_stage(
    annotation_engine: AnnotationEngine,
    image_path: Path,
) -> tuple[list[dict], Path]:
    """
    Run the final annotation stage for one image.

    The AnnotationEngine:
        - finds the pipeline cache corresponding to image_path
        - reads the existing scene/ontology/grounding results
        - matches Locate Anything grounding prompts
        - resolves the final ontology labels
        - creates the final visualization on the original image

    Returns:
        final_detections
        visualization_path
    """

    final_detections = annotation_engine.annotate_image(
        image_path
    )

    visualization_path = (
        annotation_engine.output_dir
        / f"{image_path.stem}_annotated.png"
    )

    return (
        final_detections,
        visualization_path,
    )


def clear_grounding_for_retry(
    cache: PipelineCache,
    image_name: str,
) -> None:
    """
    Reset all grounding-stage data for one image before re-running
    Locate Anything.

    Grounding results are stored by APPENDING to a list per object
    (see PipelineCache.add_grounding_result), plus a top-level orphan
    bucket for unmatched detections. Simply re-running Locate Anything
    without clearing these first would pile the retry's detections on
    top of the original over-triggering results instead of replacing
    them -- which would guarantee the retry produces an even higher
    count than the first pass, defeating its entire purpose.

    This only resets in-memory grounding state. The on-disk pipeline
    cache JSON that AnnotationEngine reads is refreshed separately,
    when LocateAnythingEngine._process_batch calls
    cache.save_image_cache(..., stage="grounding") again during the
    retry pass (gated by config.pipeline.save_intermediate_cache,
    the same flag that produced the original cache files read by the
    first annotation pass).
    """

    image_cache = getattr(cache, "_cache", {}).get(image_name)

    if not isinstance(image_cache, dict):
        logger.warning(
            "clear_grounding_for_retry: no cache entry found for '{}'.",
            image_name,
        )
        return

    cleared_objects = 0

    for key, entry in image_cache.items():

        if key in ("grounding_detections", "_unmatched_grounding"):
            continue

        if not isinstance(entry, dict):
            continue

        if PipelineStage.GROUNDING.value in entry:
            entry[PipelineStage.GROUNDING.value] = []
            cleared_objects += 1

    # Top-level orphan buckets (see PipelineCache.add_unmatched_grounding_result
    # and the legacy "grounding_detections" list written by
    # LocateAnythingEngine._update_pipeline_cache).
    image_cache["grounding_detections"] = []
    image_cache["_unmatched_grounding"] = []

    logger.info(
        "Cleared grounding state for '{}' before retry "
        "({} scene object(s) reset).",
        image_name,
        cleared_objects,
    )


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

    annotation_engine = AnnotationEngine(
        pipeline_cache_dir=config.paths.pipeline_cache,
        output_dir=config.paths.annotations,
    )


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

    # ==============================================================
    # ANNOTATION
    # ==============================================================

    if config.pipeline.run_annotation:

        results: dict[str, dict[str, Any]] = {}
        stage_results: dict[str, dict[str, Any]] = {}
        processed_count = 0

        RETRY_THRESHOLD = 30

        for image_path in image_paths:

            processed_count += 1
            retry_attempted = False

            try:

                # ------------------------------------------------------
                # First annotation pass
                # ------------------------------------------------------

                final_detections, visualization_path = (
                    run_annotation_stage(
                        annotation_engine=annotation_engine,
                        image_path=image_path,
                    )
                )

                logger.info(
                    "Annotation completed for %s",
                    image_path.name,
                )

                logger.info(
                    "Initial final detections: %d",
                    len(final_detections),
                )

                # ------------------------------------------------------
                # Retry ONCE if detection count is abnormal
                # ------------------------------------------------------

                if (
                    len(final_detections) > RETRY_THRESHOLD
                    and not retry_attempted
                ):

                    retry_attempted = True

                    logger.warning(
                        "Detection count for '%s' is %d (> %d). "
                        "Running one Locate Anything retry.",
                        image_path.name,
                        len(final_detections),
                        RETRY_THRESHOLD,
                    )

                    if "grounding_engine" not in locals():

                        raise RuntimeError(
                            "Grounding engine is not available for retry."
                        )

                    # --------------------------------------------------
                    # Remove previous grounding results
                    # --------------------------------------------------

                    clear_grounding_for_retry(
                        cache=cache,
                        image_name=image_path.name,
                    )

                    # --------------------------------------------------
                    # Second Locate Anything pass
                    # --------------------------------------------------

                    grounding_engine.process_images(
                        image_paths=[image_path],
                        cache=cache,
                    )

                    logger.info(
                        "Locate Anything retry completed for '%s'.",
                        image_path.name,
                    )

                    # --------------------------------------------------
                    # Re-run annotation using retry results
                    # --------------------------------------------------

                    final_detections, visualization_path = (
                        run_annotation_stage(
                            annotation_engine=annotation_engine,
                            image_path=image_path,
                        )
                    )

                    logger.info(
                        "Retry annotation completed for %s",
                        image_path.name,
                    )

                    logger.info(
                        "Retry final detections: %d",
                        len(final_detections),
                    )

                    # --------------------------------------------------
                    # IMPORTANT:
                    # Do NOT retry again even if this is still >30.
                    # --------------------------------------------------

                    if len(final_detections) > RETRY_THRESHOLD:

                        logger.warning(
                            "Retry for '%s' still produced %d detections "
                            "(> %d). Accepting the retry result without "
                            "another retry.",
                            image_path.name,
                            len(final_detections),
                            RETRY_THRESHOLD,
                        )

                # ------------------------------------------------------
                # Final result
                # ------------------------------------------------------

                logger.info(
                    "Final detections: %d",
                    len(final_detections),
                )

                logger.info(
                    "Final visualization: %s",
                    visualization_path,
                )

                stage_results[image_path.name] = {
                    "status": "completed",
                    "detections": len(final_detections),
                    "visualization_path": str(
                        visualization_path
                    ),
                }

                results[image_path.name] = {
                    "annotation": stage_results[image_path.name],
                }

            except Exception as exc:

                logger.error(
                    "Annotation failed for %s",
                    image_path.name,
                )

                stage_results[image_path.name] = {
                    "status": "failed",
                    "error": str(exc),
                }

                results[image_path.name] = {
                    "annotation": stage_results[
                        image_path.name
                    ],
                }

                failed_images.append(
                    (
                        image_path.name,
                        "Annotation",
                        str(exc),
                    ),
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

    # --------------------------------------------------------------
    # Successful annotation outputs
    # --------------------------------------------------------------

    if processed_count > 0:
        logger.info("")
        logger.info("FINAL ANNOTATION OUTPUTS")
        logger.info("-" * 70)

        for image_path in image_paths:

            image_name = image_path.name

            result = results.get(
                image_name
            )

            if not isinstance(
                result,
                dict,
            ):
                continue

            annotation_result = result.get(
                "annotation"
            )

            if not isinstance(
                annotation_result,
                dict,
            ):
                continue

            if (
                annotation_result.get("status")
                != "completed"
            ):
                continue

            detections = annotation_result.get(
                "detections",
                0,
            )

            visualization_path = (
                annotation_result.get(
                    "visualization_path"
                )
            )

            logger.info(
                "  {}",
                image_name,
            )

            logger.info(
                "      Final detections : {}",
                detections,
            )

            if visualization_path:
                logger.info(
                    "      Visualization    : {}",
                    visualization_path,
                )

    # --------------------------------------------------------------
    # Failed images
    # --------------------------------------------------------------

    if failed_images:

        logger.warning("")
        logger.warning(
            "{} image(s) failed during processing.",
            len(failed_images),
        )

        for failure in failed_images:

            if isinstance(
                failure,
                dict,
            ):
                image_name = failure.get(
                    "image",
                    "unknown",
                )

                stage = failure.get(
                    "stage",
                    "unknown",
                )

                reason = failure.get(
                    "reason",
                    "unknown",
                )

                logger.warning(
                    "    {} | Stage: {} | Reason: {}",
                    image_name,
                    stage,
                    reason,
                )

            else:
                logger.warning(
                    "    {}",
                    failure,
                )

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    if annotation_engine is not None:
        logger.info("")
        logger.info(
            "Final visualization directory: {}",
            annotation_engine.output_dir,
        )

    logger.info("=" * 70)


if __name__ == "__main__":

    main()
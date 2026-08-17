"""
locate_anything.py

Locate Anything Grounding Engine.

This module implements the grounding stage of the VLM-First annotation
pipeline.

Pipeline
--------
Scene Cache
        │
        ▼
Prompt Builder
        │
        ▼
Locate Anything
        │
        ▼
Locate Anything Parser
        │
        ▼
Pipeline Cache
        │
        ▼
Visualization
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from custom_logger import CustomLogger

from annotation_pipeline.cache.pipeline_cache import (
    PipelineCache,
)

from annotation_pipeline.configs.settings import (
    PipelineConfig,
)

from annotation_pipeline.models.grounding.locate_anything import (
    LocateAnything,
)

from annotation_pipeline.pipeline.prompt_builder import (
    PromptBuilder,
)

from annotation_pipeline.models.grounding.locate_anything_parser import (
    LocateAnythingParser,
)

from annotation_pipeline.models.grounding.visualize_grounding import (
    GroundingVisualizer,
)

logger = CustomLogger(__name__)


class LocateAnythingEngine:
    """
    Locate Anything Grounding stage.

    Responsibilities
    ----------------

    • Read Scene Understanding results
    • Build prompts
    • Run Locate Anything
    • Parse model outputs
    • Update Pipeline Cache
    • Save intermediate outputs
    • Generate visualizations
    """

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:

        self.config = config

        # ----------------------------------------------------------
        # Components
        # ----------------------------------------------------------

        self.prompt_builder = PromptBuilder()

        self.model = LocateAnything(
            config=config,
        )

        self.parser = LocateAnythingParser()

        self.visualizer = GroundingVisualizer()

        logger.info(
            "Initialized LocateAnythingEngine."
        )
    
        # ----------------------------------------------------------
        # Benchmark Statistics
        # ----------------------------------------------------------

        self.last_grounding_time = 0.0

        self.last_images_processed = 0

        self.last_objects_processed = 0
        

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def process_images(
        self,
        image_paths: list[Path],
        cache: PipelineCache,
    ) -> None:
        """
        Run Locate Anything on all images.
        """

        if not image_paths:

            logger.warning(
                "No images supplied."
            )

            return

        logger.info(
            "Starting Locate Anything on {} image(s).",
            len(image_paths),
        )

        start_time = time.perf_counter()

        batch_size = self.config.pipeline.batch_size

        for batch_start in range(
            0,
            len(image_paths),
            batch_size,
        ):

            batch = image_paths[
                batch_start:
                batch_start + batch_size
            ]

            logger.info(
                "Processing batch ({}/{})",
                batch_start // batch_size + 1,
                (len(image_paths) + batch_size - 1)
                // batch_size,
            )

            self._process_batch(
                batch=batch,
                cache=cache,
            )

        self.last_grounding_time = (
            time.perf_counter() - start_time
        )

        self.last_images_processed = len(
            image_paths,
        )

        logger.info(
            "Locate Anything completed."
        )

        logger.info(
            "Images Processed : {}",
            self.last_images_processed,
        )

        logger.info(
            "Objects Processed : {}",
            self.last_objects_processed,
        )

        logger.info(
            "Grounding Time : {:.2f} s",
            self.last_grounding_time,
        )

        if self.last_images_processed:

            logger.info(
                "Average/Image : {:.2f} s",
                self.last_grounding_time
                / self.last_images_processed,
            )

    # ----------------------------------------------------------
    # Batch Processing
    # ----------------------------------------------------------

    def _process_batch(
        self,
        batch: list[Path],
        cache: PipelineCache,
    ) -> None:
        """
        Process one image batch.
        """

        images: list[Image.Image] = []

        prompts: list[str] = []

        image_names: list[str] = []

        image_paths: list[Path] = []

        scene_objects_batch: list[list[dict[str, Any]]] = []

        # ------------------------------------------------------
        # Load images + scene cache + build prompts
        # ------------------------------------------------------

        for image_path in batch:

            image_name = image_path.name

            logger.info(
                "Preparing '{}'.",
                image_name,
            )

            scene_file = (

                self.config.paths.scene_cache

                / "parsed"

                / f"{image_path.stem}.json"

            )

            if not scene_file.exists():

                logger.warning(
                    "Scene cache not found: '{}'.",
                    scene_file,
                )

                continue

            with scene_file.open(
                "r",
                encoding="utf-8",
            ) as file:

                scene_objects = json.load(
                    file,
                )

            if not scene_objects:

                logger.warning(
                    "No scene objects found for '{}'.",
                    image_name,
                )

                continue

            scene_objects.sort(
                key=lambda obj: obj["object_id"],
            )

            prompt = self.prompt_builder.build_prompt(
                scene_objects=scene_objects,
            )

            image = Image.open(
                image_path,
            ).convert(
                "RGB",
            )

            images.append(
                image,
            )

            prompts.append(
                prompt,
            )

            image_names.append(
                image_name,
            )

            image_paths.append(
                image_path,
            )

            scene_objects_batch.append(
                scene_objects,
            )

        if not images:

            logger.warning(
                "No valid images in this batch."
            )

            return

        # ------------------------------------------------------
        # Run Locate Anything
        # ------------------------------------------------------

        logger.info(
            "Calling NVIDIA LocateAnything batch runtime."
        )

        raw_outputs = self.model.generate_batch(

            images=images,

            prompts=prompts,

        )

        if len(raw_outputs) != len(images):

            raise RuntimeError(
                "LocateAnything returned an unexpected number "
                f"of outputs. Expected {len(images)}, "
                f"received {len(raw_outputs)}."
            )

        # ------------------------------------------------------
        # Process results
        # ------------------------------------------------------

        for (

            image,

            image_name,

            image_path,

            scene_objects,

            prompt,

            raw_output,

        ) in zip(

            images,

            image_names,

            image_paths,

            scene_objects_batch,

            prompts,

            raw_outputs,

        ):

            logger.info(
                "Processing Locate Anything output for '{}'.",
                image_name,
            )

            # --------------------------------------------------
            # Save Prompt
            # --------------------------------------------------

            self._save_output(

                category="prompts",

                image_name=image_name,

                data=prompt,

                extension=".txt",

            )

            # --------------------------------------------------
            # Save Raw Output
            # --------------------------------------------------

            self._save_output(

                category="raw",

                image_name=image_name,

                data=raw_output,

                extension=".txt",

            )

            # --------------------------------------------------
            # Parse
            # --------------------------------------------------

            detections = self.parser.parse(
                raw_output,
                scene_objects,
                image_size=image.size,
            )

            self.last_objects_processed += len(
                detections,
            )

            # --------------------------------------------------
            # Save Parsed Output
            # --------------------------------------------------

            self._save_output(

                category="parsed",

                image_name=image_name,

                data=detections,

                extension=".json",

            )

            # --------------------------------------------------
            # Update Pipeline Cache
            # --------------------------------------------------

            self._update_pipeline_cache(
                image_name=image_path.name,
                scene_objects=scene_objects,
                detections=detections,
                cache=cache,
            )

            # --------------------------------------------------
            # Visualization
            # --------------------------------------------------

            self.config.paths.visualizations.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (

                self.config.paths.visualizations

                / f"{Path(image_name).stem}.png"

            )

            self.visualizer.visualize(

                image=image,

                detections=detections,

                output_path=output_path,

            )

            # --------------------------------------------------
            # Save Pipeline Cache
            # --------------------------------------------------

            if self.config.pipeline.save_intermediate_cache:

                self.config.paths.pipeline_cache.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                cache.save_image_cache(

                    image_name=image_name,

                    directory=self.config.paths.pipeline_cache,

                    stage="grounding",

                )

            try:
                image.close()
            except Exception:
                pass

        logger.info(
            "Finished batch containing {} image(s).",
            len(images),
        )

    # ----------------------------------------------------------
    # Saving
    # ----------------------------------------------------------

    def _save_output(
        self,
        category: str,
        image_name: str,
        data: Any,
        extension: str,
    ) -> None:
        """
        Save any intermediate output.
        """

        output_dir = (
            self.config.paths.outputs
            / "locate_anything"
            / category
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_dir
            / f"{Path(image_name).stem}{extension}"
        )

        if extension == ".json":

            with output_file.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    data,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

        else:

            output_file.write_text(
                str(data),
                encoding="utf-8",
            )

    # ----------------------------------------------------------
    # Cache
    # ----------------------------------------------------------

    def _update_pipeline_cache(
        self,
        image_name: str,
        scene_objects: list[dict[str, Any]],
        detections: list[dict[str, Any]],
        cache: PipelineCache,
    ) -> None:
        """
        Store Locate Anything grounding results in the shared pipeline cache.

        Scene Understanding objects are inserted into the cache first so that
        grounding results can be attached to their existing object IDs.
        """

        logger.info(
            "Updating pipeline cache for '{}'.",
            image_name,
        )

        # ----------------------------------------------------------
        # Ensure Scene Understanding objects exist in cache
        # ----------------------------------------------------------

        if image_name not in cache._cache:

            logger.info(
                "Initializing cache with {} scene object(s) for '{}'.",
                len(scene_objects),
                image_name,
            )

            cache.add_scene_objects(
                image_name=image_name,
                objects=scene_objects,
            )

        # ----------------------------------------------------------
        # Store grounding results
        # ----------------------------------------------------------

        for detection in detections:

            object_id = detection.get(
                "object_id",
            )

            if object_id is None:

                logger.warning(
                    "Skipping grounding detection without object_id."
                )

                continue

            grounding_result = {
                key: value
                for key, value in detection.items()
                if key != "object_id"
            }

            try:

                cache.add_grounding_result(
                    image_name=image_name,
                    object_id=int(object_id),
                    grounding_result=grounding_result,
                )

            except KeyError as exc:

                logger.warning(
                    "Unable to cache grounding result for "
                    "object {} in '{}': {}",
                    object_id,
                    image_name,
                    exc,
                )

        logger.info(
            "Pipeline cache updated for '{}'.",
            image_name,
        )
    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    def print_statistics(
        self,
    ) -> None:
        """
        Print Locate Anything benchmark.
        """

        logger.info(
            "=" * 80,
        )

        logger.info(
            "Locate Anything Summary"
        )

        logger.info(
            "=" * 80,
        )

        logger.info(
            "Images Processed : {}",
            self.last_images_processed,
        )

        logger.info(
            "Objects Processed : {}",
            self.last_objects_processed,
        )

        logger.info(
            "Grounding Time : {:.2f} s",
            self.last_grounding_time,
        )

        if self.last_images_processed:

            logger.info(
                "Average/Image : {:.2f} s",
                self.last_grounding_time
                / self.last_images_processed,
            )

        if self.last_objects_processed:

            logger.info(
                "Average/Object : {:.4f} s",
                self.last_grounding_time
                / self.last_objects_processed,
            )

        logger.info(
            "=" * 80,
        )
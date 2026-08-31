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

        locate_anything_batch_size = self.config.pipeline.locate_anything_batch_size

        for batch_start in range(
            0,
            len(image_paths),
            locate_anything_batch_size,
        ):

            batch = image_paths[
                batch_start:
                batch_start + locate_anything_batch_size
            ]

            logger.info(
                "Processing batch ({}/{})",
                batch_start // locate_anything_batch_size + 1,
                (len(image_paths) + locate_anything_batch_size - 1)
                // locate_anything_batch_size,
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

    def _load_scene_objects_for_grounding(
        self,
        image_path: Path,
        cache: PipelineCache,
    ) -> list[dict[str, Any]]:
        """
        Load scene objects in the form the grounding stage actually needs.

        The ontology stage enriches the cached object entries with
        ``ontology_reasoning.prediction.grounding_prompt``. Prefer that over
        the raw scene-understanding JSON so Locate Anything receives the
        canonical grounding prompt instead of the generic object label.
        """

        image_name = image_path.name

        image_cache = getattr(cache, "_cache", {}).get(image_name, {})
        if isinstance(image_cache, dict) and image_cache:
            scene_objects: list[dict[str, Any]] = []

            for object_id, entry in sorted(
                image_cache.items(),
                key=lambda item: (
                    not str(item[0]).isdigit(),
                    int(item[0]) if str(item[0]).isdigit() else 0,
                ),
            ):
                if not isinstance(entry, dict):
                    continue

                scene_object = entry.get("scene_understanding")
                if not isinstance(scene_object, dict):
                    continue

                merged = dict(scene_object)
                merged["object_id"] = scene_object.get("object_id", object_id)

                ontology = entry.get("ontology_reasoning")
                if isinstance(ontology, dict):
                    prediction = ontology.get("prediction", {})
                    if isinstance(prediction, dict):
                        merged["ontology_reasoning"] = ontology
                        merged["class_id"] = prediction.get("class_id")
                        merged["class_name"] = prediction.get("class_name")
                        merged["grounding_prompt"] = (
                            prediction.get("grounding_prompt")
                            or merged.get("grounding_prompt")
                        )

                scene_objects.append(merged)

            if scene_objects:
                scene_objects.sort(key=lambda obj: int(obj.get("object_id", 0)))
                return scene_objects

        scene_file = (
            self.config.paths.scene_cache
            / "parsed"
            / f"{image_path.stem}.json"
        )

        if not scene_file.exists():
            return []

        with scene_file.open("r", encoding="utf-8") as file:
            scene_objects = json.load(file)

        if not isinstance(scene_objects, list):
            return []

        return [
            obj for obj in scene_objects if isinstance(obj, dict)
        ]

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

            scene_objects = self._load_scene_objects_for_grounding(
                image_path=image_path,
                cache=cache,
            )

            if not scene_objects:
                logger.warning(
                    "No scene objects found for '{}'.",
                    image_name,
                )
                continue

            scene_objects.sort(
                key=lambda obj: int(obj.get("object_id", 0)),
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

            # --------------------------------------------------
            # Resolve None object_ids using same-label propagation.
            #
            # This was previously defined but never invoked, which
            # meant every detection past the first per label stayed
            # unresolved and fell into the top-level orphan bucket
            # instead of being appended alongside its siblings under
            # the correct object_id.
            # --------------------------------------------------

            detections = self._resolve_missing_object_ids(
                detections,
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

    def _resolve_missing_object_ids(
        self,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Resolve Locate Anything detections whose object_id is None.

        Locate Anything may return multiple detections with the same
        object_name, while only some of them may have an object_id.

        If a label has an existing valid integer object_id, all detections
        with that same normalized label and object_id=None inherit that ID.

        Example
        -------
        bus -> 12
        bus -> None
        bus -> None

        becomes

        bus -> 12
        bus -> 12
        bus -> 12

        A label is NOT resolved if no valid integer ID already exists
        for that label.

        If multiple different integer IDs exist for the same label,
        the None detections remain unresolved because the label alone
        cannot determine which object they belong to.
        """

        if not detections:
            return detections

        def normalize_label(value: Any) -> str:
            if value is None:
                return ""

            label = str(value).strip().lower()

            # Normalize common formatting differences.
            label = label.replace("_", " ")
            label = label.replace("-", " ")

            # Collapse repeated whitespace.
            label = " ".join(label.split())

            return label

        # --------------------------------------------------------------
        # First pass:
        # Build label -> valid integer object IDs
        # --------------------------------------------------------------

        label_to_ids: dict[str, set[int]] = {}

        for detection in detections:

            object_id = detection.get("object_id")

            # Only existing integer IDs are authoritative.
            #
            # bool is technically an int subclass, so explicitly reject it.
            if (
                isinstance(object_id, int)
                and not isinstance(object_id, bool)
            ):
                label = normalize_label(
                    detection.get("object_name")
                )

                if not label:
                    continue

                label_to_ids.setdefault(
                    label,
                    set(),
                ).add(object_id)

        # --------------------------------------------------------------
        # Second pass:
        # Resolve object_id=None using the authoritative ID
        # for the same label.
        # --------------------------------------------------------------

        resolved_count = 0

        for detection in detections:

            if detection.get("object_id") is not None:
                continue

            label = normalize_label(
                detection.get("object_name")
            )

            if not label:
                continue

            candidate_ids = label_to_ids.get(
                label,
                set(),
            )

            # Exactly one existing ID means the association is
            # deterministic.
            if len(candidate_ids) == 1:

                resolved_id = next(
                    iter(candidate_ids)
                )

                detection["object_id"] = resolved_id

                resolved_count += 1

                logger.debug(
                    "Resolved missing object_id: "
                    "'{}' -> {}",
                    detection.get("object_name"),
                    resolved_id,
                )

            # Multiple IDs for the same label are ambiguous.
            # Do NOT arbitrarily choose one.
            elif len(candidate_ids) > 1:

                logger.warning(
                    "Could not resolve object_id for '{}' "
                    "because multiple IDs exist: {}",
                    detection.get("object_name"),
                    sorted(candidate_ids),
                )

        logger.info(
            "Resolved {} Locate Anything detection(s) "
            "with previously missing object_id.",
            resolved_count,
        )

        return detections

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

        IMPORTANT:
        Detections with a valid (possibly just-resolved) integer
        object_id are appended per-object via add_grounding_result,
        which now appends to a list rather than overwriting a single
        dict -- so multiple boxes sharing one object_id are all kept.

        Detections that still have object_id=None after
        _resolve_missing_object_ids (genuinely no scene object to
        attach to) are stored via add_unmatched_grounding_result
        rather than a bare top-level list, so annotation.py can find
        them through a single, well-defined path instead of relying
        on an early-return key lookup that silently hid every
        per-object detection stored alongside it.
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

        matched_count = 0
        unmatched_count = 0
        failed_count = 0

        for detection in detections:

            object_id = detection.get(
                "object_id",
            )

            grounding_result = {
                key: value
                for key, value in detection.items()
                if key != "object_id"
            }

            if object_id is not None:

                try:
                    cache.add_grounding_result(
                        image_name=image_name,
                        object_id=int(object_id),
                        grounding_result=grounding_result,
                    )
                    matched_count += 1

                except KeyError as exc:
                    logger.warning(
                        "Unable to cache grounding result for "
                        "object {} in '{}': {}",
                        object_id,
                        image_name,
                        exc,
                    )
                    failed_count += 1

            else:
                cache.add_unmatched_grounding_result(
                    image_name=image_name,
                    grounding_result=grounding_result,
                )
                unmatched_count += 1

                logger.info(
                    "Preserved unmatched grounding detection "
                    "(no object_id) for '{}'.",
                    image_name,
                )

        logger.info(
            "Pipeline cache updated for '{}': "
            "{} matched, {} unmatched, {} failed to cache.",
            image_name,
            matched_count,
            unmatched_count,
            failed_count,
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
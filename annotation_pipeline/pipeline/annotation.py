"""
Final annotation stage.

The annotation stage consumes the pipeline cache belonging to the
current image and produces the final ontology-resolved detections.

Pipeline:

    pipeline_cache/<image>.json
            |
            +-- scene objects
            |       |
            |       +-- ontology prediction
            |       +-- grounding_prompt
            |
            +-- Locate Anything results
                    |
                    +-- grounding prompt / ref
                    +-- bbox
            |
            v
        match grounding result
            |
            v
        final ontology label
            |
            v
        visualization on ORIGINAL image
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from annotation_pipeline.pipeline.annotation_visualizer import (
    AnnotationVisualizer,
)

logger = logging.getLogger(__name__)


class AnnotationEngine:
    """
    Build final annotations from the pipeline cache for an image.

    This stage does NOT rerun:
        - Scene Understanding
        - Ontology reasoning
        - Prompt building
        - Locate Anything

    It only consumes their outputs already stored in the
    corresponding pipeline cache.
    """

    def __init__(
        self,
        pipeline_cache_dir: str | Path,
        output_dir: str | Path,
        visualizer: AnnotationVisualizer | None = None,
    ) -> None:
        self.pipeline_cache_dir = Path(
            pipeline_cache_dir
        )

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.visualizer = (
            visualizer
            if visualizer is not None
            else AnnotationVisualizer()
        )

    # ================================================================
    # PUBLIC API
    # ================================================================

    def annotate_image(
        self,
        image_path: str | Path,
    ) -> list[dict[str, Any]]:
        """
        Annotate one image using its corresponding pipeline cache.

        Example:

            random_frames/
                0bt44qohUMg_1081.png

        automatically maps to:

            pipeline_cache/
                0bt44qohUMg_1081.json
        """

        image_path = Path(image_path)

        cache_path = self._find_pipeline_cache(
            image_path
        )

        logger.info(
            "Annotation cache: %s",
            cache_path,
        )

        cache = self._load_cache(
            cache_path
        )

        scene_objects = self._get_scene_objects(
            cache
        )

        if not scene_objects:
            raise ValueError(
                f"No scene objects found in {cache_path}"
            )

        grounding_detections = (
            self._get_grounding_detections(
                cache
            )
        )

        if not grounding_detections:
            raise ValueError(
                f"No Locate Anything detections found "
                f"in {cache_path}"
            )

        logger.info(
            "Scene objects: %d",
            len(scene_objects),
        )

        logger.info(
            "Grounding detections: %d",
            len(grounding_detections),
        )

        final_detections = (
            self._build_final_detections(
                scene_objects=scene_objects,
                grounding_detections=grounding_detections,
            )
        )

        logger.info(
            "Final detections: %d",
            len(final_detections),
        )

        # ------------------------------------------------------------
        # Final visualization is ALWAYS done on the original image.
        # ------------------------------------------------------------

        output_path = (
            self.output_dir
            / f"{image_path.stem}_annotated.png"
        )

        self.visualizer.visualize(
            image_path=image_path,
            detections=final_detections,
            output_path=output_path,
        )

        logger.info(
            "Final visualization written to: %s",
            output_path,
        )

        return final_detections

    def annotate_images(
        self,
        image_paths: list[str | Path],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Annotate multiple images.

        Each image independently resolves its own pipeline cache.
        """

        results: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for image_path in image_paths:

            image_path = Path(
                image_path
            )

            try:
                results[
                    image_path.name
                ] = self.annotate_image(
                    image_path
                )

            except Exception:
                logger.exception(
                    "Annotation failed for %s",
                    image_path.name,
                )

        return results

    # ================================================================
    # PIPELINE CACHE
    # ================================================================

    def _find_pipeline_cache(
        self,
        image_path: Path,
    ) -> Path:
        """
        Find the pipeline cache belonging to the image.

        Matching is based on the image filename stem.
        """

        cache_path = (
            self.pipeline_cache_dir
            / f"{image_path.stem}.json"
        )

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Pipeline cache not found for "
                f"{image_path.name}.\n"
                f"Expected: {cache_path}"
            )

        return cache_path

    @staticmethod
    def _load_cache(
        cache_path: Path,
    ) -> dict[str, Any]:
        """Load one pipeline-cache JSON file."""

        with cache_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                f"Pipeline cache must contain a JSON object: "
                f"{cache_path}"
            )

        return data

    # ================================================================
    # SCENE OBJECT EXTRACTION
    # ================================================================

    @staticmethod
    def _get_scene_objects(
        cache: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract scene objects from the pipeline cache.

        The live cache is object-keyed by image object id, so this method
        supports both the canonical top-level array structure and the
        already-written object-keyed nested structure.
        """

        objects = cache.get("scene_objects")
        if isinstance(objects, list):
            return [obj for obj in objects if isinstance(obj, dict)]

        scene = cache.get("scene")
        if isinstance(scene, dict):
            objects = scene.get("objects")
            if isinstance(objects, list):
                return [obj for obj in objects if isinstance(obj, dict)]

        objects = cache.get("objects")
        if isinstance(objects, list):
            return [obj for obj in objects if isinstance(obj, dict)]

        flattened: list[dict[str, Any]] = []
        for key, entry in cache.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            if key == "grounding_detections":
                continue
            if not isinstance(entry, dict):
                continue

            scene_obj = entry.get("scene_understanding")
            if not isinstance(scene_obj, dict):
                continue

            flat = {"object_id": entry.get("object_id", key), **scene_obj}
            ontology = entry.get("ontology_reasoning")
            if isinstance(ontology, dict):
                flat["ontology_reasoning"] = ontology
                prediction = ontology.get("prediction", {})
                if isinstance(prediction, dict):
                    flat["class_id"] = prediction.get("class_id")
                    flat["class_name"] = prediction.get("class_name")
                    flat["score"] = prediction.get("score")
                    flat["grounding_prompt"] = prediction.get("grounding_prompt")
                    flat["ontology"] = prediction
            grounding = entry.get("grounding")
            if isinstance(grounding, (dict, list)):
                flat["grounding"] = grounding
            flattened.append(flat)

        return flattened

    # ================================================================
    # GROUNDING EXTRACTION
    # ================================================================

    @staticmethod
    def _bbox_iou(
        bbox_a: list[float],
        bbox_b: list[float],
    ) -> float:
        """Calculate IoU between two bounding boxes."""

        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (
            (ix2 - ix1) *
            (iy2 - iy1)
        )

        area_a = (
            (ax2 - ax1) *
            (ay2 - ay1)
        )

        area_b = (
            (bx2 - bx1) *
            (by2 - by1)
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _remove_unreadable_duplicates(
        self,
        grounding_detections: list[dict[str, Any]],
        scene_object_by_id: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Remove an unreadable_traffic_sign detection when the same physical
        sign is also detected with a specific traffic-sign class.

        A detection is considered the same physical object when its bbox
        has high IoU with another detection.

        Specific ontology classification always wins over the generic
        unreadable_traffic_sign classification.
        """

        unreadable_indices: set[int] = set()

        for i, detection in enumerate(grounding_detections):

            object_id = self._coerce_object_id(
                detection.get("object_id")
            )

            if object_id is None:
                continue

            scene_object = scene_object_by_id.get(
                object_id
            )

            if scene_object is None:
                continue

            class_name = self._get_class_name(
                scene_object
            )

            if (
                class_name
                and self._normalize_grounding_label(class_name)
                == "unreadable traffic sign"
            ):
                unreadable_indices.add(i)

        to_remove: set[int] = set()

        for unreadable_index in unreadable_indices:

            unreadable = grounding_detections[
                unreadable_index
            ]

            unreadable_bbox = self._validate_bbox(
                unreadable.get("bbox")
            )

            if unreadable_bbox is None:
                continue

            for other_index, other in enumerate(
                grounding_detections
            ):

                if other_index == unreadable_index:
                    continue

                if other_index in unreadable_indices:
                    continue

                other_bbox = self._validate_bbox(
                    other.get("bbox")
                )

                if other_bbox is None:
                    continue

                iou = self._bbox_iou(
                    unreadable_bbox,
                    other_bbox,
                )

                if iou < 0.5:
                    continue

                other_id = self._coerce_object_id(
                    other.get("object_id")
                )

                if other_id is None:
                    continue

                other_scene_object = (
                    scene_object_by_id.get(
                        other_id
                    )
                )

                if other_scene_object is None:
                    continue

                other_class = self._get_class_name(
                    other_scene_object
                )

                if not other_class:
                    continue

                normalized_other_class = (
                    self._normalize_grounding_label(
                        other_class
                    )
                )

                if (
                    normalized_other_class
                    != "unreadable traffic sign"
                ):
                    logger.info(
                        "Removing duplicate unreadable "
                        "traffic-sign detection because "
                        "specific class '%s' overlaps it "
                        "(IoU=%.2f).",
                        other_class,
                        iou,
                    )

                    to_remove.add(
                        unreadable_index
                    )

                    break

        return [
            detection
            for index, detection in enumerate(
                grounding_detections
            )
            if index not in to_remove
        ]

    @staticmethod
    def _get_grounding_detections(
        cache: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Extract Locate Anything detections from the cache.

        IMPORTANT — this replaces a prior version that used an
        early-return priority chain. That was the root cause of valid
        detections (e.g. bus/truck/streetlight) silently vanishing:
        locate_anything.py writes detections to TWO different places
        depending on whether they had a resolvable object_id --

            - detections WITH a valid object_id are stored per-object,
              nested under cache[<object_id>]["grounding"]  (a list)

            - detections that never resolved to any object_id are
              stored in a single top-level list at
              cache["grounding_detections"]

        The previous implementation returned as soon as it found
        cache["grounding_detections"], and never reached the
        per-object loop below it -- so every valid-ID detection was
        silently invisible to annotation.py, even though it was safely
        sitting in the cache the whole time.

        This version MERGES both sources instead of prioritizing one
        over the other. Legacy single-key formats (top-level
        "grounding"/"detections" arrays from older cache writers) are
        supported as a fallback only when nothing else is found.
        """

        merged: list[dict[str, Any]] = []

        # ------------------------------------------------------------
        # Path A: per-object grounding entries.
        # ------------------------------------------------------------

        for key, entry in cache.items():

            if key == "grounding_detections":
                continue
            if isinstance(key, str) and key.startswith("_"):
                continue
            if not isinstance(entry, dict):
                continue

            grounding_info = entry.get("grounding")

            # Current format: list of detections.
            if isinstance(grounding_info, list):
                for det in grounding_info:
                    if isinstance(det, dict) and isinstance(
                        det.get("bbox"), (list, tuple)
                    ):
                        detection = dict(det)
                        detection.setdefault(
                            "object_id", entry.get("object_id", key)
                        )
                        merged.append(detection)

            # Back-compat: older caches stored a single dict here.
            elif isinstance(grounding_info, dict):
                if isinstance(grounding_info.get("bbox"), (list, tuple)):
                    detection = dict(grounding_info)
                    detection.setdefault(
                        "object_id", entry.get("object_id", key)
                    )
                    merged.append(detection)

        # ------------------------------------------------------------
        # Path B: top-level orphan list (object_id=None detections
        # that locate_anything.py could not resolve to a scene object).
        # ------------------------------------------------------------

        orphans = cache.get("grounding_detections")
        if isinstance(orphans, list):
            merged.extend(
                dict(d) for d in orphans if isinstance(d, dict)
            )

        # ------------------------------------------------------------
        # Path C: genuinely-unmatched detections stored separately
        # (see PipelineCache.add_unmatched_grounding_result).
        # ------------------------------------------------------------

        unmatched = cache.get("_unmatched_grounding")
        if isinstance(unmatched, list):
            merged.extend(
                dict(d) for d in unmatched if isinstance(d, dict)
            )

        if merged:
            return merged

        # ------------------------------------------------------------
        # Legacy fallback formats (only if nothing else matched).
        # ------------------------------------------------------------

        grounding = cache.get("grounding")
        if isinstance(grounding, dict):
            detections = grounding.get("detections")
            if isinstance(detections, list):
                return [d for d in detections if isinstance(d, dict)]

        detections = cache.get("detections")
        if isinstance(detections, list):
            return [d for d in detections if isinstance(d, dict)]

        return []

    def _normalize_grounding_label(
        self,
        label: Any,
    ) -> str:
        """
        Normalize a Locate Anything grounding label for
        identity propagation.

        This is intentionally conservative. It should normalize
        formatting noise, not change the semantic meaning.
        """

        if label is None:
            return ""

        label = str(label).strip().lower()

        # Remove common Locate Anything formatting noise.
        label = label.replace("_", " ")
        label = label.replace("-", " ")

        # Remove numeric reference wrappers such as:
        # <948> marking
        # 948 marking
        import re

        label = re.sub(
            r"<\s*\d+\s*>",
            "",
            label,
        )

        label = re.sub(
            r"^\s*\d+\s+",
            "",
            label,
        )

        # Collapse whitespace.
        label = re.sub(
            r"\s+",
            " ",
            label,
        ).strip()

        return label

    @staticmethod
    def _coerce_object_id(value: Any) -> int | None:
        """
        Normalize object_id to a canonical integer.

        Scene/grounding entries can carry object_id as:
          - an int (canonical)
          - a numeric string (e.g. "3"), when it was recovered from a
            JSON object key in the flattened pipeline-cache format
          - None / non-numeric (unresolved)

        Booleans are explicitly rejected: bool is a subclass of int in
        Python, so isinstance(True, int) is True and would otherwise
        silently pass through as object_id=1.
        """
        if isinstance(value, bool):
            return None

        if isinstance(value, int):
            return value

        if isinstance(value, str):
            value = value.strip()

            if value.lstrip("-").isdigit():
                try:
                    return int(value)
                except ValueError:
                    pass

        return None

    # ================================================================
    # FINAL DETECTIONS
    # ================================================================

    def _build_final_detections(
        self,
        scene_objects: list[dict[str, Any]],
        grounding_detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert Locate Anything detections into final ontology annotations.

        Rules:
        - Locate Anything supplies bbox/localization.
        - Scene/ontology cache supplies canonical object_id.
        - Ontology supplies canonical class_id and class_name.
        - A None grounding object_id must be resolved from the grounding
          label when a valid canonical object_id exists for that label.
        - Multiple valid detections with the same label/object_id are
          valid and must NOT be deduplicated.
        """

        # ------------------------------------------------------------
        # 1. Normalize IDs.
        # ------------------------------------------------------------

        normalized_scene_objects: list[dict[str, Any]] = []

        for obj in scene_objects:
            if not isinstance(obj, dict):
                continue

            normalized = dict(obj)
            normalized["object_id"] = self._coerce_object_id(
                obj.get("object_id")
            )
            normalized_scene_objects.append(normalized)

        scene_objects = normalized_scene_objects

        normalized_grounding: list[dict[str, Any]] = []

        for detection in grounding_detections:
            if not isinstance(detection, dict):
                continue

            normalized = dict(detection)
            normalized["object_id"] = self._coerce_object_id(
                detection.get("object_id")
            )
            normalized_grounding.append(normalized)

        grounding_detections = normalized_grounding

        # ------------------------------------------------------------
        # 2. Build canonical label -> object_id mapping.
        #
        # IMPORTANT:
        # Only IDs coming from the scene/ontology cache are canonical.
        # Locate Anything's raw label is never treated as the ontology
        # class name.
        # ------------------------------------------------------------

        valid_ids_by_label: dict[str, int] = {}

        for scene_object in scene_objects:

            object_id = self._coerce_object_id(
                scene_object.get("object_id")
            )

            if object_id is None:
                continue

            labels = [
                self._get_grounding_prompt(scene_object),
                self._get_class_name(scene_object),
                self._get_final_label(scene_object),
                scene_object.get("observed_object"),
            ]

            for label in labels:
                normalized_label = self._normalize_grounding_label(
                    label
                )

                if normalized_label:
                    valid_ids_by_label.setdefault(
                        normalized_label,
                        object_id,
                    )

        logger.debug(
            "Canonical grounding label -> object_id map: %s",
            valid_ids_by_label,
        )

        # ------------------------------------------------------------
        # 3. Resolve None IDs from an existing canonical ID.
        #
        # Example:
        #
        #   ID 3 | car
        #   IDNone | car
        #   IDNone | car
        #
        # becomes:
        #
        #   ID 3 | car
        #   ID 3 | car
        #   ID 3 | car
        #
        # Every bbox is preserved.
        # ------------------------------------------------------------

        for grounding in grounding_detections:

            grounding_id = self._coerce_object_id(
                grounding.get("object_id")
            )

            if grounding_id is not None:
                grounding["object_id"] = grounding_id
                continue

            raw_label = (
                grounding.get("object_name")
                or grounding.get("ref")
                or grounding.get("grounding_prompt")
            )

            normalized_label = self._normalize_grounding_label(
                raw_label
            )

            resolved_id = valid_ids_by_label.get(
                normalized_label
            )

            if resolved_id is not None:

                grounding["object_id"] = resolved_id

                logger.debug(
                    "Resolved None grounding ID: "
                    "label=%r -> object_id=%d",
                    raw_label,
                    resolved_id,
                )

        # ------------------------------------------------------------
        # 4. Build direct object_id lookup.
        # ------------------------------------------------------------

        scene_object_by_id: dict[int, dict[str, Any]] = {}

        for scene_object in scene_objects:

            object_id = self._coerce_object_id(
                scene_object.get("object_id")
            )

            if object_id is None:
                continue

            scene_object_by_id.setdefault(
                object_id,
                scene_object,
            )

            grounding_detections = self._remove_unreadable_duplicates(
                grounding_detections=grounding_detections,
                scene_object_by_id=scene_object_by_id,
            )

        # ------------------------------------------------------------
        # 5. Convert every valid grounding bbox.
        #
        # DO NOT use used_scene_indices here.
        #
        # Multiple Locate Anything boxes may legitimately correspond
        # to the same canonical object/class.
        # ------------------------------------------------------------

        final_detections: list[dict[str, Any]] = []

        dropped_invalid_bbox = 0

        for grounding in grounding_detections:

            bbox = self._validate_bbox(
                grounding.get("bbox")
            )

            if bbox is None:
                dropped_invalid_bbox += 1
                logger.debug(
                    "Skipping invalid grounding bbox: %s",
                    grounding.get("bbox"),
                )
                continue

            grounding_object_id = self._coerce_object_id(
                grounding.get("object_id")
            )

            scene_object = None

            # --------------------------------------------------------
            # First priority: canonical resolved object_id.
            # --------------------------------------------------------

            if grounding_object_id is not None:
                scene_object = scene_object_by_id.get(
                    grounding_object_id
                )

            # --------------------------------------------------------
            # Second priority: semantic matching.
            #
            # This is only a fallback for genuinely unresolved IDs.
            # --------------------------------------------------------

            if scene_object is None:

                match = self._match_grounding(
                    grounding=grounding,
                    scene_objects=scene_objects,
                    used_scene_indices=set(),
                )

                if match is not None:
                    _, scene_object = match

            # --------------------------------------------------------
            # Could not associate with ontology object.
            #
            # Preserve bbox instead of silently deleting it.
            # --------------------------------------------------------

            if scene_object is None:

                logger.warning(
                    "Could not associate Locate Anything detection "
                    "with an ontology scene object; preserving bbox: "
                    "object_name=%s object_id=%s bbox=%s",
                    grounding.get("object_name")
                    or grounding.get("ref"),
                    grounding_object_id,
                    bbox,
                )

                final_detections.append(
                    {
                        "object_id": grounding_object_id,
                        "bbox": bbox,
                        "final_label": None,
                        "class_name": None,
                        "class_id": None,
                        "grounding_prompt": (
                            grounding.get("grounding_prompt")
                        ),
                        "grounding_label": (
                            grounding.get("object_name")
                            or grounding.get("ref")
                        ),
                    }
                )

                continue

            # --------------------------------------------------------
            # Canonical ontology identity.
            # --------------------------------------------------------

            canonical_object_id = self._coerce_object_id(
                scene_object.get("object_id")
            )

            class_name = self._get_class_name(
                scene_object
            )

            final_label = self._get_final_label(
                scene_object
            )

            class_id = self._get_class_id(
                scene_object
            )

            grounding_prompt = self._get_grounding_prompt(
                scene_object
            )

            # --------------------------------------------------------
            # IMPORTANT:
            # Use ontology values, never Locate Anything's raw label.
            # --------------------------------------------------------

            final_detections.append(
                {
                    "object_id": canonical_object_id,
                    "bbox": bbox,
                    "final_label": final_label,
                    "class_name": class_name,
                    "class_id": class_id,
                    "grounding_prompt": grounding_prompt,
                    "grounding_label": (
                        grounding.get("object_name")
                        or grounding.get("ref")
                    ),
                }
            )

            logger.debug(
                "Final detection resolved: "
                "grounding=%r -> object_id=%s, "
                "class_name=%r, class_id=%r, bbox=%s",
                grounding.get("object_name")
                or grounding.get("ref"),
                canonical_object_id,
                class_name,
                class_id,
                bbox,
            )

        resolved_count = sum(
            1 for d in final_detections if d.get("class_name")
        )
        unresolved_count = len(final_detections) - resolved_count

        logger.info(
            "Final annotation resolution: "
            "%d grounding detections -> %d final detections "
            "(%d with ontology class, %d bbox-only/unresolved, "
            "%d dropped for invalid bbox)",
            len(grounding_detections),
            len(final_detections),
            resolved_count,
            unresolved_count,
            dropped_invalid_bbox,
        )

        return final_detections
    # ================================================================
    # MATCHING
    # ================================================================

    def _match_grounding(
        self,
        grounding: dict[str, Any],
        scene_objects: list[dict[str, Any]],
        used_scene_indices: set[int],
    ) -> tuple[int, dict[str, Any]] | None:
        """
        Match one Locate Anything result to its scene object.

        Matching priority:

        1. Exact grounding_prompt.
        2. Normalized grounding_prompt.
        3. Observed object / requested object.
        4. Locate Anything ref against the grounding prompt.

        The canonical ontology label is NEVER taken from the
        Locate Anything ref.
        """

        grounding_values = self._get_grounding_values(
            grounding
        )

        # Prefer the ontology grounding relationship over raw labels.
        # Locate Anything output may be noisy or object_id=None, but the
        # original scene object still carries the canonical ontology class
        # and grounding_prompt that created the request.

        # 1) explicit object_id match when the grounding record carries one.
        # Multiple valid bounding boxes may legitimately resolve to the same
        # canonical object_id, so this match must never be blocked solely by a
        # previous used_scene_indices entry for the same object.
        grounding_object_id = grounding.get("object_id")
        if grounding_object_id is not None:
            for index, scene_object in enumerate(scene_objects):
                scene_object_id = scene_object.get("object_id")
                if (
                    index in used_scene_indices
                    and scene_object_id != grounding_object_id
                ):
                    continue
                if scene_object_id == grounding_object_id:
                    return (index, scene_object)

        # 2) exact/normalized ontology grounding_prompt match
        for index, scene_object in enumerate(scene_objects):
            if index in used_scene_indices:
                # The used_scene_indices set may be non-empty for a different
                # object, but it must not cause valid grounding bboxes to be
                # discarded when they already map to a canonical object ID.
                if grounding_object_id is not None:
                    scene_object_id = scene_object.get("object_id")
                    if scene_object_id != grounding_object_id:
                        continue

            scene_prompt = self._get_grounding_prompt(scene_object)
            if not scene_prompt:
                continue

            scene_class = self._get_class_name(scene_object)
            observed_values = self._get_observed_values(scene_object)
            scene_values = [scene_prompt]
            if scene_class:
                scene_values.append(scene_class)
            scene_values.extend(observed_values)

            for scene_value in scene_values:
                normalized_scene = self._normalize_text(scene_value)
                if not normalized_scene:
                    continue

                for grounding_value in grounding_values:
                    normalized_grounding = self._normalize_text(grounding_value)
                    if normalized_grounding == normalized_scene:
                        return (index, scene_object)

        # 3) token-overlap fallback using ontology-grounding values and the
        # original observed object names, but never forcing identity from a
        # generic raw label when the ontology relationship is available.
        for index, scene_object in enumerate(scene_objects):
            if index in used_scene_indices and grounding_object_id is not None:
                scene_object_id = scene_object.get("object_id")
                if scene_object_id != grounding_object_id:
                    continue

            scene_prompt = self._get_grounding_prompt(scene_object)
            if not scene_prompt:
                continue

            scene_tokens = set(self._normalize_text(scene_prompt).split())
            scene_class = self._get_class_name(scene_object)
            if scene_class:
                scene_tokens |= set(self._normalize_text(scene_class).split())

            if not scene_tokens:
                continue

            for grounding_value in grounding_values:
                grounding_tokens = set(self._normalize_text(grounding_value).split())
                if not grounding_tokens:
                    continue

                overlap = scene_tokens & grounding_tokens
                if self._meaningful_overlap(overlap):
                    return (index, scene_object)

        return None

    @staticmethod
    def _get_grounding_values(
        grounding: dict[str, Any],
    ) -> list[str]:
        """
        Extract all potentially useful grounding names.

        Locate Anything may return None as its object ID/ref, so
        several cache fields are considered.
        """

        values: list[str] = []

        for key in (
            "grounding_prompt",
            "prompt",
            "object_name",
            "label",
            "ref",
        ):
            value = grounding.get(
                key
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                values.append(
                    value.strip()
                )

        return values

    @staticmethod
    def _get_observed_values(
        scene_object: dict[str, Any],
    ) -> list[str]:
        """Extract observed-object names from a scene object."""

        values: list[str] = []

        for key in (
            "observed_object",
            "object_name",
            "label",
            "name",
        ):
            value = scene_object.get(
                key
            )

            if (
                isinstance(
                    value,
                    str,
                )
                and value.strip()
            ):
                values.append(
                    value.strip()
                )

        return values

    @staticmethod
    def _meaningful_overlap(
        overlap: set[str],
    ) -> bool:
        """
        Determine whether token overlap is meaningful.

        Avoid using completely generic words as the only reason for
        a match.
        """

        generic_tokens = {
            "object",
            "thing",
            "item",
            "road",
            "traffic",
            "vehicle",
            "sign",
        }

        meaningful = (
            overlap
            - generic_tokens
        )

        return bool(
            meaningful
        )

    # ================================================================
    # FINAL LABEL
    # ================================================================

    @staticmethod
    def _get_final_label(
        scene_object: dict[str, Any],
    ) -> str | None:
        """
        Get the canonical ontology-resolved label.

        Ontology information has priority over any Locate Anything
        label.
        """

        for key in (
            "ontology_label",
            "predicted_label",
            "final_label",
            "canonical_label",
            "class_name",
        ):
            value = scene_object.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        ontology = scene_object.get("ontology")
        if isinstance(ontology, dict):
            for key in ("class_name", "label", "predicted_label", "canonical_label"):
                value = ontology.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        prediction = scene_object.get("ontology_prediction")
        if isinstance(prediction, dict):
            for key in ("class_name", "label", "predicted_label", "canonical_label"):
                value = prediction.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        ontology_reasoning = scene_object.get("ontology_reasoning")
        if isinstance(ontology_reasoning, dict):
            prediction = ontology_reasoning.get("prediction", {})
            if isinstance(prediction, dict):
                value = prediction.get("class_name")
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    @staticmethod
    def _get_class_name(
        scene_object: dict[str, Any],
    ) -> str | None:
        """Get the ontology class_name for a scene object."""

        value = scene_object.get("class_name")
        if isinstance(value, str) and value.strip():
            return value.strip()

        ontology = scene_object.get("ontology")
        if isinstance(ontology, dict):
            value = ontology.get("class_name")
            if isinstance(value, str) and value.strip():
                return value.strip()

        ontology_reasoning = scene_object.get("ontology_reasoning")
        if isinstance(ontology_reasoning, dict):
            prediction = ontology_reasoning.get("prediction", {})
            if isinstance(prediction, dict):
                value = prediction.get("class_name")
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    # ================================================================
    # GROUNDING PROMPT
    # ================================================================

    @staticmethod
    def _get_grounding_prompt(
        scene_object: dict[str, Any],
    ) -> str | None:
        """Get the ontology-generated grounding prompt."""

        value = scene_object.get("grounding_prompt")
        if isinstance(value, str) and value.strip():
            return value.strip()

        ontology = scene_object.get("ontology")
        if isinstance(ontology, dict):
            value = ontology.get("grounding_prompt")
            if isinstance(value, str) and value.strip():
                return value.strip()

        ontology_reasoning = scene_object.get("ontology_reasoning")
        if isinstance(ontology_reasoning, dict):
            prediction = ontology_reasoning.get("prediction", {})
            if isinstance(prediction, dict):
                value = prediction.get("grounding_prompt")
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    # ================================================================
    # OBJECT ID
    # ================================================================

    @staticmethod
    def _get_object_id(
        scene_object: dict[str, Any],
    ) -> Any:
        """
        Get the stable pipeline object ID.

        The object_id is informative but not authoritative. The final
        canonical label still comes from the ontology / grounding prompt.
        """

        return scene_object.get("object_id")

    @staticmethod
    def _get_class_id(
        scene_object: dict[str, Any],
    ) -> str | None:
        """Get the ontology class_id from the scene object."""

        value = scene_object.get("class_id")
        if isinstance(value, str) and value.strip():
            return value.strip()

        ontology = scene_object.get("ontology")
        if isinstance(ontology, dict):
            value = ontology.get("class_id")
            if isinstance(value, str) and value.strip():
                return value.strip()

        ontology_reasoning = scene_object.get("ontology_reasoning")
        if isinstance(ontology_reasoning, dict):
            prediction = ontology_reasoning.get("prediction", {})
            if isinstance(prediction, dict):
                value = prediction.get("class_id")
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return None

    # ================================================================
    # BBOX VALIDATION
    # ================================================================

    @staticmethod
    def _validate_bbox(
        bbox: Any,
    ) -> list[float] | None:
        """
        Validate a Locate Anything bounding box.

        Degenerate boxes are discarded.
        """

        if not isinstance(
            bbox,
            (list, tuple),
        ):
            return None

        if len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = map(
                float,
                bbox,
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        if x2 <= x1:
            return None

        if y2 <= y1:
            return None

        return [
            x1,
            y1,
            x2,
            y2,
        ]

    # ================================================================
    # TEXT NORMALIZATION
    # ================================================================

    @staticmethod
    def _normalize_text(
        value: Any,
    ) -> str:
        """Normalize text for grounding-prompt matching."""

        if value is None:
            return ""

        text = str(
            value
        ).strip().lower()

        replacements = {
            "_": " ",
            "-": " ",
        }

        for old, new in replacements.items():
            text = text.replace(
                old,
                new,
            )

        # Collapse repeated whitespace.
        return " ".join(
            text.split()
        )
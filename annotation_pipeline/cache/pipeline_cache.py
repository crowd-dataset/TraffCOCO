"""
pipeline_cache.py

Shared pipeline cache used for communication between annotation stages.

The cache stores intermediate outputs for every image and every detected
object. Each pipeline stage enriches the existing object information without
overwriting data produced by previous stages.

Cache hierarchy

Image
    └── Object
            ├── scene_understanding
            ├── ontology_reasoning
            ├── prompt_builder
            ├── grounding
            └── segmentation
"""

from __future__ import annotations

import json
from custom_logger import CustomLogger
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any

logger = CustomLogger(__name__)


# ============================================================================
# Pipeline Stages
# ============================================================================


class PipelineStage(str, Enum):
    """
    Enumeration of pipeline stages.
    """

    SCENE_UNDERSTANDING = "scene_understanding"

    ONTOLOGY_REASONING = "ontology_reasoning"

    PROMPT_BUILDER = "prompt_builder"

    GROUNDING = "grounding"

    SEGMENTATION = "segmentation"


# ============================================================================
# Pipeline Cache
# ============================================================================


class PipelineCache:
    """
    Shared cache used throughout the annotation pipeline.

    Structure

    {
        image_name:

            {
                object_id:

                    {
                        "object_id": int,

                        "scene_understanding": {},

                        "ontology_reasoning": {},

                        "prompt_builder": {},

                        "grounding": {},

                        "segmentation": {}
                    }
            }
    }
    """

    def __init__(self) -> None:

        self._cache: dict[
            str,
            dict[int, dict[str, Any]]
        ] = {}

        logger.info("Initialized PipelineCache")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _create_image(
        self,
        image_name: str,
    ) -> None:
        """
        Create an image entry if it does not already exist.
        """

        if image_name not in self._cache:

            self._cache[image_name] = {}

    def _normalize_object(
        self,
        obj: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Normalize Scene Understanding output.

        Missing fields are automatically created instead of rejecting
        the object.
        """

        obj = deepcopy(obj)

        obj.setdefault(
            "object_id",
            -1,
        )

        obj.setdefault(
            "observed_object",
            "unknown",
        )

        obj.setdefault(
            "description",
            "",
        )

        if "object_group" in obj:

            obj.setdefault(
                "visual_attributes",
                {},
            )

            visual = obj["visual_attributes"]

            visual.setdefault(
                "primary_color",
                "unknown",
            )

            visual.setdefault(
                "secondary_color",
                "unknown",
            )

            visual.setdefault(
                "shape",
                "unknown",
            )

            visual.setdefault(
                "background_color",
                "unknown",
            )

            visual.setdefault(
                "material",
                "unknown",
            )

            visual.setdefault(
                "reflective",
                "unknown",
            )

            visual.setdefault(
                "text",
                "none",
            )

            visual.setdefault(
                "symbol",
                "none",
            )

            visual.setdefault(
                "distinguishing_features",
                [],
            )

            visual.setdefault(
                "border",
                {
                    "shape": "none",
                    "color": "none",
                },
            )

        return obj
    # ------------------------------------------------------------------
    # Scene Understanding
    # ------------------------------------------------------------------

    def add_scene_objects(
        self,
        image_name: str,
        objects: list[dict[str, Any]],
    ) -> None:
        """
        Insert Scene Understanding results into the cache.

        Missing fields are automatically normalized instead of causing
        the object to be discarded.
        """

        self._create_image(image_name)

        inserted = 0

        normalized_objects = []

        for obj in objects:

            try:

                obj = self._normalize_object(
                    obj,
                )

                normalized_objects.append(
                    obj,
                )

            except Exception as exc:

                logger.warning(
                    "Unable to normalize object:\n{}",
                    json.dumps(
                        obj,
                        indent=4,
                        ensure_ascii=False,
                    ),
                )

                logger.warning(
                    "{}",
                    exc,
                )

        for obj in normalized_objects:

            object_id = int(
                obj["object_id"],
            )

            logger.info(
                "Cache currently contains object IDs for '{}': {}",
                image_name,
                list(
                    self._cache[image_name].keys()
                ),
            )

            logger.info(
                "Attempting to insert object_id={}",
                object_id,
            )

            if object_id in self._cache[image_name]:

                raise ValueError(
                    f"Duplicate object_id {object_id} "
                    f"for image '{image_name}'."
                )

            self._cache[image_name][object_id] = {

                "object_id": object_id,

                PipelineStage.SCENE_UNDERSTANDING.value:
                    deepcopy(obj),

                PipelineStage.ONTOLOGY_REASONING.value:
                    {},

                PipelineStage.PROMPT_BUILDER.value:
                    {},

                PipelineStage.GROUNDING.value:
                    {},

                PipelineStage.SEGMENTATION.value:
                    {},

            }

            inserted += 1

        if inserted == 0:

            raise ValueError(
                f"No valid Scene Understanding objects found for '{image_name}'."
            )

        logger.info(
            "Cached {} normalized object(s) for '{}'.",
            inserted,
            image_name,
        )
    def get_scene_objects(
        self,
        image_name: str,
    ) -> list[dict[str, Any]]:
        """
        Return the Scene Understanding objects for a single image.
        """

        if image_name not in self._cache:

            raise KeyError(
                f"No cache exists for image '{image_name}'."
            )

        objects = []

        for object_data in self._cache[image_name].values():

            scene_object = object_data.get(
                PipelineStage.SCENE_UNDERSTANDING.value,
                {},
            )

            if scene_object:

                objects.append(
                    deepcopy(scene_object)
                )

        return objects

    def add_ontology_result(
        self,
        image_name: str,
        object_id: int,
        ontology_result: dict[str, Any],
    ) -> None:
        """
        Store Ontology Reasoning results for an object.
        """

        if image_name not in self._cache:

            raise KeyError(
                f"No cache exists for image '{image_name}'."
            )

        if object_id not in self._cache[image_name]:

            raise KeyError(
                f"Object {object_id} does not exist for image '{image_name}'."
            )

        self._cache[
            image_name
        ][
            object_id
        ][
            PipelineStage.ONTOLOGY_REASONING.value
        ] = deepcopy(
            ontology_result
        )

        logger.debug(
            "Stored ontology result for object {} in '{}'.",
            object_id,
            image_name,
        )

    def add_grounding_result(
        self,
        image_name: str,
        object_id: int,
        grounding_result: dict[str, Any],
    ) -> None:
        """
        Store Locate Anything grounding results for an object.

        Grounding results are merged into the existing grounding entry
        rather than replacing the entire entry.

        Parameters
        ----------
        image_name
            Name of the image.

        object_id
            Original Scene Understanding object ID.

        grounding_result
            Parsed Locate Anything grounding information, typically:

            {
                "object_name": "passenger car",
                "bbox": [x1, y1, x2, y2]
            }
        """

        # ----------------------------------------------------------
        # Validate image
        # ----------------------------------------------------------

        if image_name not in self._cache:

            raise KeyError(
                f"No cache exists for image '{image_name}'."
            )

        # ----------------------------------------------------------
        # Validate object
        # ----------------------------------------------------------

        if object_id not in self._cache[image_name]:

            raise KeyError(
                f"Object {object_id} does not exist "
                f"for image '{image_name}'."
            )

        # ----------------------------------------------------------
        # Validate grounding result
        # ----------------------------------------------------------

        if not isinstance(
            grounding_result,
            dict,
        ):

            raise TypeError(
                "grounding_result must be a dictionary."
            )

        # ----------------------------------------------------------
        # Existing grounding data
        # ----------------------------------------------------------

        grounding = self._cache[
            image_name
        ][
            object_id
        ].setdefault(
            PipelineStage.GROUNDING.value,
            {},
        )

        # ----------------------------------------------------------
        # Merge new grounding result
        # ----------------------------------------------------------

        grounding.update(
            deepcopy(
                grounding_result,
            )
        )

        # ----------------------------------------------------------
        # Ensure object ID is never lost
        # ----------------------------------------------------------

        grounding["object_id"] = object_id

        logger.debug(
            "Stored grounding result for object {} in '{}': {}",
            object_id,
            image_name,
            grounding,
        )

    def save_image_cache(
        self,
        image_name: str,
        directory: Path | str,
        stage: str | None = None,
    ) -> Path:
        """
        Save the cached information for a single image.

        Each pipeline stage can optionally persist its intermediate cache to
        disk. These cache files are primarily intended for debugging,
        inspection, and pipeline development.

        The saved JSON represents the complete cache state for the specified
        image at the time this function is called.

        Parameters
        ----------
        image_name
            Name of the processed image.

        directory
            Output directory where the cache JSON should be written.

        stage
            Optional pipeline stage name used for logging.

        Returns
        -------
        Path
            Path to the saved cache file.

        Raises
        ------
        KeyError
            If the requested image does not exist in the cache.
        """

        if image_name not in self._cache:

            raise KeyError(
                f"No cache exists for image '{image_name}'."
            )

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = directory / f"{Path(image_name).stem}.json"

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self._cache[image_name],
                f,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "Saved {} cache to '{}'.",
            stage or "pipeline",
            output_path,
        )

        return output_path

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

    def _validate_object(
        self,
        obj: dict[str, Any],
    ) -> None:
        """
        Validate a Scene Understanding object before inserting it into the
        cache.
        """

        required_fields = (

            "object_id",

            "observed_object",

            "object_group",

        )

        missing = [

            field

            for field in required_fields

            if field not in obj

        ]

        if missing:

            raise ValueError(

                f"Missing required fields: {missing}"

            )

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

        Existing object IDs are not allowed and will raise an exception.
        """

        self._create_image(image_name)

        inserted = 0

        for obj in objects:

            self._validate_object(obj)

            object_id = int(obj["object_id"])

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

        logger.info(

            "Cached {} object(s) for '{}'.",

            inserted,

            image_name,

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

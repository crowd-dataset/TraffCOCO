"""
scene_understanding.py

Scene Understanding Engine.

This module implements the first stage of the VLM-First annotation pipeline.

Pipeline
--------
Image
    │
    ▼
Load Prompt
    │
    ▼
Vision-Language Model
    │
    ▼
Parse JSON
    │
    ▼
Pipeline Cache
"""

from __future__ import annotations

import json
from custom_logger import CustomLogger
import re
import traceback
from pathlib import Path
from typing import Any
import os
from annotation_pipeline.cache.pipeline_cache import PipelineCache
from annotation_pipeline.configs.settings import PipelineConfig
from annotation_pipeline.models.vlm.base_vlm import BaseVLM
from annotation_pipeline.prompts.load_prompt import load_prompt

logger = CustomLogger(__name__)


class SceneUnderstandingEngine:
    """
    Scene Understanding stage of the annotation pipeline.

    Responsibilities
    ----------------
    • Load scene-understanding prompt
    • Invoke the configured VLM
    • Parse returned JSON
    • Validate returned objects
    • Store objects in PipelineCache
    """

    def __init__(
        self,
        config: PipelineConfig,
        model: BaseVLM,
    ) -> None:

        self.config = config
        self.model = model

        self.scene_prompt = load_prompt(
            config.models.scene_understanding_prompt,
            prompts_dir=config.paths.prompts,
        )

        logger.info(
            "Initialized SceneUnderstandingEngine ({})",
            model.model_name,
        )

        # --------------------------------------------------------------
        # Benchmark Statistics
        # --------------------------------------------------------------

        self.last_generation_time = 0.0

        self.last_generated_tokens = 0

        self.last_total_tokens = 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def process_images(
        self,
        image_paths: list[Path],
        cache: PipelineCache,
    ) -> list[dict[str, Any]]:

        # --------------------------------------------------------------
        # Accept single image
        # --------------------------------------------------------------

        if isinstance(image_paths, Path):
            image_paths = [image_paths]

        # --------------------------------------------------------------
        # Validate images
        # --------------------------------------------------------------

        for image_path in image_paths:

            if not image_path.exists():

                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

        logger.info(
            "Running Scene Understanding for {} image(s)...",
            len(image_paths),
        )

        # --------------------------------------------------------------
        # Run VLM (batch inference)
        # --------------------------------------------------------------

        result = self.model.infer(
            image_paths=image_paths,
            prompt=self.scene_prompt,
        )

        responses = result["responses"]

        self.last_generation_time = result["generation_time"]
        self.last_generated_tokens = result["generated_tokens"]
        self.last_total_tokens = result["total_tokens"]

        logger.info(
            "Generation Time : {:.2f} s",
            self.last_generation_time,
        )

        logger.info(
            "Generated Tokens : {}",
            self.last_generated_tokens,
        )

        # --------------------------------------------------------------
        # Output directories
        # --------------------------------------------------------------

        raw_dir = self.config.paths.scene_cache / "raw"
        parsed_dir = self.config.paths.scene_cache / "parsed"

        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        parsed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------------
        # Process every image
        # --------------------------------------------------------------

        all_objects = []

        for image_path, raw_response in zip(image_paths, responses):

            image_name = image_path.name

            # ----------------------------------------------------------
            # Save raw output
            # ----------------------------------------------------------

            if self.config.pipeline.save_raw_outputs:

                raw_file = raw_dir / f"{image_path.stem}.txt"

                raw_file.write_text(
                    raw_response,
                    encoding="utf-8",
                )

                logger.info(
                    "Saved raw model output to '{}'.",
                    raw_file,
                )

            # ----------------------------------------------------------
            # Parse JSON
            # ----------------------------------------------------------

            objects = self._parse_json_response(
                raw_response,
            )

            parsed_file = parsed_dir / f"{image_path.stem}.json"

            with open(
                parsed_file,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    objects,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

            logger.info(
                "Saved parsed scene output to '{}'.",
                parsed_file,
            )

            logger.info(
                "Parsed {} object(s).",
                len(objects),
            )

            # ----------------------------------------------------------
            # Limit object count
            # ----------------------------------------------------------

            max_objects = self.config.pipeline.max_objects_per_image

            if len(objects) > max_objects:

                logger.warning(
                    "Model returned {} objects. Truncating to {}.",
                    len(objects),
                    max_objects,
                )

                objects = objects[:max_objects]

            # ----------------------------------------------------------
            # Cache
            # ----------------------------------------------------------

            cache.add_scene_objects(
                image_name=image_name,
                objects=objects,
            )

            if self.config.pipeline.save_intermediate_cache:

                cache.save_image_cache(
                    image_name=image_name,
                    directory=self.config.paths.scene_cache,
                    stage="scene_understanding",
                )

            logger.info(
                "Scene Understanding completed for '{}'.",
                image_name,
            )

            all_objects.append(objects)

        return all_objects

    # -------------------------------------------------------------------------
    # Normalizing
    # -------------------------------------------------------------------------

    def _normalize_objects(
        self,
        objects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        normalized = []

        for i, obj in enumerate(objects):

            if not isinstance(obj, dict):
                continue

            # -------------------------------------------------
            # object_id
            # -------------------------------------------------

            obj.setdefault("object_id", i + 1)

            # -------------------------------------------------
            # observed_object
            # -------------------------------------------------

            if "observed_object" not in obj:

                if "object" in obj:
                    obj["observed_object"] = obj.pop("object")

                elif "name" in obj:
                    obj["observed_object"] = obj.pop("name")

                elif "label" in obj:
                    obj["observed_object"] = obj.pop("label")

            # -------------------------------------------------
            # description
            # -------------------------------------------------

            obj.setdefault("description", "")

            # -------------------------------------------------
            # Schema B
            # -------------------------------------------------

            if "object_group" in obj:

                obj.setdefault("visual_attributes", {})

            if "observed_object" not in obj:

                logger.warning(
                    "Object {} still missing observed_object:\n{}",
                    obj.get("object_id", "?"),
                    json.dumps(obj, indent=4, ensure_ascii=False),
                )

            normalized.append(obj)

        return normalized

    # -------------------------------------------------------------------------
    # JSON Parsing
    # -------------------------------------------------------------------------

    def _parse_json_response(
        self,
        response: str,
    ) -> list[dict[str, Any]]:
        """
        Parse the raw Scene Understanding response into a JSON object.

        The parser accepts:

        • Markdown-wrapped JSON
        • Plain JSON arrays
        • Plain JSON objects

        Trailing commas are automatically removed before parsing.

        Parameters
        ----------
        response
            Raw response generated by the Vision-Language Model.

        Returns
        -------
        list[dict[str, Any]]
            Parsed Scene Understanding objects.

        Raises
        ------
        ValueError
            If no valid JSON can be extracted from the response.
        """

        logger.debug(
            "Parsing Scene Understanding response."
        )

        # ------------------------------------------------------------------
        # Try Markdown JSON first
        # ------------------------------------------------------------------

        match = re.search(
            r"```(?:json)?\s*(.*?)```",
            response,
            re.DOTALL,
        )

        if match is not None:

            json_text = match.group(1)

            logger.debug(
                "Detected markdown-wrapped JSON."
            )

        else:

            logger.debug(
                "No markdown JSON detected. Searching for plain JSON."
            )

            array_start = response.find("[")
            array_end = response.rfind("]")

            object_start = response.find("{")
            object_end = response.rfind("}")

            if array_start != -1 and array_end != -1:

                json_text = response[array_start:array_end + 1]

            elif object_start != -1 and object_end != -1:

                json_text = response[object_start:object_end + 1]

            else:

                logger.error(
                    "No JSON found in model response."
                )

                logger.debug(
                    "Raw model response:\n{}",
                    response,
                )

                raise ValueError(
                    "Scene model returned invalid output."
                )

        # ------------------------------------------------------------------
        # Remove trailing commas
        # ------------------------------------------------------------------

        json_text = re.sub(
            r",\s*([\]}])",
            r"\1",
            json_text,
        )

        # ------------------------------------------------------------------
        # Parse JSON
        # ------------------------------------------------------------------

        try:

            objects = json.loads(json_text)

        except json.JSONDecodeError as exc:

            logger.error(
                "Failed to parse Scene Understanding JSON."
            )

            logger.debug(
                "Invalid JSON:\n{}",
                json_text,
            )

            raise ValueError(
                "Failed to parse Scene Understanding output."
            ) from exc

        # ------------------------------------------------------------------
        # Normalize output
        # ------------------------------------------------------------------

        if isinstance(objects, dict):

            logger.warning(
                "Model returned a single JSON object. Converting to a list."
            )

            objects = [objects]

        if not isinstance(objects, list):

            logger.error(
                "Scene Understanding output is not a JSON array."
            )

            raise ValueError(
                "Scene Understanding output must be a JSON array."
            )

        logger.info(
            "Successfully parsed {} scene object(s).",
            len(objects),
        )

        objects = self._normalize_objects(objects)

        return objects
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
            "smolvlm+prompt.txt",
            prompts_dir=config.paths.prompts,
        )

        logger.info(
            "Initialized SceneUnderstandingEngine ({})",
            model.model_name,
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: Path | str,
        cache: PipelineCache,
    ) -> list[dict[str, Any]]:

        image_path = Path(image_path)

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image_name = image_path.name

        logger.info(
            "Running Scene Understanding for '{}'...",
            image_name,
        )

        # --------------------------------------------------------------
        # Model inference
        # --------------------------------------------------------------

        raw_response = self._generate_scene_description(
            image_path=image_path,
        )
        if self.config.pipeline.save_raw_outputs:

            raw_dir = Path(
                os.path.join(
                    str(self.config.paths.scene_cache),
                    "raw",
                )
            )
            raw_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            raw_file = Path(
                os.path.join(
                    str(raw_dir),
                    f"{image_path.stem}.txt",
                )
            )

            raw_file.write_text(
                raw_response,
                encoding="utf-8",
            )

            logger.info(
                "Saved raw model output to '{}'.",
                raw_file,
            )
        # --------------------------------------------------------------
        # Parse JSON
        # --------------------------------------------------------------

        objects = self._parse_json_response(
            raw_response,
        )

        parsed_dir = Path(
            os.path.join(
                str(self.config.paths.scene_cache),
                "parsed",
            )
        )

        parsed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        parsed_file = Path(
            os.path.join(
                str(parsed_dir),
                f"{image_path.stem}.json",
            )
        )

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

        # --------------------------------------------------------------
        # Limit object count
        # --------------------------------------------------------------

        max_objects = self.config.pipeline.max_objects_per_image

        if len(objects) > max_objects:

            logger.warning(
                "Model returned {} objects. Truncating to {}.",
                len(objects),
                max_objects,
            )

            objects = objects[:max_objects]

        # --------------------------------------------------------------
        # Cache
        # --------------------------------------------------------------

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

        return objects

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _generate_scene_description(
        self,
        image_path: Path,
    ) -> str:
        """
        Execute the configured Vision-Language Model.
        """

        logger.debug(
            "Running {}.",
            self.model.model_name,
        )

        try:

            return self.model.infer(
                image_path=image_path,
                prompt=self.scene_prompt,
            )

        except Exception as exc:
            

            logger.error("{}", traceback.format_exc())

            raise RuntimeError(
                "Scene Understanding model failed."
            ) from exc
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

        return objects

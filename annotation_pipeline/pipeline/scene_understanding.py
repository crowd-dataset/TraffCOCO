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
from pathlib import Path
from typing import Any
from annotation_pipeline.cache.pipeline_cache import PipelineCache
from annotation_pipeline.configs.settings import PipelineConfig
from annotation_pipeline.models.vlm.base_vlm import BaseVLM
from annotation_pipeline.prompts.load_prompt import load_prompt

logger = CustomLogger(__name__)

# With greedy decoding Qwen3-VL sometimes writes the same entry again and
# again, or spends its whole token budget on long sign/marking entries,
# and never names the cars in the frame (28 of 98 frames in the
# 2026-11-18 global test). An answer with this many repeated entries, or
# one cut off before its closing bracket, gets a short follow-up question
# for the vehicles and road users (asking it to "list what is missing"
# only got [] back).
LOOP_REPEATS = 3
FOLLOW_UP_MAX_OBJECTS = 15

FOLLOW_UP_PROMPT = """You are an expert traffic scene annotator.

List every vehicle and every road user visible in this dashcam image:
cars, SUVs, vans, pickup trucks, trucks, buses, taxis, motorcycles,
bicycles, auto-rickshaws, pedestrians, cyclists and motorcyclists.
Include parked, oncoming, distant and partly hidden ones, and the vehicle
directly ahead.

Give one entry per visually different type (for example "white sedan",
"red hatchback", "city bus", "pedestrian"); do not repeat an entry.
Return at most {max_objects} entries as a JSON array, nothing else:

[
    {{
        "object_id": 1,
        "observed_object": "...",
        "object_group": "vehicle" or "road_user",
        "description": "...",
        "confidence": 0.0 to 1.0
    }}
]
"""


def _entry_key(obj: dict) -> tuple[str, str]:
    """Name and description of an entry, used to spot repeats."""

    return (
        str(obj.get("observed_object", "")).strip().lower(),
        str(obj.get("description", "")).strip().lower(),
    )


def drop_repeated_entries(objects: list[dict]) -> tuple[list[dict], int]:
    """Keep the first of entries with the same name and description."""

    kept, seen = [], set()

    for obj in objects:

        key = _entry_key(obj)

        if key in seen:
            continue

        seen.add(key)
        kept.append(obj)

    return kept, len(objects) - len(kept)


def looks_cut_off(response: str) -> bool:
    """True when the answer stops before its JSON array is closed."""

    text = response.strip()

    if text.endswith("```"):
        text = text[:-3].rstrip()

    return not text.endswith("]")


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

    @staticmethod
    def _normalize_confidence(
        value: Any,
    ) -> float | None:
        """Normalize Scene Understanding self-reported confidence."""

        if value is None:
            return None

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None

        return max(
            0.0,
            min(1.0, confidence),
        )

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

            # One unparseable answer must not discard the other images
            # of the batch. The image gets no parsed file, so later
            # stages report it as failed on its own.
            try:
                objects = self._parse_json_response(
                    raw_response,
                )
            except ValueError as exc:
                logger.error(
                    "Scene Understanding failed for '{}': {}",
                    image_name,
                    exc,
                )
                all_objects.append([])
                continue

            # ----------------------------------------------------------
            # Repeated or cut-off answer: drop the repeats and ask once
            # more for the objects it never reached
            # ----------------------------------------------------------

            objects, repeats = drop_repeated_entries(objects)
            cut_off = looks_cut_off(raw_response)

            if repeats >= LOOP_REPEATS or cut_off:

                logger.warning(
                    "Scene Understanding for '{}' looped ({} repeated "
                    "entries, cut off: {}). Asking for the missing objects.",
                    image_name,
                    repeats,
                    cut_off,
                )

                objects = objects + self._ask_for_missing_objects(
                    image_path,
                    objects,
                    raw_dir,
                )

            # ----------------------------------------------------------
            # Limit object count (before saving, so the parsed file
            # that Ontology Reasoning reads matches the cache)
            # ----------------------------------------------------------

            max_objects = self.config.pipeline.max_objects_per_image

            if len(objects) > max_objects:

                logger.warning(
                    "Model returned {} objects. Truncating to {}.",
                    len(objects),
                    max_objects,
                )

                objects = objects[:max_objects]

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
            # Cache
            # ----------------------------------------------------------

            # A frame with no usable object (an empty road, say) is
            # skipped like an unparseable answer; the other images of
            # the batch go on.
            try:
                cache.add_scene_objects(
                    image_name=image_name,
                    objects=objects,
                )
            except ValueError as exc:
                logger.error(
                    "Scene Understanding failed for '{}': {}",
                    image_name,
                    exc,
                )
                parsed_file.unlink(missing_ok=True)
                all_objects.append([])
                continue

            if self.config.pipeline.save_intermediate_cache:

                cache.save_image_cache(
                    image_name=image_name,
                    directory=self.config.paths.pipeline_cache,
                    stage="scene_understanding",
                )

            logger.info(
                "Scene Understanding completed for '{}'.",
                image_name,
            )

            all_objects.append(objects)

        return all_objects

    # -------------------------------------------------------------------------
    # Follow-up for looping answers
    # -------------------------------------------------------------------------

    def _ask_for_missing_objects(
        self,
        image_path: Path,
        objects: list[dict[str, Any]],
        raw_dir: Path,
    ) -> list[dict[str, Any]]:
        """Ask once more for the vehicles and road users in a looping frame.

        Returns the new entries only, numbered after the existing ones.
        Any failure returns [] and leaves the first answer as it is.
        """

        names = sorted({_entry_key(obj)[0] for obj in objects if _entry_key(obj)[0]})
        prompt = FOLLOW_UP_PROMPT.format(
            max_objects=FOLLOW_UP_MAX_OBJECTS,
        )

        try:
            result = self.model.infer(
                image_paths=[image_path],
                prompt=prompt,
            )
            response = result["responses"][0]
            self.last_generation_time += result["generation_time"]
            self.last_generated_tokens += result["generated_tokens"]

            if self.config.pipeline.save_raw_outputs:
                (raw_dir / f"{image_path.stem}_follow_up.txt").write_text(
                    response,
                    encoding="utf-8",
                )

            extra = self._parse_json_response(response)

        except Exception as exc:  # the first answer still stands
            logger.error(
                "Follow-up question failed for '{}': {}",
                image_path.name,
                exc,
            )
            return []

        known = {_entry_key(obj) for obj in objects}
        extra, _ = drop_repeated_entries(
            [obj for obj in extra if _entry_key(obj) not in known]
        )
        extra = [obj for obj in extra if _entry_key(obj)[0] not in names]
        extra = extra[:FOLLOW_UP_MAX_OBJECTS]

        next_id = 1 + max(
            (obj["object_id"] for obj in objects
             if isinstance(obj.get("object_id"), int)),
            default=0,
        )

        for offset, obj in enumerate(extra):
            obj["object_id"] = next_id + offset

        logger.info(
            "Follow-up for '{}' added {} object(s): {}",
            image_path.name,
            len(extra),
            [obj.get("observed_object") for obj in extra],
        )

        return extra

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

            raw_confidence = obj.get(
                "confidence",
                obj.get("confidenc_score"),
            )

            obj["scene_confidence"] = self._normalize_confidence(
                raw_confidence
            )

            obj.pop("confidenc_score", None)

        return normalized

    # -------------------------------------------------------------------------
    # JSON Parsing
    # -------------------------------------------------------------------------

    @staticmethod
    def _recover_truncated_array(
        response: str,
    ) -> list[Any]:
        """Return the complete elements of a JSON array cut off mid-way."""

        start = response.find("[")

        if start == -1:
            return []

        decoder = json.JSONDecoder()
        objects: list[Any] = []
        pos = start + 1

        while True:

            while pos < len(response) and response[pos] in " \t\r\n,":
                pos += 1

            if pos >= len(response) or response[pos] == "]":
                break

            try:
                obj, pos = decoder.raw_decode(response, pos)
            except json.JSONDecodeError:
                break

            objects.append(obj)

        return objects

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

            # The model may stop mid-array when it hits max_new_tokens
            # (e.g. when it starts repeating objects). Keep every
            # complete object before the cut instead of dropping all.
            objects = self._recover_truncated_array(response)

            if not objects:

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

            logger.warning(
                "Scene Understanding JSON was truncated. Recovered {} "
                "complete object(s).",
                len(objects),
            )

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

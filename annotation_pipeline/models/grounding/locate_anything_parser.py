"""Locate Anything output parser with conservative scene-object matching."""

from __future__ import annotations

import re
from typing import Any

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


class LocateAnythingParser:
    """Parse Locate Anything boxes and attach them to scene objects."""

    def __init__(self) -> None:
        logger.info("Initialized LocateAnythingParser.")

        self.ref_pattern = re.compile(
            r"<ref>\s*(.*?)\s*</ref>",
            flags=re.DOTALL | re.IGNORECASE,
        )

        self.box_pattern = re.compile(
            r"<box>\s*"
            r"(?:"
            r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
            r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
            r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>\s*"
            r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>"
            r"|"
            r"([0-9]+(?:\.[0-9]+)?)\s*,\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*,\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*,\s*"
            r"([0-9]+(?:\.[0-9]+)?)"
            r")\s*</box>",
            flags=re.DOTALL | re.IGNORECASE,
        )

        self.point_pattern = re.compile(
            r"<box>\s*"
            r"<([0-9]+(?:\.[0-9]+)?)>\s*"
            r"<([0-9]+(?:\.[0-9]+)?)>\s*"
            r"</box>",
            flags=re.DOTALL | re.IGNORECASE,
        )

    def parse(
        self,
        raw_output: str,
        scene_objects: list[dict[str, Any]] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Parse native Locate Anything refs/boxes and optionally match IDs."""

        if not raw_output:
            logger.warning("Locate Anything returned empty output.")
            return []

        image_width: int | None = None
        image_height: int | None = None

        if image_size is not None:
            image_width, image_height = image_size
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"Invalid image_size: {image_size}")

        parsed: list[dict[str, Any]] = []
        current_ref: str | None = None

        token_pattern = re.compile(
            r"<ref>\s*(.*?)\s*</ref>|<box>\s*(.*?)\s*</box>",
            flags=re.DOTALL | re.IGNORECASE,
        )

        for token_match in token_pattern.finditer(raw_output):
            ref_text = token_match.group(1)
            box_text = token_match.group(2)

            if ref_text is not None:
                current_ref = self._clean_object_name(ref_text)
                continue

            if box_text is None or current_ref is None:
                continue

            numbers = re.findall(
                r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>",
                box_text,
            )
            if len(numbers) == 0:
                numbers = re.findall(
                    r"[0-9]+(?:\.[0-9]+)?",
                    box_text,
                )

            if len(numbers) == 2:
                logger.debug("Ignoring point output for '{}'.", current_ref)
                continue

            if len(numbers) != 4:
                logger.warning(
                    "Ignoring malformed box for '{}': {} coordinate(s).",
                    current_ref,
                    len(numbers),
                )
                continue

            raw_bbox = [self._to_number(value) for value in numbers]
            x1, y1, x2, y2 = raw_bbox
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))

            x1 = max(0, min(1000, x1))
            y1 = max(0, min(1000, y1))
            x2 = max(0, min(1000, x2))
            y2 = max(0, min(1000, y2))

            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    "Ignoring degenerate Locate Anything bounding box for '{}': {}",
                    current_ref,
                    raw_bbox,
                )
                continue

            if image_width is not None and image_height is not None:
                bbox = [
                    round(x1 / 1000.0 * image_width),
                    round(y1 / 1000.0 * image_height),
                    round(x2 / 1000.0 * image_width),
                    round(y2 / 1000.0 * image_height),
                ]
            else:
                bbox = [x1, y1, x2, y2]

            parsed.append(
                {
                    "object_id": None,
                    "object_name": current_ref,
                    "bbox": bbox,
                }
            )

        parsed = self._deduplicate(parsed)

        logger.info(
            "Found {} unique Locate Anything bounding-box detection(s).",
            len(parsed),
        )

        if not parsed or scene_objects is None:
            return parsed

        self._match_scene_objects(parsed, scene_objects)

        matched = sum(
            detection.get("object_id") is not None
            for detection in parsed
        )
        logger.info(
            "Created {} parsed detection(s); {} matched to scene objects.",
            len(parsed),
            matched,
        )

        unmatched = len(parsed) - matched
        if unmatched:
            logger.warning(
                "{} Locate Anything detection(s) could not be matched to scene objects.",
                unmatched,
            )

        return parsed

    def _match_scene_objects(
        self,
        detections: list[dict[str, Any]],
        scene_objects: list[dict[str, Any]],
    ) -> None:
        """Match LA refs using canonical names and conservative aliases.

        Multiple LA boxes may map to the same scene object. A scene object is
        therefore not consumed after the first successful match.
        """

        scene_by_name: dict[str, list[dict[str, Any]]] = {}

        for scene_object in scene_objects:
            seen_keys: set[str] = set()
            for field in (
                "grounding_prompt",
                "observed_object",
                "class_name",
                "object_name",
            ):
                value = scene_object.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue

                normalized = self._normalize_name(value)
                if not normalized or normalized in seen_keys:
                    continue

                seen_keys.add(normalized)
                scene_by_name.setdefault(normalized, []).append(scene_object)

        for detection in detections:
            detected_name = str(detection.get("object_name", ""))
            normalized_name = self._normalize_name(detected_name)
            matched_scene_object: dict[str, Any] | None = None
            match_type: str | None = None

            candidates = scene_by_name.get(normalized_name, [])

            # Exact match priority: grounding_prompt -> observed_object ->
            # class_name -> object_name.
            for field, label in (
                ("grounding_prompt", "grounding_prompt"),
                ("observed_object", "observed_object"),
                ("class_name", "class_name"),
                ("object_name", "object_name"),
            ):
                exact = [
                    candidate
                    for candidate in candidates
                    if self._normalize_name(
                        str(candidate.get(field) or "")
                    ) == normalized_name
                ]
                matched_scene_object = self._choose_unique(exact)
                if matched_scene_object is not None:
                    match_type = label
                    break

            # Conservative aliases only. No generic one-token overlap.
            if matched_scene_object is None:
                for alias in self._get_name_aliases(normalized_name):
                    alias_candidates = scene_by_name.get(alias, [])
                    matched_scene_object = self._choose_unique(alias_candidates)
                    if matched_scene_object is not None:
                        match_type = "alias"
                        break

            if matched_scene_object is None:
                logger.warning(
                    "No scene-object match for Locate Anything detection '{}'.",
                    detected_name,
                )
                continue

            object_id = matched_scene_object.get("object_id")
            if object_id is None:
                logger.warning(
                    "Matched Locate Anything '{}' to a scene object without object_id.",
                    detected_name,
                )
                continue

            detection["object_id"] = object_id
            detection["object_name"] = matched_scene_object.get(
                "observed_object",
                matched_scene_object.get("object_name", detected_name),
            )
            detection.setdefault("_match_metadata", {})
            detection["_match_metadata"].update(
                {
                    "match_type": match_type or "direct",
                    "matched_object_id": object_id,
                    "candidate_ids": [
                        candidate.get("object_id")
                        for candidate in candidates
                    ],
                }
            )

            logger.debug(
                "Matched Locate Anything '{}' to scene object ID {} (match_type={}).",
                detected_name,
                object_id,
                match_type or "direct",
            )

    @staticmethod
    def _choose_unique(
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return one candidate only when its canonical ID is unambiguous."""
        if not candidates:
            return None

        ids = [candidate.get("object_id") for candidate in candidates]
        unique_ids = {value for value in ids if value is not None}

        if len(unique_ids) == 1:
            object_id = next(iter(unique_ids))
            for candidate in candidates:
                if candidate.get("object_id") == object_id:
                    return candidate

        if len(candidates) == 1:
            return candidates[0]

        return None

    @staticmethod
    def _get_name_aliases(name: str) -> list[str]:
        """Return conservative semantic aliases."""
        alias_groups = {
            "car": {"passenger car", "car"},
            "passenger car": {"passenger car", "car"},
            "truck": {"truck", "lorry"},
            "lorry": {"truck", "lorry"},
            "bus": {"bus", "city bus"},
            "city bus": {"bus", "city bus"},
            "van": {"van"},
            "traffic signal": {"traffic signal", "traffic light", "signal head"},
            "traffic light": {"traffic signal", "traffic light", "signal head"},
            "signal head": {"traffic signal", "traffic light", "signal head"},
            "traffic sign": {"traffic sign", "road sign", "guide sign", "warning sign"},
            "road sign": {"traffic sign", "road sign", "guide sign"},
            "zebra": {"zebra", "zebra crossing", "crosswalk", "pedestrian crossing"},
            "zebra crossing": {"zebra", "zebra crossing", "crosswalk", "pedestrian crossing"},
            "crosswalk": {"zebra", "zebra crossing", "crosswalk", "pedestrian crossing"},
            "pedestrian crossing": {"zebra", "zebra crossing", "crosswalk", "pedestrian crossing"},
            "guardrail": {"guardrail", "guard rail", "guardail"},
            "guard rail": {"guardrail", "guard rail", "guardail"},
            "guardail": {"guardrail", "guard rail", "guardail"},
            "street light": {"street light", "streetlight", "street lamp", "lamp post", "light pole"},
            "streetlight": {"street light", "streetlight", "street lamp", "lamp post", "light pole"},
            "road marking": {"road marking", "road markings", "marking"},
            "road markings": {"road marking", "road markings", "marking"},
            "marking": {"road marking", "road markings", "marking"},
        }
        return sorted(alias_groups.get(name, set()) - {name})

    @staticmethod
    def _clean_object_name(name: str) -> str:
        """Clean accidental wrappers from a raw LA reference."""
        name = name.strip()
        name = re.sub(
            r"^(?:object\s+name|object|name)\s*:\s*",
            "",
            name,
            flags=re.IGNORECASE,
        )
        name = re.sub(
            r"^<\s*[-+]?\d+(?:\.\d+)?\s*>\s*",
            "",
            name,
        )
        return " ".join(name.split())

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize names and canonicalize recurring LA spelling errors."""
        name = str(name).strip().lower()
        name = re.sub(
            r"^(?:object\s+name|object|name)\s*:\s*",
            "",
            name,
            flags=re.IGNORECASE,
        )
        name = re.sub(r"[_-]+", " ", name)
        name = " ".join(name.split())

        if name == "guardail":
            return "guardrail"

        return name

    @staticmethod
    def _to_number(value: str) -> int | float:
        number = float(value)
        return int(number) if number.is_integer() else number

    @staticmethod
    def _deduplicate(
        detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove only exact duplicate ref+bbox detections."""
        seen: set[tuple[str, tuple[Any, ...]]] = set()
        unique: list[dict[str, Any]] = []

        for detection in detections:
            key = (
                str(detection.get("object_name", "")),
                tuple(detection.get("bbox", [])),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(detection)

        return unique

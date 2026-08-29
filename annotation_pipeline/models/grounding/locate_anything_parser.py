"""
locate_anything_parser.py

Parser for NVIDIA Locate Anything outputs.

Pipeline

Raw Model Output
        │
        ▼
Native Locate Anything Parsing
        │
        ▼
Structured Bounding Boxes
        │
        ▼
Scene Object Matching
"""

from __future__ import annotations

import re
from typing import Any

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


class LocateAnythingParser:
    """
    Parse native Locate Anything text outputs into structured detections.

    Locate Anything can return multiple bounding boxes for a single
    semantic reference, for example:

        <ref>passenger car</ref>
        <box><x1><y1><x2><y2></box>
        <box><x1><y1><x2><y2></box>

    The parser therefore treats every <box> following a <ref> as a
    separate detection while preserving the reference name.
    """

    def __init__(self) -> None:

        logger.info(
            "Initialized LocateAnythingParser."
        )

        # ------------------------------------------------------
        # Native Locate Anything tokens
        # ------------------------------------------------------

        self.ref_pattern = re.compile(
            r"<ref>\s*(.*?)\s*</ref>",
            flags=re.DOTALL | re.IGNORECASE,
        )

        self.box_pattern = re.compile(
            r"""
            <box>
            \s*
            (?:
                <\s*([0-9]+)\s*>
                <\s*([0-9]+)\s*>
                <\s*([0-9]+)\s*>
                <\s*([0-9]+)\s*>
                |
                ([0-9]+)\s*,\s*
                ([0-9]+)\s*,\s*
                ([0-9]+)\s*,\s*
                ([0-9]+)
            )
            \s*</box>
            """,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        # A point is also represented by <box>, but has only two
        # coordinates. We do not treat points as bounding boxes.
        self.point_pattern = re.compile(
            r"<box>\s*"
            r"<([0-9]+(?:\.[0-9]+)?)>\s*"
            r"<([0-9]+(?:\.[0-9]+)?)>\s*"
            r"</box>",
            flags=re.DOTALL | re.IGNORECASE,
        )

        

    # ==========================================================
    # Public API
    # ==========================================================

    def parse(
        self,
        raw_output: str,
        scene_objects: list[dict[str, Any]] | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Parse native Locate Anything output.

        Locate Anything returns bounding-box coordinates in a
        0-1000 coordinate space. These coordinates are converted
        to pixel coordinates using the ORIGINAL image dimensions
        supplied through ``image_size``.

        Parameters
        ----------
        raw_output:
            Raw Locate Anything model output.

        scene_objects:
            Optional Scene Understanding objects used for matching.

        image_size:
            Original image size as ``(width, height)``.

            Example:
                PIL image size -> (1280, 720)

            Locate Anything:
                <box><832><494><986><652></box>

            Converted:
                [1065, 356, 1262, 469]
        """

        logger.info(
            "Parsing Locate Anything output."
        )

        if not raw_output:
            logger.warning(
                "Locate Anything returned empty output."
            )
            return []

        # ----------------------------------------------------------
        # Validate image size
        # ----------------------------------------------------------

        if image_size is not None:

            image_width, image_height = image_size

            if image_width <= 0 or image_height <= 0:

                raise ValueError(
                    f"Invalid image_size: {image_size}"
                )

            logger.debug(
                "Using original image dimensions for bbox scaling: "
                "{}x{}",
                image_width,
                image_height,
            )

        else:

            image_width = None
            image_height = None

            logger.warning(
                "No image_size supplied to LocateAnythingParser. "
                "Bounding boxes will remain in Locate Anything's "
                "0-1000 coordinate space."
            )

        parsed_detections: list[dict[str, Any]] = []

        current_ref: str | None = None

        # ----------------------------------------------------------
        # Process <ref> and <box> tokens in model output order
        # ----------------------------------------------------------

        token_pattern = re.compile(
            r"<ref>\s*(.*?)\s*</ref>"
            r"|<box>\s*(.*?)\s*</box>",
            flags=re.DOTALL | re.IGNORECASE,
        )

        for token_match in token_pattern.finditer(
            raw_output,
        ):

            ref_text = token_match.group(1)
            box_text = token_match.group(2)

            # ------------------------------------------------------
            # New reference
            # ------------------------------------------------------

            if ref_text is not None:

                current_ref = self._clean_object_name(
                    ref_text,
                )

                continue

            # ------------------------------------------------------
            # Box
            # ------------------------------------------------------

            if box_text is None:
                continue

            if current_ref is None:

                logger.warning(
                    "Found Locate Anything box before a reference. "
                    "Skipping it."
                )

                continue

            # ------------------------------------------------------
            # Extract coordinates
            #
            # Native:
            # <box><608><489><696><610></box>
            #
            # Also:
            # <box>608,489,696,610</box>
            # ------------------------------------------------------

            numbers = re.findall(
                r"<\s*([0-9]+(?:\.[0-9]+)?)\s*>",
                box_text,
            )

            if len(numbers) == 0:

                numbers = re.findall(
                    r"[0-9]+(?:\.[0-9]+)?",
                    box_text,
                )

            # ------------------------------------------------------
            # Ignore point outputs
            # ------------------------------------------------------

            if len(numbers) == 2:

                logger.debug(
                    "Ignoring point output for '{}'.",
                    current_ref,
                )

                continue

            # ------------------------------------------------------
            # Reject malformed boxes
            # ------------------------------------------------------

            if len(numbers) != 4:

                logger.warning(
                    "Ignoring malformed box for '{}': "
                    "{} coordinate(s).",
                    current_ref,
                    len(numbers),
                )

                continue

            # ------------------------------------------------------
            # Convert model coordinates to numbers
            # ------------------------------------------------------

            raw_bbox = [
                self._to_number(numbers[0]),
                self._to_number(numbers[1]),
                self._to_number(numbers[2]),
                self._to_number(numbers[3]),
            ]

            x1, y1, x2, y2 = raw_bbox

            # ------------------------------------------------------
            # Normalize coordinate ordering FIRST
            # ------------------------------------------------------

            x1, x2 = sorted(
                (x1, x2),
            )

            y1, y2 = sorted(
                (y1, y2),
            )

            # ------------------------------------------------------
            # Validate native Locate Anything coordinates
            #
            # Locate Anything uses a 0-1000 coordinate system.
            # ------------------------------------------------------

            if (
                x1 < 0
                or y1 < 0
                or x2 > 1000
                or y2 > 1000
            ):

                logger.warning(
                    "Locate Anything bbox outside expected "
                    "0-1000 coordinate range for '{}': {}",
                    current_ref,
                    raw_bbox,
                )

                # Clamp rather than allowing invalid coordinates
                # into the visualization pipeline.

                x1 = max(0, min(1000, x1))
                y1 = max(0, min(1000, y1))
                x2 = max(0, min(1000, x2))
                y2 = max(0, min(1000, y2))

            # ------------------------------------------------------
            # Reject degenerate boxes AFTER normalization
            # ------------------------------------------------------

            if x2 <= x1 or y2 <= y1:

                logger.warning(
                    "Ignoring degenerate Locate Anything bounding box "
                    "for '{}': {}",
                    current_ref,
                    raw_bbox,
                )

                continue

            # ------------------------------------------------------
            # Convert 0-1000 coordinates to ORIGINAL IMAGE pixels
            # ------------------------------------------------------

            if (
                image_width is not None
                and image_height is not None
            ):

                x1 = x1 / 1000.0 * image_width
                y1 = y1 / 1000.0 * image_height
                x2 = x2 / 1000.0 * image_width
                y2 = y2 / 1000.0 * image_height

                # Keep coordinates inside the original image.

                x1 = max(
                    0,
                    min(image_width, x1),
                )

                y1 = max(
                    0,
                    min(image_height, y1),
                )

                x2 = max(
                    0,
                    min(image_width, x2),
                )

                y2 = max(
                    0,
                    min(image_height, y2),
                )

                # Use integers for downstream drawing/COCO operations.

                bbox = [
                    round(x1),
                    round(y1),
                    round(x2),
                    round(y2),
                ]

            else:

                # No original image dimensions were supplied.
                # Preserve the native 0-1000 coordinates.

                bbox = [
                    x1,
                    y1,
                    x2,
                    y2,
                ]

            # ------------------------------------------------------
            # Create detection
            # ------------------------------------------------------

            detection = {
                "object_id": None,
                "object_name": current_ref,
                "bbox": bbox,
            }

            parsed_detections.append(
                detection,
            )

            logger.debug(
                "Parsed Locate Anything detection: {}",
                detection,
            )

        logger.info(
            "Found {} Locate Anything bounding-box detection(s).",
            len(parsed_detections),
        )

        # ----------------------------------------------------------
        # Nothing parsed
        # ----------------------------------------------------------

        if not parsed_detections:

            logger.warning(
                "No Locate Anything bounding boxes could "
                "be parsed from the model output."
            )

            return []

        # ----------------------------------------------------------
        # No scene objects
        # ----------------------------------------------------------

        if scene_objects is None:

            logger.warning(
                "No scene objects supplied. "
                "Returning detections without scene IDs."
            )

            return parsed_detections

        # ----------------------------------------------------------
        # Match detections to Scene Understanding objects
        # ----------------------------------------------------------

        logger.info(
            "========== LOCATE ANYTHING PARSER DEBUG =========="
        )

        logger.info(
            "Parsed detections: {}",
            parsed_detections,
        )

        logger.info(
            "Scene objects: {}",
            scene_objects,
        )

        logger.info(
            "=================================================="
        )

        self._match_scene_objects(
            detections=parsed_detections,
            scene_objects=scene_objects,
        )

        # ----------------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------------

        matched_count = sum(
            1
            for detection in parsed_detections
            if detection.get("object_id") is not None
        )

        unmatched_detection_count = (
            len(parsed_detections)
            - matched_count
        )

        matched_scene_ids = {
            detection.get("object_id")
            for detection in parsed_detections
            if detection.get("object_id") is not None
        }

        unmatched_scene_count = sum(
            1
            for scene_object in scene_objects
            if scene_object.get("object_id")
            not in matched_scene_ids
        )

        logger.info(
            "Created {} parsed detection(s); "
            "{} matched to scene objects.",
            len(parsed_detections),
            matched_count,
        )

        if unmatched_detection_count > 0:

            logger.warning(
                "{} Locate Anything detection(s) "
                "could not be matched to scene objects.",
                unmatched_detection_count,
            )

        if unmatched_scene_count > 0:

            logger.warning(
                "{} scene object(s) did not receive "
                "a Locate Anything detection.",
                unmatched_scene_count,
            )

        return parsed_detections

    # ==========================================================
    # Scene Matching
    # ==========================================================

    def _match_scene_objects(
        self,
        detections: list[dict[str, Any]],
        scene_objects: list[dict[str, Any]],
    ) -> None:
        """
        Match Locate Anything detections to Scene Understanding objects.

        Matching strategy:
        1. Exact normalized object name.
        2. Known semantic aliases.
        3. Each scene object can only be assigned once.

        Extra Locate Anything boxes that do not correspond to another
        scene object remain unmatched and keep object_id=None.
        """

        scene_by_name: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        # ----------------------------------------------------------
        # Build normalized scene-object lookup
        # ----------------------------------------------------------

        for scene_object in scene_objects:

            object_name = scene_object.get(
                "observed_object",
                scene_object.get(
                    "object_name",
                    "",
                ),
            )

            if not object_name:
                continue

            normalized_name = self._normalize_name(
                str(object_name),
            )

            scene_by_name.setdefault(
                normalized_name,
                [],
            ).append(
                scene_object,
            )

        # ----------------------------------------------------------
        # Track already-used scene objects
        # ----------------------------------------------------------

        used_scene_ids: set[Any] = set()

        # ----------------------------------------------------------
        # Match detections
        # ----------------------------------------------------------

        for detection in detections:

            detected_name = str(
                detection.get(
                    "object_name",
                    "",
                )
            )

            normalized_name = self._normalize_name(
                detected_name,
            )

            # ------------------------------------------------------
            # First try exact match
            # ------------------------------------------------------

            candidates = scene_by_name.get(
                normalized_name,
                [],
            )

            matched_scene_object = None

            for candidate in candidates:

                candidate_id = candidate.get(
                    "object_id",
                )

                if candidate_id in used_scene_ids:
                    continue

                matched_scene_object = candidate
                break

            # ------------------------------------------------------
            # Try semantic aliases if exact match failed
            # ------------------------------------------------------

            if matched_scene_object is None:

                aliases = self._get_name_aliases(
                    normalized_name,
                )

                for alias in aliases:

                    candidates = scene_by_name.get(
                        alias,
                        [],
                    )

                    for candidate in candidates:

                        candidate_id = candidate.get(
                            "object_id",
                        )

                        if candidate_id in used_scene_ids:
                            continue

                        matched_scene_object = candidate
                        break

                    if matched_scene_object is not None:
                        break

            # ------------------------------------------------------
            # No matching scene object
            # ------------------------------------------------------

            if matched_scene_object is None:

                logger.warning(
                    "No scene-object match for Locate Anything "
                    "detection '{}'.",
                    detected_name,
                )

                continue

            # ------------------------------------------------------
            # Extract scene-object identity
            # ------------------------------------------------------

            object_id = matched_scene_object.get(
                "object_id",
            )

            object_name = matched_scene_object.get(
                "observed_object",
                matched_scene_object.get(
                    "object_name",
                    detected_name,
                ),
            )

            # ------------------------------------------------------
            # Do NOT assign None as a valid object ID
            # ------------------------------------------------------

            if object_id is None:

                logger.warning(
                    "Matched Locate Anything '{}' to a scene object "
                    "without object_id.",
                    detected_name,
                )

                continue

            detection["object_id"] = object_id

            detection["object_name"] = object_name

            used_scene_ids.add(
                object_id,
            )

            logger.debug(
                "Matched Locate Anything '{}' "
                "to scene object ID {}.",
                detected_name,
                object_id,
            )

    @staticmethod
    def _get_name_aliases(
        name: str,
    ) -> list[str]:
        """
        Return known semantic aliases for Locate Anything labels.

        These aliases are intentionally conservative. They are only used
        after an exact normalized-name match fails.
        """

        alias_groups = {
            "car": {
                "passenger car",
                "car",
                "vehicle",
            },
            "passenger car": {
                "passenger car",
                "car",
                "vehicle",
            },
            "vehicle": {
                "passenger car",
                "car",
                "van",
                "truck",
                "bus",
                "vehicle",
            },
            "road marking": {
                "road marking",
                "road markings",
                "lane divider",
                "lane divider broken",
                "dashed lane divider",
                "lane boundary",
                "marking",
            },
            "road markings": {
                "road marking",
                "road markings",
                "lane divider",
                "lane divider broken",
                "dashed lane divider",
                "lane boundary",
                "marking",
            },
            "street light": {
                "street light",
                "streetlight",
                "lamp post",
                "light pole",
                "street lamp",
            },
            "streetlight": {
                "street light",
                "streetlight",
                "lamp post",
                "light pole",
                "street lamp",
            },
            "truck": {
                "truck",
                "lorry",
                "vehicle",
            },
            "lorry": {
                "truck",
                "lorry",
                "vehicle",
            },
            "traffic sign": {
                "traffic sign",
                "road sign",
                "guide sign",
                "warning sign",
                "regulatory traffic sign",
                "bilingual road sign",
            },
            "road sign": {
                "traffic sign",
                "road sign",
                "guide sign",
                "warning sign",
                "regulatory traffic sign",
                "bilingual road sign",
            },
            "bilingual road sign": {
                "traffic sign",
                "road sign",
                "guide sign",
                "bilingual road sign",
            },
            "traffic signal": {
                "traffic signal",
                "traffic light",
                "signal head",
            },
            "traffic light": {
                "traffic signal",
                "traffic light",
                "signal head",
            },
            "bus": {
                "bus",
                "city bus",
                "vehicle",
            },
            "van": {
                "van",
                "vehicle",
            },
            "building": {
                "building",
            },
            "marking": {
                "marking",
                "road marking",
                "lane divider",
                "dashed lane divider",
            },
        }

        return sorted(
            alias_groups.get(
                name,
                set(),
            )
            - {name},
        )
    # ==========================================================
    # Utilities
    # ==========================================================

    @staticmethod
    def _clean_object_name(
        name: str,
    ) -> str:
        """
        Clean a raw Locate Anything reference name.
        """

        name = name.strip()

        # Remove accidental model-generated wrappers such as:
        # "<948> marking" or "object name: passenger car"
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
            flags=re.IGNORECASE,
        )
        name = re.sub(
            r"^(?:[-+]?\d+(?:\.\d+)?\s*[,;:.-]\s*)+",
            "",
            name,
            flags=re.IGNORECASE,
        )

        return " ".join(
            name.split()
        )

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """
        Normalize an object name for scene-object matching.
        """

        name = name.strip().lower()

        name = re.sub(
            r"^(?:object\s+name|object|name)\s*:\s*",
            "",
            name,
            flags=re.IGNORECASE,
        )

        # Treat underscores/hyphens as spaces so:
        # road_marking == road marking
        # street-light == street light
        name = re.sub(
            r"[_\-]+",
            " ",
            name,
        )

        return " ".join(
            name.split()
        )

    @staticmethod
    def _to_number(
        value: str,
    ) -> int | float:
        """
        Convert a numeric string to int when possible.
        """

        number = float(value)

        if number.is_integer():
            return int(number)

        return number
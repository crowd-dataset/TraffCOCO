"""
retrieval_query_builder.py

Converts parsed Scene Understanding output into a RetrievalQuery
for ontology retrieval.
"""

from __future__ import annotations

import re
from typing import Any

from custom_logger import CustomLogger

from annotation_pipeline.models.ontology.ontology_models import RetrievalQuery

logger = CustomLogger(__name__)


class RetrievalQueryBuilder:
    """Converts one parsed Scene Understanding object into a RetrievalQuery."""

    def __init__(self) -> None:
        logger.info("Initialized Retrieval Query Builder.")

    def build(self, scene_object: dict[str, Any]) -> RetrievalQuery:
        logger.info(
            "Building RetrievalQuery for '{}'.",
            scene_object.get("observed_object", scene_object.get("object_name", "unknown")),
        )

        return RetrievalQuery(
            observed_object=self._get_object_name(scene_object),
            object_group=self._get_object_group(scene_object),
            description=self._get_description(scene_object),
            shape=self._get_shape(scene_object),
            primary_color=self._get_primary_color(scene_object),
            secondary_color=self._get_secondary_color(scene_object),
            material=self._get_material(scene_object),
            text=self._get_text(scene_object),
            symbol=self._get_symbol(scene_object),
            distinguishing_features=self._get_distinctive_features(scene_object),
            attached_to=self._get_attachment(scene_object),
            nearby_objects=self._get_nearby_objects(scene_object),
            road_side=self._get_road_side(scene_object),
            possible_function=self._get_function(scene_object),
            classification_hint=self._get_category(scene_object),

            # Canonical Scene Understanding confidence.
            # No fabricated 1.0 fallback.
            scene_confidence=self._get_scene_confidence(scene_object),

            # Confidence is metadata, not semantic content.
            embedding_text=self._build_embedding_text(scene_object),
        )

    @staticmethod
    def _get_scene_confidence(scene_object: dict[str, Any]) -> float | None:
        """Return canonical Scene Understanding confidence in [0, 1]."""
        value = scene_object.get("confidence")

        # Backward compatibility with the historical typo.
        if value is None:
            value = scene_object.get("confidenc_score")

        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        # Canonical representation.
        if 0.0 <= value <= 1.0:
            return value

        # Legacy integer percentage representation.
        if 0.0 <= value <= 100.0:
            return value / 100.0

        return None

    @staticmethod
    def _get_object_name(scene_object: dict[str, Any]) -> str:
        return scene_object.get(
            "observed_object",
            scene_object.get("object_name", ""),
        )

    @staticmethod
    def _get_object_group(scene_object: dict[str, Any]) -> str:
        return scene_object.get("object_group", "")

    @staticmethod
    def _get_description(scene_object: dict[str, Any]) -> str:
        return scene_object.get("description", "")

    @staticmethod
    def _get_visual_attributes(scene_object: dict[str, Any]) -> dict[str, Any]:
        visual = scene_object.get("visual_attributes", {})
        return visual if isinstance(visual, dict) else {}

    @classmethod
    def _get_shape(cls, scene_object: dict[str, Any]) -> str:
        return cls._get_visual_attributes(scene_object).get("shape", "")

    @classmethod
    def _get_primary_color(cls, scene_object: dict[str, Any]) -> str:
        return cls._get_visual_attributes(scene_object).get("primary_color", "")

    @classmethod
    def _get_secondary_color(cls, scene_object: dict[str, Any]) -> str:
        return cls._get_visual_attributes(scene_object).get("secondary_color", "")

    @classmethod
    def _get_material(cls, scene_object: dict[str, Any]) -> str:
        return cls._get_visual_attributes(scene_object).get("material", "")

    @classmethod
    def _get_text(cls, scene_object: dict[str, Any]) -> str:
        visual = cls._get_visual_attributes(scene_object)
        return visual.get("text", visual.get("text_or_symbol", ""))

    @classmethod
    def _get_symbol(cls, scene_object: dict[str, Any]) -> str:
        """Extract a symbolic icon/pictogram from the parser output."""
        visual = cls._get_visual_attributes(scene_object)

        text = visual.get("symbol", "")
        if not text:
            text = visual.get("text_or_symbol", "")

        if not text:
            return ""

        text = str(text).strip()

        unicode_symbols = re.findall(
            r"[↑↓←→↖↗↘↙↔↕⟲⟳⇧⇩⇦⇨⇪⇵⚠🚸🚳🚲🚶🅿ⓅⓂ]",
            text,
        )
        if unicode_symbols:
            return unicode_symbols[0]

        letter_match = re.search(
            r"\b(P|H|M|T|E)\b",
            text,
            flags=re.IGNORECASE,
        )
        if letter_match:
            return letter_match.group(1).upper()

        lower = text.lower()
        symbol_keywords = [
            "upward arrow", "downward arrow", "left arrow", "right arrow",
            "straight arrow", "straight ahead", "go straight", "turn left",
            "turn right", "u-turn", "roundabout", "pedestrian",
            "walking person", "bicycle", "cyclist", "motorcycle", "bus",
            "tram", "taxi", "parking", "hospital", "airport", "railway",
            "train", "fuel", "telephone", "camera", "wheelchair",
            "children", "school", "stop", "yield",
        ]

        for keyword in symbol_keywords:
            if keyword in lower:
                return keyword

        return text

    @classmethod
    def _get_distinctive_features(cls, scene_object: dict[str, Any]) -> list[str]:
        features: list[str] = []
        visual = cls._get_visual_attributes(scene_object)

        location = scene_object.get("location", {})
        if not isinstance(location, dict):
            location = {}

        context = scene_object.get("context", {})
        if not isinstance(context, dict):
            context = {}

        size = visual.get("size", "")
        if size:
            features.append(str(size))

        depth = location.get("approx_depth", "")
        if depth:
            features.append(str(depth))

        visibility = context.get("visibility", "")
        if visibility:
            features.append(str(visibility))

        if context.get("is_occluded", False):
            features.append("occluded")

        return features

    @staticmethod
    def _get_attachment(scene_object: dict[str, Any]) -> str:
        context = scene_object.get("context", {})
        if not isinstance(context, dict):
            return ""
        return context.get("attached_to", "")

    @staticmethod
    def _get_nearby_objects(scene_object: dict[str, Any]) -> list[str]:
        context = scene_object.get("context", {})
        if not isinstance(context, dict):
            return []
        nearby = context.get("nearby_objects", [])
        return nearby if isinstance(nearby, list) else []

    @staticmethod
    def _get_road_side(scene_object: dict[str, Any]) -> str:
        context = scene_object.get("context", {})
        if not isinstance(context, dict):
            return ""
        return context.get("road_side", "")

    @staticmethod
    def _get_function(scene_object: dict[str, Any]) -> str:
        return scene_object.get("possible_function", "")

    @staticmethod
    def _get_category(scene_object: dict[str, Any]) -> str:
        return scene_object.get("classification_hint", "")

    def _build_embedding_text(self, scene_object: dict[str, Any]) -> str:
        parts: list[str] = []

        name = self._get_object_name(scene_object)
        if name:
            parts.append(str(name))

        group = self._get_object_group(scene_object)
        if group:
            parts.append(f"Group: {group}")

        description = self._get_description(scene_object)
        if description:
            parts.append(str(description))

        shape = self._get_shape(scene_object)
        if shape:
            parts.append(f"Shape: {shape}")

        colors = []
        primary = self._get_primary_color(scene_object)
        if primary:
            colors.append(str(primary))
        secondary = self._get_secondary_color(scene_object)
        if secondary:
            colors.append(str(secondary))
        if colors:
            parts.append("Colors: " + ", ".join(colors))

        material = self._get_material(scene_object)
        if material:
            parts.append(f"Material: {material}")

        text = self._get_text(scene_object)
        if text:
            parts.append(f"Text: {text}")

        symbol = self._get_symbol(scene_object)
        if symbol:
            parts.append(f"Symbol: {symbol}")

        nearby = self._get_nearby_objects(scene_object)
        if nearby:
            parts.append("Nearby: " + ", ".join(map(str, nearby)))

        features = self._get_distinctive_features(scene_object)
        if features:
            parts.append("Features: " + ", ".join(features))

        function = self._get_function(scene_object)
        if function:
            parts.append(f"Purpose: {function}")

        return ". ".join(parts)

"""
retrieval_query_builder.py

Converts parsed Scene Understanding output into a RetrievalQuery
for ontology retrieval.

Pipeline

Parsed Scene Understanding JSON
        │
        ▼
RetrievalQueryBuilder
        │
        ▼
RetrievalQuery
"""


from __future__ import annotations

import re

from typing import Any

from custom_logger import CustomLogger

from annotation_pipeline.models.ontology.ontology_models import (
    RetrievalQuery,
)

logger = CustomLogger(__name__)


class RetrievalQueryBuilder:
    """
    Converts one parsed Scene Understanding object into a
    RetrievalQuery used for ontology retrieval.
    """

    def __init__(
        self,
    ) -> None:

        logger.info(
            "Initialized Retrieval Query Builder."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        scene_object: dict[str, Any],
    ) -> RetrievalQuery:
        """
        Build a RetrievalQuery from one Scene Understanding object.
        """

        logger.info(
            "Building RetrievalQuery for '{}'.",
            scene_object.get(
                "object_name",
                "unknown",
            ),
        )

        return RetrievalQuery(

            observed_object=self._get_object_name(
                scene_object,
            ),

            object_group=self._get_object_group(
                scene_object,
            ),

            description=self._get_description(
                scene_object,
            ),

            shape=self._get_shape(
                scene_object,
            ),

            primary_color=self._get_primary_color(
                scene_object,
            ),

            secondary_color=self._get_secondary_color(
                scene_object,
            ),

            material=self._get_material(
                scene_object,
            ),

            text=self._get_text(
                scene_object,
            ),

            symbol=self._get_symbol(
                scene_object,
            ),

            distinguishing_features=self._get_distinctive_features(
                scene_object,
            ),

            attached_to=self._get_attachment(
                scene_object,
            ),

            nearby_objects=self._get_nearby_objects(
                scene_object,
            ),

            road_side=self._get_road_side(
                scene_object,
            ),

            possible_function=self._get_function(
                scene_object,
            ),

            classification_hint=self._get_category(
                scene_object,
            ),

            confidence=scene_object.get(
                "confidence",
                1.0,
            ),

            embedding_text=self._build_embedding_text(
                scene_object,
            ),
        )

    # ------------------------------------------------------------------
    # Basic Field Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_object_name(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract observed object name.
        """

        return scene_object.get(
            "observed_object",
            "",
        )

    @staticmethod
    def _get_object_group(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract object group from Schema B.

        Schema A objects simply return an empty string.
        """

        return scene_object.get(
            "object_group",
            "",
        )


    @staticmethod
    def _get_description(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract object description.
        """

        return scene_object.get(
            "description",
            "",
        )


    @staticmethod
    def _get_visual_attributes(
        scene_object: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return the visual_attributes dictionary.
        """

        return scene_object.get(
            "visual_attributes",
            {},
        )


    @staticmethod
    def _get_shape(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract object shape.
        """

        visual = scene_object.get(
            "visual_attributes",
            {},
        )

        return visual.get(
            "shape",
            "",
        )


    @staticmethod
    def _get_primary_color(
        scene_object: dict[str, Any],
    ) -> str:

        visual = scene_object.get(
            "visual_attributes",
            {},
        )

        return visual.get(
            "primary_color",
            "",
        )


    @staticmethod
    def _get_secondary_color(
        scene_object: dict[str, Any],
    ) -> str:

        visual = scene_object.get(
            "visual_attributes",
            {},
        )

        return visual.get(
            "secondary_color",
            "",
        )


    @staticmethod
    def _get_material(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract object material.
        """

        visual = scene_object.get(
            "visual_attributes",
            {},
        )

        return visual.get(
            "material",
            "",
        )

    # ------------------------------------------------------------------
    # Context & Semantic Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _get_text(
        scene_object: dict[str, Any],
    ) -> str:

        visual = scene_object.get(
            "visual_attributes",
            {},
        )

        return visual.get(
            "text",
            "",
        )




    @staticmethod
    def _get_symbol(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract a symbolic icon or traffic pictogram.

        Supports the new parser schema while retaining the previous
        normalization logic.
        """

        visual = scene_object.get(
            "visual_attributes",
            {},
        )

        # -------------------------------------------------------------
        # Prefer the explicit symbol field from the parser
        # -------------------------------------------------------------

        text = visual.get(
            "symbol",
            "",
        )

        # Fall back to OCR text if symbol is empty
        if not text:

            text = visual.get(
                "text",
                "",
            )

        if not text:
            return ""

        text = text.strip()

        # -------------------------------------------------------------
        # 1. Unicode traffic symbols
        # -------------------------------------------------------------

        unicode_symbols = re.findall(
            r"[↑↓←→↖↗↘↙↔↕⟲⟳⇧⇩⇦⇨⇪⇵⚠🚸🚳🚲🚶🅿ⓅⓂ]",
            text,
        )

        if unicode_symbols:
            return unicode_symbols[0]

        # -------------------------------------------------------------
        # 2. Single-letter traffic icons
        # -------------------------------------------------------------

        letter_match = re.search(
            r"\b(P|H|M|T|E)\b",
            text,
            flags=re.IGNORECASE,
        )

        if letter_match:
            return letter_match.group(1).upper()

        # -------------------------------------------------------------
        # 3. Common traffic symbols
        # -------------------------------------------------------------

        lower = text.lower()

        symbol_keywords = [

            "upward arrow",
            "downward arrow",
            "left arrow",
            "right arrow",
            "straight arrow",
            "straight ahead",
            "go straight",
            "turn left",
            "turn right",
            "u-turn",
            "roundabout",

            "pedestrian",
            "walking person",
            "bicycle",
            "cyclist",
            "motorcycle",
            "bus",
            "tram",
            "taxi",

            "parking",
            "hospital",
            "airport",
            "railway",
            "train",
            "fuel",
            "telephone",
            "camera",

            "wheelchair",
            "children",
            "school",
            "stop",
            "yield",
        ]

        for keyword in symbol_keywords:

            if keyword in lower:
                return keyword

        # -------------------------------------------------------------
        # Return parser output if nothing matched
        # -------------------------------------------------------------

        return text


    @staticmethod
    def _get_distinctive_features(
        scene_object: dict[str, Any],
    ) -> list[str]:
        """
        Build a list of distinctive visual features from the parsed output.
        """

        features: list[str] = []

        visual = scene_object.get(
            "visual_attributes",
            {},
        )

        location = scene_object.get(
            "location",
            {},
        )

        context = scene_object.get(
            "context",
            {},
        )

        size = visual.get(
            "size",
            "",
        )

        if size:

            features.append(size)

        depth = location.get(
            "approx_depth",
            "",
        )

        if depth:

            features.append(depth)

        visibility = context.get(
            "visibility",
            "",
        )

        if visibility:

            features.append(visibility)

        if context.get(
            "is_occluded",
            False,
        ):

            features.append(
                "occluded",
            )

        return features


    @staticmethod
    def _get_attachment(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract object attachment.
        """

        context = scene_object.get(
            "context",
            {},
        )

        return context.get(
            "attached_to",
            "",
        )


    @staticmethod
    def _get_nearby_objects(
        scene_object: dict[str, Any],
    ) -> list[str]:
        """
        Extract nearby contextual objects.
        """

        context = scene_object.get(
            "context",
            {},
        )

        return context.get(
            "nearby_objects",
            [],
        )


    @staticmethod
    def _get_road_side(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract road side information from the parsed scene object.
        """

        context = scene_object.get(
            "context",
            {},
        )

        return context.get(
            "road_side",
            "",
        )


    @staticmethod
    def _get_function(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract inferred object function.
        """

        return scene_object.get(
            "possible_function",
            "",
        )


    @staticmethod
    def _get_category(
        scene_object: dict[str, Any],
    ) -> str:
        """
        Extract classification hint.

        The current Scene Understanding schema does not provide one.
        """

        return ""

    def _build_embedding_text(
        self,
        scene_object: dict[str, Any],
    ) -> str:
        """
        Build the semantic embedding text for ontology retrieval.
        """

        parts: list[str] = []

        name = self._get_object_name(scene_object)
        if name:
            parts.append(name)

        group = self._get_object_group(scene_object)

        if group:
            parts.append(f"Group: {group}")

        description = self._get_description(scene_object)
        if description:
            parts.append(description)

        shape = self._get_shape(scene_object)
        if shape:
            parts.append(f"Shape: {shape}")

        colors = []

        primary = self._get_primary_color(scene_object)
        if primary:
            colors.append(primary)

        secondary = self._get_secondary_color(scene_object)
        if secondary:
            colors.append(secondary)

        if colors:
            parts.append(
                "Colors: " + ", ".join(colors)
            )

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
            parts.append(
                "Nearby: " + ", ".join(nearby)
            )

        features = self._get_distinctive_features(scene_object)
        if features:
            parts.append(
                "Features: " + ", ".join(features)
            )

        function = self._get_function(scene_object)
        if function:
            parts.append(f"Purpose: {function}")

        return ". ".join(parts)
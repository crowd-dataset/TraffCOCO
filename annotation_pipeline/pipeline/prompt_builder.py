"""
prompt_builder.py

Prompt Builder for the Locate Anything grounding stage.

This module converts parsed Scene Understanding objects into a Locate
Anything prompt while preserving the original object ordering and IDs.

Pipeline
--------
Scene Objects
        │
        ▼
Prompt Builder
        │
        ▼
Prompt String
"""

from __future__ import annotations

from typing import Any

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


class PromptBuilder:
    """
    Build Locate Anything prompts.

    Responsibilities
    ----------------
    • Preserve object ordering
    • Preserve object IDs
    • Generate deterministic prompts
    """

    def __init__(self) -> None:

        logger.info(
            "Initialized PromptBuilder."
        )

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def build_prompt(
        self,
        scene_objects: list[dict[str, Any]],
    ) -> str:
        """
        Build a native Locate Anything category prompt from
        Scene Understanding output.

        The object categories are extracted dynamically from the
        parsed scene-understanding JSON. No object classes are
        hard-coded here.

        Locate Anything expects categories to be separated by:

            </c>

        Example
        -------
        Scene JSON:
            passenger car
            van
            truck
            road marking
            street light
            building

        Generated prompt:
            passenger car</c>van</c>truck</c>road marking</c>street light</c>building
        """

        logger.info(
            "Building Locate Anything grounding prompt."
        )

        object_names: list[str] = []

        for obj in scene_objects:

            if not isinstance(
                obj,
                dict,
            ):
                logger.warning(
                    "Skipping invalid scene object: {}",
                    obj,
                )
                continue

            object_name = obj.get(
                "observed_object",
                obj.get(
                    "object_name",
                    "",
                ),
            )

            if not object_name:
                continue

            object_name = str(
                object_name
            ).strip()

            if not object_name:
                continue

            # ------------------------------------------------------
            # Preserve the class name exactly as provided by the
            # parsed Scene Understanding output.
            # Do NOT replace spaces with underscores.
            # ------------------------------------------------------

            if object_name not in object_names:

                object_names.append(
                    object_name
                )

        if not object_names:

            logger.warning(
                "No valid object names found for grounding."
            )

            return ""

        # ----------------------------------------------------------
        # Native Locate Anything category separator.
        # ----------------------------------------------------------

        prompt = "</c>".join(
            object_names
        )

        logger.info(
            "Created grounding prompt for {} unique "
            "object categories.",
            len(object_names),
        )

        logger.info(
            "Locate Anything prompt: {}",
            prompt,
        )

        return prompt
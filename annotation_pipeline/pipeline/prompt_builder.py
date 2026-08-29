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

import re
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

    @staticmethod
    def _normalize_label(value: Any) -> str:
        """Normalize a visual label into a concise grounding phrase."""
        text = str(value).strip()
        text = re.sub(r"^(?:object\s+name|object|name)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" -_:;,.\n\r\t")
        return text

    def build_prompt(
        self,
        scene_objects: list[dict[str, Any]],
    ) -> str:
        """
        Build a native Locate Anything category prompt from the available
        ontology-grounding information when present.

        This preserves the intended architecture: ontology provides the
        grounding_prompt, and Locate Anything consumes that value instead
        of reconstructing prompts from raw scene labels.
        """

        logger.info(
            "Building Locate Anything grounding prompt."
        )

        object_names: list[str] = []

        for obj in scene_objects:

            if not isinstance(obj, dict):
                logger.warning(
                    "Skipping invalid scene object: {}",
                    obj,
                )
                continue

            grounding_prompt = obj.get("grounding_prompt")

            if not grounding_prompt:
                ontology = obj.get("ontology_reasoning")
                if isinstance(ontology, dict):
                    prediction = ontology.get("prediction", {})
                    if isinstance(prediction, dict):
                        grounding_prompt = prediction.get("grounding_prompt")

            if not grounding_prompt:
                grounding_prompt = obj.get(
                    "observed_object",
                    obj.get("object_name", ""),
                )

            if not grounding_prompt:
                continue

            label = self._normalize_label(grounding_prompt)
            if not label:
                continue

            # DO NOT collapse distinct scene objects merely because they share
            # the same grounding prompt. Repeated prompts are still meaningful
            # when multiple objects of the same category are present in the same
            # image, and the downstream annotation stage must retain that
            # identity through the object-ordering of the prompt list.
            object_names.append(label)

        if not object_names:
            logger.warning(
                "No valid object names found for grounding."
            )
            return ""

        prompt = "</c>".join(object_names)

        logger.info(
            "Created grounding prompt for {} unique object categories.",
            len(object_names),
        )
        logger.info(
            "Locate Anything prompt: {}",
            prompt,
        )

        return prompt
"""
gemma.py

Gemma Vision-Language Model implementation.

This module provides the Gemma-specific implementation of the abstract
Hugging Face Vision-Language Model interface used by the KGFLM Traffic
Annotation Pipeline.

GemmaVL inherits the common loading, unloading, and inference pipeline
implemented by HFVLM. The only Gemma-specific responsibility is defining
the underlying Hugging Face model class and constructing the multimodal
chat message format expected by Gemma.

Pipeline

Image
    │
    ▼
Scene Understanding Prompt
    │
    ▼
GemmaVL
    │
    ▼
HFVLM
    │
    ▼
Gemma3ForConditionalGeneration
    │
    ▼
Raw Scene Description

Responsibilities

• Register the Gemma model implementation.
• Construct Gemma-compatible multimodal chat messages.
• Reuse the common Hugging Face VLM infrastructure.
"""

from __future__ import annotations

# ============================================================================
# Imports
# ============================================================================

from typing import Any

from transformers import Gemma3ForConditionalGeneration

from custom_logger import CustomLogger

from annotation_pipeline.common.models.vlm.hf_vlm import HFVLM

logger = CustomLogger(__name__)


# ============================================================================
# Gemma Vision-Language Model
# ============================================================================


class GemmaVL(HFVLM):
    """
    Gemma Scene Understanding implementation.

    This class adapts Google's Gemma multimodal models to the common
    Vision-Language Model interface used throughout the annotation
    pipeline.

    All model loading, inference, and resource management are inherited
    from HFVLM. This class only provides Gemma-specific configuration
    and message formatting.
    """

    # -------------------------------------------------------------------------
    # Hugging Face Model
    # -------------------------------------------------------------------------

    MODEL_CLASS = Gemma3ForConditionalGeneration

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        config: Any,
    ) -> None:
        """
        Initialize the Gemma Vision-Language Model.

        Parameters
        ----------
        config
            Hugging Face model configuration loaded from the pipeline
            configuration.
        """

        super().__init__(config)

        logger.info(
            "Initialized Gemma Scene Understanding backend '{}'.",
            self.model_name,
        )

    # -------------------------------------------------------------------------
    # Message Construction
    # -------------------------------------------------------------------------

    def build_messages(
        self,
        image: Any,
        prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Construct a Gemma-compatible multimodal conversation.

        The Hugging Face processor expects a chat conversation consisting
        of an image followed by the textual Scene Understanding prompt.

        Parameters
        ----------
        image
            Loaded PIL image.

        prompt
            Scene Understanding prompt.

        Returns
        -------
        list[dict[str, Any]]
            Conversation formatted according to the Gemma chat template.
        """

        logger.debug(
            "Building Gemma chat message."
        )

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

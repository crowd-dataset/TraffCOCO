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

import flash_attn

from transformers import Gemma3ForConditionalGeneration

from custom_logger import CustomLogger

from annotation_pipeline.models.vlm.hf_vlm import HFVLM

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

    DEFAULT_DTYPE = "auto"

    DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"

    DEFAULT_MAX_NEW_TOKENS = 3500

    DEFAULT_TEMPERATURE = 0.0

    DEFAULT_DO_SAMPLE = False

    DEFAULT_QUANTIZATION = "4bit"

    

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        model_id: str,
    ) -> None:
        """
        Initialize the Gemma Vision-Language Model.

        Parameters
        ----------
        config
            Hugging Face model configuration loaded from the pipeline
            configuration.
        """

        super().__init__(model_id)

        logger.info(
            "Initialized Gemma Scene Understanding backend '{}'.",
            self.model_id,
        )

    
"""
smolvlm.py

SmolVLM2 implementation.
"""

from __future__ import annotations


from transformers import SmolVLMForConditionalGeneration
from annotation_pipeline.models.vlm.hf_vlm import HFVLM

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


class SmolVLM(HFVLM):
    """
    SmolVLM2 Scene Understanding implementation.

    Expected config example:

    models:
      scene_understanding:
        backend: smolvlm
        model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    """

    MODEL_CLASS = SmolVLMForConditionalGeneration

    DEFAULT_DEVICE = "cuda"

    DEFAULT_DTYPE = "auto"

    DEFAULT_MAX_NEW_TOKENS = 256

    DEFAULT_TEMPERATURE = 0.0

    DEFAULT_DO_SAMPLE = False

    def __init__(
        self,
        model_id: str,
    ) -> None:

        super().__init__(model_id)

        logger.info(
            "Initialized SmolVLM Scene Understanding backend '{}'.",
            self.model_id,
        )

    
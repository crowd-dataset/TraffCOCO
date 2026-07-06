"""
smolvlm.py

SmolVLM2 implementation.
"""

from __future__ import annotations


from transformers import SmolVLMForConditionalGeneration
from annotation_pipeline.common.models.vlm.hf_vlm import HFVLM

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

    def __init__(self, config):

        super().__init__(config)

        logger.info(
            "Initialized SmolVLM Scene Understanding backend '{}'.",
            self.model_name,
        )

    def build_messages(
        self,
        image,
        prompt: str,
    ) -> list:

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

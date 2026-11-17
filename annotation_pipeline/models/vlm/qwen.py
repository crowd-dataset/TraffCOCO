"""
qwen.py

Qwen3-VL backend for the Scene Understanding stage.

QwenVL reuses the shared Hugging Face loading, batched inference and
cleanup implemented by HFVLM, exactly like the Gemma backends, so the
SceneUnderstandingEngine can drive it through the same interface:

    model = QwenVL(model_id)
    model.load()
    result = model.infer(image_paths=[...], prompt=...)
    model.unload()

``result`` is the HFVLM dictionary with ``responses``, ``generation_time``,
``generated_tokens`` and ``total_tokens``.
"""

from __future__ import annotations

from transformers import Qwen3VLForConditionalGeneration

from custom_logger import CustomLogger

from annotation_pipeline.models.vlm.hf_vlm import HFVLM

logger = CustomLogger(__name__)


class QwenVL(HFVLM):
    """Qwen3-VL Scene Understanding backend (e.g. Qwen/Qwen3-VL-8B-Instruct)."""

    MODEL_CLASS = Qwen3VLForConditionalGeneration

    DEFAULT_DTYPE = "auto"

    DEFAULT_ATTENTION_IMPLEMENTATION = "sdpa"

    DEFAULT_MAX_NEW_TOKENS = 4000

    DEFAULT_TEMPERATURE = 0.0

    DEFAULT_DO_SAMPLE = False

    DEFAULT_QUANTIZATION = "4bit"

    # Greedy decoding without a penalty made Qwen3-VL repeat the same
    # object until max_new_tokens (Spark test, 2026-11-17).
    DEFAULT_REPETITION_PENALTY = 1.05

    def __init__(
        self,
        model_id: str,
    ) -> None:

        super().__init__(model_id)

        logger.info(
            "Initialized Qwen3-VL Scene Understanding backend '{}'.",
            self.model_id,
        )

    def load(self) -> None:
        """Load the model, then pad on the left for batched generation.

        HFVLM slices each generated sequence at the shared input length,
        which is only correct when every prompt in a batch is left-padded.
        """

        super().load()

        tokenizer = getattr(self.processor, "tokenizer", None)

        if tokenizer is not None:
            tokenizer.padding_side = "left"

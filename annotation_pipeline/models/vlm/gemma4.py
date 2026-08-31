from transformers import AutoModelForImageTextToText

from custom_logger import CustomLogger

from annotation_pipeline.models.vlm.hf_vlm import HFVLM


logger = CustomLogger(__name__)


class Gemma4VL(HFVLM):

    MODEL_CLASS = AutoModelForImageTextToText

    DEFAULT_DTYPE = "auto"

    DEFAULT_ATTENTION_IMPLEMENTATION = "sdpa"

    DEFAULT_MAX_NEW_TOKENS = 4000

    DEFAULT_TEMPERATURE = 0.0

    DEFAULT_DO_SAMPLE = False

    DEFAULT_QUANTIZATION = "4bit"

    def __init__(
        self,
        model_id: str,
    ) -> None:

        super().__init__(model_id)

        logger.info(
            "Initialized Gemma 4 Scene Understanding backend '{}'.",
            self.model_id,
        )
"""
hf_vlm.py

Common Hugging Face Vision-Language Model implementation used by the
KGFLM Traffic Annotation Pipeline.

This module provides a reusable implementation of the BaseVLM interface
for Hugging Face Vision-LLanguage Models. It encapsulates all framework-
specific functionality including model loading, processor creation,
resource management and inference.

Concrete model implementations such as GemmaVL and SmolVLM inherit from
this class and only provide:

    • MODEL_CLASS
    • Optional message formatting

Pipeline

Image
    │
    ▼
Build Messages
    │
    ▼
Processor
    │
    ▼
Tokenizer
    │
    ▼
Model Generation
    │
    ▼
Raw Scene Description

Responsibilities

• Load Hugging Face processors.
• Load model weights.
• Configure quantization.
• Configure attention implementation.
• Execute Scene Understanding inference.
• Release CPU/GPU resources.
• Provide a common inference interface for every Hugging Face VLM.
"""

from __future__ import annotations

import gc
from custom_logger import CustomLogger
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
)

from annotation_pipeline.models.vlm.base_vlm import BaseVLM

logger = CustomLogger(__name__)


class HFVLM(BaseVLM):
    """
    Base class for HuggingFace Vision-Language Models.
    """

    # ------------------------------------------------------------------
    # Child classes MUST override
    # ------------------------------------------------------------------

    MODEL_CLASS = None

    DEFAULT_DEVICE = "cuda"

    DEFAULT_DTYPE = "bfloat16"

    DEFAULT_ATTENTION_IMPLEMENTATION = "eager"

    DEFAULT_TRUST_REMOTE_CODE = True

    DEFAULT_COMPILE = False

    DEFAULT_MAX_NEW_TOKENS = 512

    DEFAULT_TEMPERATURE = 0.0

    DEFAULT_QUANTIZATION = None

    DEFAULT_DO_SAMPLE = False

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    """
    Initialize the Hugging Face Vision-Language Model.

    The configuration object is populated from default.config and
    contains every model-specific parameter required for loading and
    executing inference.
    """

    def __init__(
        self,
        model_id: str,
    ):
        super().__init__(model_id)

        self.model_id = model_id

        self.device = self.DEFAULT_DEVICE

        if self.DEFAULT_DTYPE == "auto":
            if torch.cuda.is_available():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = getattr(torch, self.DEFAULT_DTYPE)

        self.attention = self.DEFAULT_ATTENTION_IMPLEMENTATION

        self.trust_remote_code = self.DEFAULT_TRUST_REMOTE_CODE

        self.compile = self.DEFAULT_COMPILE

        self.max_new_tokens = self.DEFAULT_MAX_NEW_TOKENS

        self.temperature = self.DEFAULT_TEMPERATURE

        self.quantization = self.DEFAULT_QUANTIZATION

        self.do_sample = self.DEFAULT_DO_SAMPLE

        logger.debug(
            "Initialized Hugging Face backend '{}'.",
            self.model_id,
        )
        

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        """
        Load the configured Hugging Face Vision-Language Model.

        The loading procedure initializes:

        • Processor
        • Tokenizer
        • Model weights
        • Quantization configuration
        • Attention implementation

        The model is placed on the configured execution device and switched
        to evaluation mode.
        """
        logger.info(
            "Loading Hugging Face model '{}'.",
            self.model_id,
        )

        if self.MODEL_CLASS is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must define MODEL_CLASS."
            )

        logger.info("Loading {} ...", self.model_id)

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        logger.debug(
            "Processor loaded successfully."
        )
        logger.debug(
            "Preparing model loading arguments."
        )

        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": "auto",
            "attn_implementation": self.attention,
        }

        # --------------------------------------------------------------
        # Quantization
        # --------------------------------------------------------------

        if self.quantization is None:

            model_kwargs["dtype"] = self.dtype
            logger.debug(
                "Using dtype '{}'.",
                self.dtype,
            )

        elif self.quantization == "4bit":

            logger.info("Using 4-bit quantization.")

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        elif self.quantization == "8bit":

            logger.info("Using 8-bit quantization.")

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        else:

            raise ValueError(
                f"Unsupported quantization: {self.quantization}"
            )
        logger.info(
            "Loading model weights..."
        )
        self.model = self.MODEL_CLASS.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        logger.debug(
            "Switching model to evaluation mode."
        )
        self.model.eval()

        self.loaded = True

        logger.info(
            "Successfully loaded '{}'.",
            self.model_id,
        )
        logger.info(
            "Model dtype: {}",
            next(self.model.parameters()).dtype,
        )
    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def unload(self):

        if not self.loaded:
            return

        logger.info("{} unloaded.", self.model_id)

        del self.model
        del self.processor

        self.model = None
        self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.loaded = False

        logger.info(
            "Unloading '{}'.",
            self.model_id,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        image_path: Path,
        prompt: str,
    ) -> str:
        logger.info(
            "Running Scene Understanding inference on '{}'.",
            image_path.name,
        )
        if not self.loaded:
            raise RuntimeError(
                "Model has not been loaded."
            )

        image = Image.open(image_path).convert("RGB")
        logger.debug(
            "Loaded image '{}'.",
            image_path.name,
        )
        messages = self.build_messages(
            image=image,
            prompt=prompt,
        )
        logger.debug(
            "Constructed multimodal prompt."
        )
        chat_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        logger.debug(
            "Applied Hugging Face chat template."
        )
        inputs = self.processor(
            images=image,
            text=chat_prompt,
            return_tensors="pt",
        )
        logger.debug(
            "Prepared model inputs."
        )
        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
        }
        for k, v in inputs.items():
            if torch.is_tensor(v):
                logger.info("{} -> device={}, dtype={}", k, v.device, v.dtype)

        with torch.no_grad():
            logger.info(
                "Generating scene description..."
            )
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
            }

            # Only use temperature when sampling
            if self.do_sample:
                generate_kwargs["temperature"] = self.temperature

            # Reduce repetition
            generate_kwargs["repetition_penalty"] = 1.15
            generate_kwargs["no_repeat_ngram_size"] = 3

            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_ids = outputs[:, input_length:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        logger.info(
            "Model inference completed."
        )
        logger.info("=" * 80)
        logger.info("RAW MODEL OUTPUT")
        logger.info("=" * 80)
        logger.info(
            "{}",
            response,
        )
        logger.info("=" * 80)

        return response.strip()

    # ------------------------------------------------------------------
    # Prompt Builder
    # ------------------------------------------------------------------

    def build_messages(
        self,
        image: Any,
        prompt: str,
    ) -> list:
        """
        Construct the default multimodal conversation.

        Most Hugging Face Vision-Language Models accept a conversation
        consisting of an image followed by the textual Scene Understanding
        prompt.

        Models requiring a different conversation format should override
        this method.
        """
        logger.debug(
            "Building default multimodal conversation."
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

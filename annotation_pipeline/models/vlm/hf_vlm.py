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
import time

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

        # ----------------------------------------------------------
        # Benchmark statistics
        # ----------------------------------------------------------

        self.last_generation_time = 0.0

        self.last_generated_tokens = 0

        self.last_total_tokens = 0

        logger.debug(
            "Initialized Hugging Face backend '{}'.",
            self.model_id,
        )
        

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load the Hugging Face Vision-Language Model.

        This implementation follows the benchmark implementation exactly.
        """

        if self.loaded:
            logger.debug(
                "Model '{}' already loaded.",
                self.model_id,
            )
            return

        logger.info("=" * 80)
        logger.info("LOADING MODEL")
        logger.info("=" * 80)

        # ----------------------------------------------------------
        # Processor
        # ----------------------------------------------------------

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            use_fast=True,
        )

        # ----------------------------------------------------------
        # Quantization
        # ----------------------------------------------------------

        quantization_config = None

        if self.quantization == "4bit":

            logger.info(
                "Using 4-bit quantization."
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        elif self.quantization == "8bit":

            logger.info(
                "Using 8-bit quantization."
            )

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # ----------------------------------------------------------
        # Model
        # ----------------------------------------------------------

        self.model = self.MODEL_CLASS.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=self.attention,
        )

        self.model.eval()

        # ----------------------------------------------------------
        # Debug
        # ----------------------------------------------------------

        logger.info("=" * 80)
        logger.info("MODEL CONFIG")
        logger.info("=" * 80)

        try:

            logger.info(
                "Device      : {}",
                self.model.device,
            )

        except Exception:

            logger.info(
                "Device      : <device_map>"
            )

        logger.info(
            "Quantized   : {}",
            hasattr(
                self.model,
                "hf_quantizer",
            ),
        )

        logger.info(
            "Attention   : {}",
            self.model.config._attn_implementation,
        )

        logger.info(
            "GPU Allocated : {:.2f} GB",
            torch.cuda.memory_allocated() / 1024**3,
        )

        logger.info(
            "GPU Reserved  : {:.2f} GB",
            torch.cuda.memory_reserved() / 1024**3,
        )

        logger.info(
            "HF Device Map:"
        )

        logger.info(
            "{}",
            self.model.hf_device_map,
        )

        self.loaded = True

        logger.info(
            "'{}' loaded successfully.",
            self.model_id,
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
        image_paths: list[Path] | Path,
        prompt: str,
    ):
        if isinstance(image_paths, Path):

            image_paths = [image_paths]

        logger.info(
            "Running Scene Understanding on {} image(s).",
            len(image_paths),
        )

        if not self.loaded:

            raise RuntimeError(
                "Model has not been loaded."
            )

        logger.info(
            "Images in batch: {}",
            ", ".join(path.name for path in image_paths),
        )

        inputs = self._prepare_inputs(
            image_paths=image_paths,
            prompt=prompt,
        )

        generated, generation_time = self._generate(
            inputs,
        )

        responses = self._decode(
            generated,
        )

        self.last_generation_time = generation_time

        self.last_generated_tokens = generated.shape[1]

        self.last_total_tokens = generated.numel()

        logger.info(
            "Generation Time : {:.2f} s",
            generation_time,
        )

        logger.info(
            "Generated Tokens : {}",
            generated.shape[1],
        )

        logger.info(
            "{}",
            responses,
        )

        return {

            "responses": responses,

            "generation_time": generation_time,

            "generated_tokens": generated.shape[1],

            "total_tokens": generated.numel(),

        }

    # ------------------------------------------------------------------
    # Prompt Builder
    # ------------------------------------------------------------------
    
    def _prepare_inputs(
        self,
        image_paths: Path,
        prompt: str,
    ):
        """
        Construct the default multimodal conversation.

        Most Hugging Face Vision-Language Models accept a conversation
        consisting of an image followed by the textual Scene Understanding
        prompt.

        Models requiring a different conversation format should override
        this method.
        """

        images = [
            [Image.open(path).convert("RGB")]
            for path in image_paths
        ]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        chat_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [chat_prompt] * len(images)

        inputs = self.processor(

            text=prompts,

            images=images,

            padding=True,

            return_tensors="pt",

        )

        device = self.model.get_input_embeddings().weight.device

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        return inputs

    def _generate(
        self,
        inputs,
    ):
        """
        Execute model.generate() exactly as in benchmark.
        """

        start = time.perf_counter()

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        generation_time = time.perf_counter() - start

        input_len = inputs["input_ids"].shape[1]

        generated = outputs[:, input_len:]

        return (
            generated,
            generation_time,
        )


    def _decode(
        self,
        generated,
    ):
        """
        Decode benchmark output.
        """

        responses = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        return responses
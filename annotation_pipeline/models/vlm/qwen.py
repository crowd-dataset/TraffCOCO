from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from .base_vlm import BaseVLM

logger = logging.getLogger(__name__)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


class QwenVLM(BaseVLM):
    """
    Qwen3-VL backend for traffic scene understanding.

    Loads Qwen3-VL locally through Hugging Face Transformers and
    exposes the same interface expected by the VLM pipeline.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quantized: bool = True,
        attn_implementation: str = "flash_attention_2",
    ) -> None:

        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.quantized = quantized
        self.attn_implementation = attn_implementation

        logger.info(
            "Initializing Qwen VLM backend '%s'.",
            self.model_id,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
        )

        quantization_config = None

        if self.quantized:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "dtype": self.dtype,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = (
                quantization_config
            )

        if self.attn_implementation:
            model_kwargs["attn_implementation"] = (
                self.attn_implementation
            )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        self.model.eval()

        logger.info(
            "Qwen VLM loaded successfully."
        )

        logger.info(
            "Model device: %s",
            self.model.device,
        )

        logger.info(
            "Quantized: %s",
            self.quantized,
        )

        logger.info(
            "Attention: %s",
            getattr(
                self.model.config,
                "_attn_implementation",
                "unknown",
            ),
        )

    def generate(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        max_new_tokens: int = 3000,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response for a single image.
        """

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        messages = [
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

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = self.model.device

        inputs = {
            key: value.to(device)
            if torch.is_tensor(value)
            else value
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                **kwargs,
            )

        input_length = inputs["input_ids"].shape[-1]

        generated_ids = outputs[:, input_length:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response

    def generate_batch(
        self,
        images: list[Image.Image | str | Path],
        prompt: str,
        max_new_tokens: int = 3000,
        **kwargs: Any,
    ) -> list[str]:
        """
        Generate responses for a batch of images.
        """

        pil_images: list[Image.Image] = []

        for image in images:

            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")

            pil_images.append(image)

        messages = []

        for image in pil_images:
            messages.append(
                [
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
            )

        inputs = []

        for message in messages:
            processed = self.processor.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            inputs.append(processed)

        # Qwen's multimodal processor handles batched image/text
        # inputs more reliably when constructed directly.
        chat_messages = [
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
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompts = [chat_prompt] * len(pil_images)

        batch_inputs = self.processor(
            text=prompts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )

        device = self.model.device

        batch_inputs = {
            key: value.to(device)
            if torch.is_tensor(value)
            else value
            for key, value in batch_inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                **kwargs,
            )

        input_lengths = batch_inputs["attention_mask"].sum(
            dim=1
        )

        responses = []

        for output, input_length in zip(
            outputs,
            input_lengths,
        ):

            generated = output[int(input_length):]

            response = self.processor.decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            responses.append(response)

        return responses
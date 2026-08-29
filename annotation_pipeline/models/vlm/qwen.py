"""
qwen.py

Local Qwen3-VL backend for the Scene Understanding stage.

This implementation conforms to the existing BaseVLM interface:

    load()
    unload()
    infer(image_path, prompt)

The SceneUnderstandingEngine owns the model lifecycle. This class must
therefore NOT load the model in __init__().
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from custom_logger import CustomLogger

from .base_vlm import BaseVLM

logger = CustomLogger(__name__)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


class QwenVL(BaseVLM):
    """
    Qwen3-VL-8B-Instruct backend.

    The class follows the common BaseVLM lifecycle used by the pipeline:

        model = QwenVL(...)
        model.load()
        response = model.infer(image_path, prompt)
        model.unload()
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        quantized: bool = True,
        attn_implementation: str = "flash_attention_2",
        max_new_tokens: int = 3000,
    ) -> None:

        super().__init__(config=None)

        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.quantized = quantized
        self.attn_implementation = attn_implementation
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.processor = None
        self.loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load Qwen processor and model into memory.
        """

        if self.loaded:
            logger.info(
                "Qwen '{}' is already loaded.",
                self.model_id,
            )
            return

        logger.info("=" * 80)
        logger.info("LOADING QWEN")
        logger.info("=" * 80)

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but unavailable. Falling back to CPU."
            )
            self.device = "cpu"

        logger.info(
            "Loading processor: {}",
            self.model_id,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
        )

        quantization_config = None

        if self.quantized and self.device == "cuda":

            logger.info(
                "Using 4-bit Qwen quantization."
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )

        model_kwargs: dict[str, Any] = {
            "device_map": "auto" if self.device == "cuda" else self.device,
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

        logger.info(
            "Loading model: {}",
            self.model_id,
        )

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        self.model.eval()
        self.loaded = True

        logger.info(
            "Qwen loaded successfully: {}",
            self.model_id,
        )

        if torch.cuda.is_available():
            logger.info(
                "GPU Allocated : {:.2f} GB",
                torch.cuda.memory_allocated() / (1024 ** 3),
            )

            logger.info(
                "GPU Reserved  : {:.2f} GB",
                torch.cuda.memory_reserved() / (1024 ** 3),
            )

    def unload(self) -> None:
        """
        Release Qwen model and CUDA memory.
        """

        if not self.loaded and self.model is None:
            return

        logger.info(
            "Unloading Qwen '{}'.",
            self.model_id,
        )

        self.model = None
        self.processor = None
        self.loaded = False

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        logger.info(
            "Qwen unloaded."
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        image_path: Path,
        prompt: str,
    ) -> str:
        """
        Perform Scene Understanding on a single image.

        This signature intentionally matches BaseVLM exactly.
        """

        if not self.loaded or self.model is None:
            raise RuntimeError(
                "Qwen model is not loaded. Call load() before infer()."
            )

        if self.processor is None:
            raise RuntimeError(
                "Qwen processor is not loaded."
            )

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Qwen image not found: {image_path}"
            )

        logger.info(
            "Qwen image: {}",
            image_path.name,
        )

        image = Image.open(
            image_path,
        ).convert("RGB")

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

        # Qwen's processor constructs the multimodal input from the
        # chat template and image.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_device = self._input_device()

        inputs = {
            key: value.to(input_device)
            if torch.is_tensor(value)
            else value
            for key, value in inputs.items()
        }

        start_time = time.perf_counter()

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        generation_time = (
            time.perf_counter() - start_time
        )

        input_length = inputs["input_ids"].shape[-1]

        generated_ids = outputs[:, input_length:]

        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        generated_tokens = generated_ids.shape[-1]

        logger.info(
            "Qwen generation completed in {:.2f}s.",
            generation_time,
        )

        logger.info(
            "Generated Tokens : {}",
            generated_tokens,
        )

        return response

    # ------------------------------------------------------------------
    # Optional batch inference
    # ------------------------------------------------------------------

    def infer_batch(
        self,
        image_paths: list[Path],
        prompts: list[str],
    ) -> list[str]:
        """
        Batch interface expected by SceneUnderstandingEngine.

        The engine supplies image_paths and prompts as keyword arguments.
        Inference is intentionally performed one image at a time to avoid
        unnecessary GPU memory pressure with Qwen3-VL-8B.
        """

        if len(image_paths) != len(prompts):
            raise ValueError(
                f"Number of images ({len(image_paths)}) does not match "
                f"number of prompts ({len(prompts)})."
            )

        responses: list[str] = []

        for image_path, prompt in zip(
            image_paths,
            prompts,
        ):
            response = self.infer(
                image_path=Path(image_path),
                prompt=prompt,
            )

            responses.append(response)

        return responses

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _input_device(self) -> torch.device:
        """
        Determine the device to which processor tensors should be moved.

        For device_map='auto', model.device normally points to the primary
        execution device. Fall back to the first model parameter if needed.
        """

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                return self.model.device
            except Exception:
                pass

            try:
                return next(
                    self.model.parameters()
                ).device
            except StopIteration:
                return torch.device("cuda")

        return torch.device("cpu")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_id}', "
            f"loaded={self.loaded}, "
            f"device={self.device}, "
            f"quantized={self.quantized})"
        )
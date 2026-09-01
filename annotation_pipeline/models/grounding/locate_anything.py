"""
locate_anything.py

LocateAnything model wrapper.

This module encapsulates the NVIDIA LocateAnything-3B inference
runtime used by the VLM-first annotation pipeline.

Inference backend
-----------------
NVIDIA LocateAnything batch hybrid runtime with LA Flash attention.

The NVIDIA runtime is kept isolated inside this wrapper so that the
pipeline layer does not directly interact with HuggingFace or the
LocateAnything-specific batch runtime.

Expected runtime directory
--------------------------
<project_root>/locateanything_runtime/

Containing:
    batch_infer.py
    batch_utils/
    kernel_utils/
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from custom_logger import CustomLogger

from annotation_pipeline.configs.settings import (
    PipelineConfig,
)

logger = CustomLogger(__name__)


class LocateAnything:
    """
    NVIDIA LocateAnything-3B model wrapper.

    Uses NVIDIA's official batch hybrid runtime with the
    LA Flash sparse attention backend.
    """

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:

        self.config = config

        models = config.models

        self.model_id = (
            models.locate_anything_model_id
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):

            self.dtype = torch.bfloat16

        else:

            self.dtype = torch.float16

        # ----------------------------------------------------------
        # LocateAnything generation settings
        # ----------------------------------------------------------

        self.generation_mode = "hybrid"

        self.max_new_tokens = 1024
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = None
        self.repetition_penalty = 1.1
        self.scheduler = "pipeline"
        self.group_size = 0
        self.attention_backend = "la_flash"

        self.vision_attention_backend = "auto"

        self.strict_attention = True

        # ----------------------------------------------------------
        # NVIDIA runtime
        # ----------------------------------------------------------

        self.runtime_dir = self._resolve_runtime_dir()

        self.batch_utils = None

        self.generate_batch_hybrid_fn = None

        self.get_last_hybrid_stats_fn = None

        # ----------------------------------------------------------
        # Model components
        # ----------------------------------------------------------

        self.tokenizer = None

        self.processor = None

        self.model = None

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        self.last_inference_time = 0.0

        self.last_batch_size = 0

        self.last_generated_tokens = 0

        self.last_hybrid_stats = None

        # ----------------------------------------------------------
        # Initialization
        # ----------------------------------------------------------

        logger.info(
            "Initializing LocateAnything model."
        )

        logger.info(
            "LA runtime configuration: "
            "attn={}, scheduler={}, group_size={}, "
            "max_new_tokens={}, temperature={}, top_p={}, top_k={}, "
            "repetition_penalty={}",
            os.environ.get("LA_FLASH_ATTN"),
            self.scheduler,
            self.group_size,
            self.max_new_tokens,
            self.temperature,
            self.top_p,
            self.top_k,
            self.repetition_penalty,
        )

        logger.info(
            "Model ID: {}",
            self.model_id,
        )

        logger.info(
            "Attention backend: {}",
            self.attention_backend,
        )

        logger.info(
            "Hybrid scheduler: {}",
            self.scheduler,
        )

        self._load_model()

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def _resolve_runtime_dir(
        self,
    ) -> Path:
        """
        Resolve the exact NVIDIA LocateAnything runtime used
        by the validated direct inference test.

        Priority
        --------
        1. LOCATEANYTHING_RUNTIME environment variable
        2. <project_root>/locateanything_runtime
        """

        environment_path = os.environ.get(
            "LOCATEANYTHING_RUNTIME",
        )

        if environment_path:

            runtime_dir = (
                Path(environment_path)
                .expanduser()
                .resolve()
            )

        else:

            project_root = (
                Path(__file__)
                .resolve()
                .parents[3]
            )

            runtime_dir = (
                project_root
                / "locateanything_runtime"
            )

        return runtime_dir

    def _load_runtime(
        self,
    ) -> None:
        """
        Load NVIDIA's local LocateAnything batch runtime.
        """

        logger.info(
            "Loading NVIDIA LocateAnything runtime."
        )

        if not self.runtime_dir.exists():

            raise FileNotFoundError(
                "LocateAnything runtime directory not found: "
                f"{self.runtime_dir}"
            )

        batch_utils_dir = (
            self.runtime_dir
            / "batch_utils"
        )

        kernel_utils_dir = (
            self.runtime_dir
            / "kernel_utils"
        )

        if not batch_utils_dir.exists():

            raise FileNotFoundError(
                "LocateAnything batch_utils directory not found: "
                f"{batch_utils_dir}"
            )

        if not kernel_utils_dir.exists():

            raise FileNotFoundError(
                "LocateAnything kernel_utils directory not found: "
                f"{kernel_utils_dir}"
            )

        # ----------------------------------------------------------
        # Put NVIDIA runtime at the front of sys.path.
        #
        # This ensures that the locally downloaded batch_utils and
        # kernel_utils are used rather than an unrelated package.
        # ----------------------------------------------------------

        runtime_string = str(
            self.runtime_dir
        )

        if runtime_string not in sys.path:

            sys.path.insert(
                0,
                runtime_string,
            )

        # ----------------------------------------------------------
        # Configure NVIDIA runtime BEFORE importing batch_utils.
        # ----------------------------------------------------------

        os.environ[
            "LA_FLASH_MODEL"
        ] = self.model_id

        os.environ[
            "LA_FLASH_ATTN"
        ] = self.attention_backend

        os.environ[
            "LA_FLASH_VISION_ATTN"
        ] = self.vision_attention_backend

        os.environ[
            "LA_FLASH_HYBRID_SCHEDULER"
        ] = self.scheduler

        os.environ[
            "LA_FLASH_HYBRID_GROUP_SIZE"
        ] = str(
            self.group_size
        )

        os.environ[
            "LA_FLASH_STRICT_ATTN"
        ] = (
            "1"
            if self.strict_attention
            else "0"
        )

        # Keep the dense/prefill backend on SDPA.
        #
        # LA Flash is the sparse range attention backend.
        os.environ[
            "LA_FLASH_DENSE_BACKEND"
        ] = "sdpa"

        logger.info(
            "NVIDIA runtime directory: {}",
            self.runtime_dir,
        )

        logger.info(
            "LA_FLASH_ATTN={}",
            os.environ[
                "LA_FLASH_ATTN"
            ],
        )

        logger.info(
            "LA_FLASH_VISION_ATTN={}",
            os.environ[
                "LA_FLASH_VISION_ATTN"
            ],
        )

        # ----------------------------------------------------------
        # Import NVIDIA runtime.
        # ----------------------------------------------------------

        from batch_utils import (
            generate_batch_hybrid,
            get_last_hybrid_stats,
            load,
        )

        self.batch_utils = load

        self.generate_batch_hybrid_fn = (
            generate_batch_hybrid
        )

        self.get_last_hybrid_stats_fn = (
            get_last_hybrid_stats
        )

        # ----------------------------------------------------------
        # Load model through NVIDIA runtime.
        # ----------------------------------------------------------

        (
            self.tokenizer,
            self.processor,
            self.model,
        ) = self.batch_utils()

        if self.model is None:

            raise RuntimeError(
                "NVIDIA LocateAnything runtime returned "
                "no model."
            )

        logger.info(
            "NVIDIA LocateAnything runtime loaded."
        )

        logger.info(
            "LA Flash attention backend is active."
        )

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def _load_model(
        self,
    ) -> None:
        """
        Load LocateAnything through NVIDIA's official runtime.

        Do NOT use AutoModel.from_pretrained() here.

        The normal Transformers generation path was responsible for
        the previous flash_attention_2 / Qwen attention failures.
        NVIDIA's batch runtime configures the hybrid decoder and
        LA Flash backend correctly.
        """

        if not torch.cuda.is_available():

            raise RuntimeError(
                "LocateAnything LA Flash requires CUDA. "
                "No CUDA device is available."
            )

        self._load_runtime()

    # ------------------------------------------------------------------
    # Input Preparation
    # ------------------------------------------------------------------

    def _prepare_image(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Normalize an input image exactly as NVIDIA's
        LocateAnything runtime does in load_pil().

        The NVIDIA runtime limits the longest image dimension
        to MAX_DIM=1024 while preserving aspect ratio.
        """

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "LocateAnything expects a PIL.Image.Image "
                "instance."
            )

        if image.mode != "RGB":
            image = image.convert("RGB")

        max_dim = 1024

        width, height = image.size

        if max(width, height) > max_dim:

            scale = (
                max_dim
                / max(width, height)
            )

            new_width = max(
                1,
                round(width * scale),
            )

            new_height = max(
                1,
                round(height * scale),
            )

            image = image.resize(
                (
                    new_width,
                    new_height,
                ),
                Image.Resampling.LANCZOS,
            )

        return image

    # ------------------------------------------------------------------
    # Single Image Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        """
        Run LocateAnything on a single image.

        Parameters
        ----------
        image
            RGB PIL image.

        prompt
            LocateAnything grounding query.

        Returns
        -------
        str
            Raw LocateAnything grounding response.
        """

        start = time.perf_counter()

        image = self._prepare_image(
            image
        )

        if not isinstance(
            prompt,
            str,
        ):

            raise TypeError(
                "LocateAnything prompt must be a string."
            )

        if not prompt.strip():

            raise ValueError(
                "LocateAnything prompt cannot be empty."
            )

        # ----------------------------------------------------------
        # NVIDIA runtime expects:
        #
        #     [(PIL image, query)]
        #
        # generate_batch_hybrid() returns:
        #
        #     list[str]
        # ----------------------------------------------------------

        logger.info(
            "LocateAnything prompt sent to runtime: {!r}",
            prompt,
        )

        logger.info(
            "LocateAnything image sent to runtime: size={}, mode={}",
            image.size,
            image.mode,
        )

        outputs = (
            self.generate_batch_hybrid_fn(
                [
                    (
                        image,
                        prompt,
                    )
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=(
                    self.repetition_penalty
                ),
                max_new_tokens=(
                    self.max_new_tokens
                ),
                scheduler=self.scheduler,
                group_size=self.group_size,
            )
        )

        if not outputs:

            raise RuntimeError(
                "LocateAnything returned no output."
            )

        answer = outputs[0]

        if not isinstance(
            answer,
            str,
        ):

            answer = str(
                answer
            )

        self.last_inference_time = (
            time.perf_counter()
            - start
        )

        self.last_batch_size = 1

        self.last_generated_tokens = (
            self._estimate_token_count(
                answer
            )
        )

        self._update_hybrid_stats()

        logger.info(
            "LocateAnything inference completed "
            "in {:.2f} s.",
            self.last_inference_time,
        )

        return answer

    # ------------------------------------------------------------------
    # Batch Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_batch(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> list[str]:
        """
        Run LocateAnything inference using NVIDIA's
        batched hybrid runtime.

        Parameters
        ----------
        images
            RGB PIL images.

        prompts
            Prompt corresponding to each image.

        Returns
        -------
        list[str]
            Raw LocateAnything outputs.
        """

        if len(images) != len(prompts):

            raise ValueError(
                "Number of images and prompts must match. "
                f"Received {len(images)} image(s) and "
                f"{len(prompts)} prompt(s)."
            )

        if not images:

            return []

        logger.info(
            "Running LocateAnything LA Flash on "
            "batch of {} image(s).",
            len(images),
        )

        start = time.perf_counter()

        # ----------------------------------------------------------
        # Normalize images and validate prompts.
        # ----------------------------------------------------------

        pairs = []

        for index, (
            image,
            prompt,
        ) in enumerate(
            zip(
                images,
                prompts,
            ),
            start=1,
        ):

            image = self._prepare_image(
                image
            )

            if not isinstance(
                prompt,
                str,
            ):

                raise TypeError(
                    "Prompt at batch index {} is not a string.".format(
                        index
                    )
                )

            if not prompt.strip():

                raise ValueError(
                    "Prompt at batch index {} is empty.".format(
                        index
                    )
                )

            pairs.append(
                (
                    image,
                    prompt,
                )
            )

        # ----------------------------------------------------------
        # NVIDIA LA Flash hybrid generation.
        # ----------------------------------------------------------

        outputs = (
            self.generate_batch_hybrid_fn(
                pairs,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=(
                    self.repetition_penalty
                ),
                max_new_tokens=(
                    self.max_new_tokens
                ),
                scheduler=self.scheduler,
                group_size=self.group_size,
            )
        )

        if outputs is None:

            raise RuntimeError(
                "LocateAnything NVIDIA runtime returned None."
            )

        outputs = list(
            outputs
        )

        if len(outputs) != len(images):

            raise RuntimeError(
                "LocateAnything returned an unexpected "
                "number of outputs. "
                f"Expected {len(images)}, "
                f"received {len(outputs)}."
            )

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        self.last_batch_size = (
            len(images)
        )

        self.last_inference_time = (
            time.perf_counter()
            - start
        )

        self.last_generated_tokens = sum(
            self._estimate_token_count(
                output
            )
            for output in outputs
        )

        self._update_hybrid_stats()

        logger.info(
            "LocateAnything LA Flash batch completed "
            "in {:.2f} s.",
            self.last_inference_time,
        )

        return [
            str(output)
            for output in outputs
        ]

    # ------------------------------------------------------------------
    # Runtime Statistics
    # ------------------------------------------------------------------

    def _update_hybrid_stats(
        self,
    ) -> None:
        """
        Retrieve statistics from the NVIDIA hybrid runtime.
        """

        if (
            self.get_last_hybrid_stats_fn
            is None
        ):

            return

        try:

            self.last_hybrid_stats = (
                self.get_last_hybrid_stats_fn()
            )

        except Exception as exc:

            logger.warning(
                "Could not retrieve NVIDIA hybrid "
                "runtime statistics: {}",
                exc,
            )

            self.last_hybrid_stats = None

    def _estimate_token_count(
        self,
        text: str,
    ) -> int:
        """
        Estimate generated token count.

        The NVIDIA batch runtime returns decoded strings rather
        than generated token IDs through its public API.

        If the tokenizer is available, use it. Otherwise return 0.
        """

        if (
            self.tokenizer is None
            or not isinstance(
                text,
                str,
            )
        ):

            return 0

        try:

            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
            )

            return len(
                encoded.get(
                    "input_ids",
                    [],
                )
            )

        except Exception:

            return 0

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def ready(
        self,
    ) -> bool:
        """
        Return whether the LocateAnything runtime is loaded.
        """

        return (
            self.model is not None
            and self.processor is not None
            and self.tokenizer is not None
            and self.generate_batch_hybrid_fn is not None
        )

    def unload(
        self,
    ) -> None:
        """
        Free LocateAnything resources and GPU memory.
        """

        logger.info(
            "Unloading LocateAnything model."
        )

        self.model = None

        self.processor = None

        self.tokenizer = None

        self.batch_utils = None

        self.generate_batch_hybrid_fn = None

        self.get_last_hybrid_stats_fn = None

        self.last_hybrid_stats = None

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            torch.cuda.ipc_collect()

        logger.info(
            "LocateAnything unloaded."
        )

    def print_statistics(
        self,
    ) -> None:
        """
        Print LocateAnything inference statistics.
        """

        logger.info(
            "=" * 60,
        )

        logger.info(
            "LocateAnything Statistics"
        )

        logger.info(
            "=" * 60,
        )

        logger.info(
            "Backend          : {}",
            self.attention_backend,
        )

        logger.info(
            "Scheduler        : {}",
            self.scheduler,
        )

        logger.info(
            "Last Batch Size  : {}",
            self.last_batch_size,
        )

        logger.info(
            "Inference Time   : {:.2f} s",
            self.last_inference_time,
        )

        logger.info(
            "Generated Tokens : {}",
            self.last_generated_tokens,
        )

        if self.last_hybrid_stats is not None:

            logger.info(
                "Hybrid Stats     : {}",
                self.last_hybrid_stats,
            )

        if torch.cuda.is_available():

            logger.info(
                "Allocated        : {:.2f} GB",
                torch.cuda.memory_allocated()
                / 1024**3,
            )

            logger.info(
                "Reserved         : {:.2f} GB",
                torch.cuda.memory_reserved()
                / 1024**3,
            )
"""
model_loader.py

Factory for Scene Understanding Vision-Language Models.

Creates the configured local VLM backend based on the pipeline
configuration.

The returned model implements BaseVLM but is NOT loaded into memory.

Usage
-----
from annotation_pipeline.common.models import create_scene_model

model = create_scene_model(config)
model.load()

response = model.infer(image_path, prompt)
"""

from __future__ import annotations

from custom_logger import CustomLogger

from annotation_pipeline.common.configs.settings import PipelineConfig

from annotation_pipeline.common.models.vlm.gemma import GemmaVL
from annotation_pipeline.common.models.vlm.smolvlm import SmolVLM

logger = CustomLogger(__name__)


_SUPPORTED_MODELS = {
    "gemma": GemmaVL,
    "smolvlm": SmolVLM,
}


def create_scene_model(config: PipelineConfig):
    """
    Instantiate the configured Scene Understanding model.

    Parameters
    ----------
    config
        Pipeline configuration.

    Returns
    -------
    BaseVLM
        Configured VLM instance.
    """

    scene_cfg = config.models.scene_understanding

    backend = scene_cfg.backend.lower()

    if backend not in _SUPPORTED_MODELS:

        supported = ", ".join(sorted(_SUPPORTED_MODELS.keys()))

        raise ValueError(
            f"Unsupported Scene Understanding backend '{backend}'. "
            f"Supported backends: {supported}"
        )

    logger.info("Scene Understanding backend: {}", backend)

    # Active model configuration selected by the backend
    model_cfg = scene_cfg.active

    model_cls = _SUPPORTED_MODELS[backend]

    return model_cls(model_cfg)

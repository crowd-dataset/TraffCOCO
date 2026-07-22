"""
common.models

Reusable Vision-Language Models used by the annotation pipeline.

This package provides interchangeable local VLM backends for the
Scene Understanding stage. Every model implements the same BaseVLM
interface, allowing models to be swapped through configuration
without modifying pipeline logic.

Supported models:
    - Qwen2.5-VL
    - Gemma (local)
    - SmolVLM
    - MiniCPM-V
    - InternVL (optional)

Typical usage:

    from annotation_pipeline.common.models.model_loader import create_scene_model

    model = create_scene_model(config)
    model.load()

    response = model.infer(image_path, prompt)

    model.unload()
"""

from annotation_pipeline.models.vlm.base_vlm import BaseVLM
from annotation_pipeline.models.vlm.model_loader import create_scene_model

__all__ = [
    "BaseVLM",
    "create_scene_model",
]

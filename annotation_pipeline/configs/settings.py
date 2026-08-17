"""
settings.py

Centralized configuration system for the KGFLM Traffic Annotation Pipeline.

This module is responsible for loading the pipeline configuration from
default.config and converting it into a strongly typed hierarchy of
Python dataclasses.

Rather than allowing every pipeline component to parse JSON files
independently, configuration is loaded exactly once and shared across all
pipeline stages.

Current pipeline

Configuration
      │
      ▼
Scene Understanding
      │
      ▼
Ontology Reasoning
      │
      ▼
Prompt Builder
      │
      ▼
Grounding
      │
      ▼
Segmentation
      │
      ▼
COCO Annotation

Responsibilities

• Locate the project root.
• Resolve filesystem paths.
• Load pipeline configuration.
• Validate configured model backends.
• Convert JSON into strongly typed configuration objects.
• Create output directories.
• Provide a single immutable configuration object shared throughout the
  pipeline.
"""

# ============================================================================
# Imports
# ============================================================================

from __future__ import annotations
import os

from custom_logger import CustomLogger
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

logger = CustomLogger(__name__)

# ============================================================================
# Default Configuration
# ============================================================================

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "default.config"
)

# ============================================================================
# Paths
# ============================================================================


@dataclass(frozen=True)
class PathsConfig:
    """
    Filesystem paths used throughout the annotation pipeline.
    """

    project_root: Path

    random_frames: Path

    ontology: Path

    ontology_file: Path

    ontology_index: Path

    ontology_metadata: Path

    prompts: Path

    outputs: Path

    scene_cache: Path

    ontology_cache: Path

    pipeline_cache: Path

    visualizations: Path

    masks: Path

    annotations: Path

    logs: Path

    def ensure_output_dirs(self) -> None:
        """
        Create output directories.
        """

        directories = [

            self.outputs,

            self.scene_cache,

            self.pipeline_cache,

            self.visualizations,

            self.masks,

            self.annotations,

            self.logs,

            self.ontology_index.parent,

        ]

        for directory in directories:

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

# ============================================================================
# Models
# ============================================================================
@dataclass(frozen=True)
class ModelsConfig:
    """
    Configuration for model selection.
    """

    scene_understanding_backend: str
    scene_understanding_prompt: str
    grounding_backend: str
    segmentation_backend: str

    gemma_model_id: str
    smolvlm_model_id: str

    embedding_model_id: str
    embedding_device: str
    save_index: bool
    rebuild_index: bool
    top_k: int

    locate_anything_model_id: str

    sam2_model_id: str

# ============================================================================
# Logging
# ============================================================================


@dataclass(frozen=True)
class LoggingConfig:
    """
    Logging configuration.
    """

    level: str = "INFO"

    log_to_file: bool = True

    log_filename: str = "pipeline.log"

    format: str = (
        "%(asctime)s | "
        "%(name)-35s | "
        "%(levelname)-8s | "
        "%(message)s"
    )

    date_format: str = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# Random frame sampler Configuration
# ============================================================================
@dataclass(frozen=True)
class RandomFrameSamplerConfig:
    """
    Configuration for the random frame sampler.
    """

    frames: str

    videos: str

    mapping: str

    interval_seconds: int

    base_url: str

    delete_downloaded_videos: bool

    logger_level: str

    num_images: int

    local_output_root: str

    day_nights: tuple[str, ...]

    vehicles: tuple[str, ...]

    video_ids: tuple[str, ...]

    location_tree: dict[str, Any]

# ============================================================================
# Top-Level Configuration
# ============================================================================

@dataclass(frozen=True)
class PipelineParams:

    download_random_frames: bool

    run_scene_understanding: bool
    run_ontology_reasoning: bool
    run_prompt_builder: bool
    run_grounding: bool
    run_segmentation: bool
    run_annotation: bool

    save_intermediate_cache: bool
    save_pipeline_cache: bool
    save_raw_outputs: bool
    save_visualizations: bool

    num_images: int
    batch_size: int
    interval_seconds: int

    confidence_threshold: float
    max_objects_per_image: int

    benchmark: bool
    supported_image_extensions: list

# ============================================================================
# Pipelie Configuration
# ============================================================================
@dataclass(frozen=True)
class PipelineConfig:
    paths: PathsConfig

    models: ModelsConfig

    pipeline: PipelineParams

    logging: LoggingConfig

    sampler: RandomFrameSamplerConfig

# ============================================================================
# Path Resolution
# ============================================================================


def _find_project_root() -> Path:
    """
    Locate the project root directory.

    The project root is identified by searching parent directories until
    a pyproject.toml file is found.

    Searching begins from the directory containing this module.

    Returns
    -------
    Path
        Absolute path to the project root.

    Raises
    ------
    FileNotFoundError
        If no project root can be located.
    """
    logger.info(
        "Locating project root..."
    )
    current = Path(__file__).resolve().parent

    while True:
        logger.debug(
            "Searching '{}'.",
            current,
        )
        if (current / "pyproject.toml").exists():

            logger.debug(
                "Project root found: {}",
                current,
            )

            return current

        if current.parent == current:

            raise FileNotFoundError(
                "Unable to locate project root "
                "(missing pyproject.toml)."
            )

        current = current.parent


def _resolve_paths(project_root: Path) -> PathsConfig:
    """
    Resolve every configured filesystem path.

    Relative paths from the JSON configuration are converted into absolute
    paths using the project root.

    Returns
    -------
    PathsConfig
        Fully resolved filesystem paths.
    """

    logger.info(
        "Resolving filesystem paths."
    )
    outputs = project_root / "annotation_pipeline" / "outputs"

    ontology_root = project_root/ "annotation_pipeline"/ "models" / "ontology"
    
    paths = PathsConfig(
        project_root=project_root,

        random_frames=project_root / "random_frames",

        ontology=ontology_root,

        ontology_file=ontology_root / "traffic_ontology_v2.json",

        ontology_index=ontology_root/ "traffic_ontology.index",

        ontology_metadata=ontology_root/ "traffic_ontology.pkl",

        prompts=project_root / "annotation_pipeline" / "prompts",

        outputs = outputs,

        logs = outputs / "logs",

        scene_cache = outputs / "scene_cache",

        ontology_cache = outputs / "ontology_cache",

        pipeline_cache = outputs / "pipeline_cache",

        annotations = outputs / "annotations",

        masks = outputs / "masks",

        visualizations = outputs / "visualizations",
    )

    logger.debug(
        "Resolved filesystem paths."
    )

    return paths

# ============================================================================
# Configuration Loading
# ============================================================================


def load_config(
    config_path: Path | str | None = None,
    project_root: Path | str | None = None,
) -> PipelineConfig:
    """
    Load the pipeline configuration from a config file.

    Parameters
    ----------
    config_path
        Optional path to default.config.

    project_root
        Optional override for the project root.

    Returns
    -------
    PipelineConfig
        Fully initialized configuration object.
    """
    logger.info(
        "Loading pipeline configuration."
    )
    config_path = (
        Path(config_path)
        if config_path is not None
        else _DEFAULT_CONFIG_PATH
    )

    if not config_path.is_file():

        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        raw: dict[str, Any] = json.load(file)

    raw_config = raw.get("annotation_pipeline", raw)

    logger.debug(
        "Configuration file successfully parsed."
    )
    # ----------------------------------------------------------------------
    # Project Root
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing project root."
    )

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else _find_project_root()
    )

    # ----------------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing filesystem paths."
    )
    paths = _resolve_paths(root)

    logger.info(
    "Initializing model configuration."
)

    models = ModelsConfig(

        scene_understanding_backend=raw["scene_understanding_backend"],

        scene_understanding_prompt=raw["scene_understanding_prompt"],

        grounding_backend=raw["grounding_backend"],

        segmentation_backend=raw["segmentation_backend"],

        gemma_model_id=raw["scene_understanding"]["gemma_model_id"],

        smolvlm_model_id=raw["scene_understanding"]["smolvlm_model_id"],

        embedding_model_id=raw["ontology_reasoning"]["embedding"]["model_id"],

        embedding_device=raw["ontology_reasoning"]["embedding"]["device"],

        save_index=raw["ontology_reasoning"]["save_index"],

        rebuild_index=raw["ontology_reasoning"]["rebuild_index"],

        top_k=raw["ontology_reasoning"].get("top_k", 10),

        locate_anything_model_id=raw["grounding"]["locate_anything_model_id"],

        sam2_model_id=raw["segmentation"]["sam2_model_id"],
    )

    # ----------------------------------------------------------------------
    # Pipeline Parameters
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing pipeline parameters."
    )
    pipeline = PipelineParams(

        download_random_frames=raw["download_random_frames"],

        run_scene_understanding=raw["run_scene_understanding"],
        run_ontology_reasoning=raw["run_ontology_reasoning"],
        run_prompt_builder=raw["run_prompt_builder"],
        run_grounding=raw["run_grounding"],
        run_segmentation=raw["run_segmentation"],
        run_annotation=raw["run_annotation"],

        save_intermediate_cache=raw["save_intermediate_cache"],
        save_pipeline_cache=raw["save_pipeline_cache"],
        save_raw_outputs=raw["save_raw_outputs"],
        save_visualizations=raw["save_visualizations"],

        num_images=raw["num_images"],
        batch_size=raw["batch_size"],
        interval_seconds=raw["interval_seconds"],

        confidence_threshold=raw["confidence_threshold"],
        max_objects_per_image=raw["max_objects_per_image"],

        benchmark=raw["benchmark"],
        supported_image_extensions=(".png", ".jpg", ".jpeg", ".webp"),
    )

    # ----------------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing logging configuration."
    )
    logging_config = LoggingConfig(
        **raw_config.get("logging", {})
    )

    # ----------------------------------------------------------------------
    # Random Frame Sampler
    # ----------------------------------------------------------------------

    sampler = RandomFrameSamplerConfig(

        frames=raw["frames"],

        videos=raw["videos"],

        mapping=raw["mapping"],

        interval_seconds=raw["interval_seconds"],

        base_url=raw["base_url"],

        delete_downloaded_videos=raw["delete_downloaded_videos"],

        logger_level=raw["logger_level"],

        num_images=raw["num_images"],

        local_output_root=raw["local_output_root"],

        day_nights=tuple(raw["DAY_NIGHTS"]),

        vehicles=tuple(raw["VEHICLES"]),

        video_ids=tuple(raw["VIDEO_IDS"]),

        location_tree=raw["LOCATION_TREE"],
    )

    # ----------------------------------------------------------------------
    # Final Configuration
    # ----------------------------------------------------------------------

    config = PipelineConfig(

        paths=paths,

        models=models,

        pipeline=pipeline,

        logging=logging_config,

        sampler=sampler,
    )

    logger.info(
        "Pipeline configuration loaded successfully from '{}'.",
        config_path,
    )

    logger.info(
        "Project root resolved to '{}'.",
        root,
    )
    logger.info(
        "Configuration initialization complete."
    )
    return config

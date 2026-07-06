"""
settings.py

Centralized configuration system for the KGFLM Traffic Annotation Pipeline.

This module is responsible for loading the pipeline configuration from
pipeline_config.yaml and converting it into a strongly typed hierarchy of
Python dataclasses.

Rather than allowing every pipeline component to parse YAML files
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
• Convert YAML into strongly typed configuration objects.
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

import yaml

logger = CustomLogger(__name__)

# ============================================================================
# Default Configuration
# ============================================================================

_DEFAULT_CONFIG_PATH = Path(
    os.path.join(os.path.dirname(__file__), "pipeline_config.yaml")
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

    prompts: Path

    outputs: Path

    scene_cache: Path

    pipeline_cache: Path

    visualizations: Path

    masks: Path

    annotations: Path

    logs: Path

    def ensure_output_dirs(self) -> None:
        """
        Create every configured output directory.

        The pipeline expects all output directories to exist before execution.
        Missing directories are automatically created.
        """

        directories = [

            self.outputs,

            self.scene_cache,

            self.pipeline_cache,

            self.visualizations,

            self.masks,

            self.annotations,

            self.logs,

        ]

        for directory in directories:

            logger.debug(
                "Ensuring directory exists: '{}'.",
                directory,
            )

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================================
# Shared HuggingFace Model Configuration
# ============================================================================

@dataclass(frozen=True)
class HFModelConfig:
    """
    Generic configuration shared by Hugging Face models.
    """

    model_id: str

    device: str = "cuda"

    dtype: str = "float16"

    attention: str = "sdpa"

    trust_remote_code: bool = False

    compile: bool = False

    max_new_tokens: int = 2048

    temperature: float = 0.1

    quantization: str | None = None

    do_sample: bool = False

    def __post_init__(self) -> None:

        if self.device not in {"cpu", "cuda"}:

            logger.error(
                "Unsupported device '{}'.",
                self.device,
            )

            raise ValueError(
                f"Unsupported device '{self.device}'."
            )

        if self.temperature < 0:

            raise ValueError(
                "Temperature must be non-negative."
            )
# ============================================================================
# Scene Understanding
# ============================================================================


@dataclass(frozen=True)
class SceneUnderstandingConfig:
    """
    Return the active Scene Understanding model configuration.

    The backend specified in pipeline_config.yaml determines which model
    configuration is returned.

    Raises
    ------
    ValueError
        If the configured backend does not exist.
    """

    backend: str

    models: dict[str, HFModelConfig]

    @property
    def active(self) -> HFModelConfig:

        try:
            logger.debug(
                "Using Scene Understanding backend '{}'.",
                self.backend,
            )
            return self.models[self.backend.strip().lower()]

        except KeyError as exc:

            raise ValueError(
                f"Unsupported Scene Understanding backend "
                f"'{self.backend}'. "
                f"Available backends: {sorted(self.models.keys())}"
            ) from exc

# ============================================================================
# Grounding
# ============================================================================


@dataclass(frozen=True)
class LocateAnythingConfig:
    """
    Configuration for a LocateAnything model.
    """

    model_id: str

    device: str = "cuda"

    coordinate_scale: int = 1000


@dataclass(frozen=True)
class GroundingConfig:
    """
    Configuration for grounding models.
    """

    backend: str

    models: dict[str, LocateAnythingConfig]

    @property
    def active(self) -> LocateAnythingConfig:

        try:
            logger.debug(
                "Using Grounding backend '{}'.",
                self.backend,
            )
            return self.models[self.backend.strip().lower()]

        except KeyError as exc:

            raise ValueError(
                f"Unsupported Grounding backend "
                f"'{self.backend}'. "
                f"Available backends: {sorted(self.models.keys())}"
            ) from exc

# ============================================================================
# Segmentation
# ============================================================================


@dataclass(frozen=True)
class SAM2Config:
    """
    Configuration for a SAM2 model.
    """

    model_id: str

    device: str = "cuda"

    checkpoint: str | None = None

    config: str | None = None

    multimask_output: bool = False


@dataclass(frozen=True)
class SegmentationConfig:
    """
    Configuration for segmentation models.
    """

    backend: str

    models: dict[str, SAM2Config]

    @property
    def active(self) -> SAM2Config:

        try:
            logger.debug(
                "Using Segmentation backend '{}'.",
                self.backend,
            )
            return self.models[self.backend.strip().lower()]

        except KeyError as exc:

            raise ValueError(
                f"Unsupported Segmentation backend "
                f"'{self.backend}'. "
                f"Available backends: {sorted(self.models.keys())}"
            ) from exc

# ============================================================================
# Models
# ============================================================================


@dataclass(frozen=True)
class ModelsConfig:
    """
    Container for every model family used by the pipeline.
    """

    scene_understanding: SceneUnderstandingConfig

    grounding: GroundingConfig

    segmentation: SegmentationConfig

# ============================================================================
# Pipeline Parameters
# ============================================================================


@dataclass(frozen=True)
class PipelineParams:
    """
    Runtime parameters controlling pipeline execution.
    """

    confidence_threshold: float = 0.30

    max_objects_per_image: int = 50

    supported_image_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
    )

    save_intermediate_cache: bool = True

    save_pipeline_cache: bool = True

    save_raw_outputs: bool = False

    save_visualizations: bool = True

    benchmark: bool = False


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
# Project
# ============================================================================

@dataclass(frozen=True)
class ProjectConfig:
    """
    Project metadata.
    """

    name: str

    version: str

    description: str = ""

    author: str = ""

# ============================================================================
# Top-Level Configuration
# ============================================================================


@dataclass(frozen=True)
class PipelineConfig:
    """
    Root configuration object.

    Every stage of the annotation pipeline receives this object.
    """

    project: ProjectConfig

    paths: PathsConfig

    models: ModelsConfig

    pipeline: PipelineParams

    logging: LoggingConfig

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


def _resolve_paths(
    raw_paths: dict[str, str],
    project_root: Path,
) -> PathsConfig:
    """
    Resolve every configured filesystem path.

    Relative paths from the YAML configuration are converted into absolute
    paths using the project root.

    Returns
    -------
    PathsConfig
        Fully resolved filesystem paths.
    """

    logger.info(
        "Resolving filesystem paths."
    )

    resolved: dict[str, Path] = {

        "project_root": project_root,

    }

    for key, value in raw_paths.items():

        if key == "project_root":

            continue

        resolved[key] = (

            project_root / value

        ).resolve()

    paths = PathsConfig(**resolved)

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
    Load the pipeline configuration from a YAML file.

    Parameters
    ----------
    config_path
        Optional path to pipeline_config.yaml.

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

        raw: dict[str, Any] = yaml.safe_load(file)
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
    # Project
    # ----------------------------------------------------------------------

    project = ProjectConfig(
        **raw.get("project", {})
    )

    # ----------------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing filesystem paths."
    )
    paths = _resolve_paths(
        raw.get("paths", {}),
        root,
    )

    # ----------------------------------------------------------------------
    # Scene Understanding
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing Scene Understanding configuration."
    )
    scene_raw = raw["models"]["scene_understanding"]

    scene_backend = scene_raw["backend"].strip().lower()

    scene_models = {
        name.lower(): HFModelConfig(**model_cfg)
        for name, model_cfg in scene_raw["models"].items()
    }

    scene_understanding = SceneUnderstandingConfig(
        backend=scene_backend,
        models=scene_models,
    )

    # Validate backend
    _ = scene_understanding.active

    # ----------------------------------------------------------------------
    # Grounding
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing Grounding configuration."
    )
    grounding_raw = raw["models"]["grounding"]

    grounding_backend = grounding_raw["backend"].strip().lower()

    grounding_models = {
        name.lower(): LocateAnythingConfig(**model_cfg)
        for name, model_cfg in grounding_raw["models"].items()
    }

    grounding = GroundingConfig(
        backend=grounding_backend,
        models=grounding_models,
    )

    # Validate backend
    _ = grounding.active

    # ----------------------------------------------------------------------
    # Segmentation
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing Segmentation configuration."
    )
    segmentation_raw = raw["models"]["segmentation"]

    segmentation_backend = segmentation_raw["backend"].strip().lower()

    segmentation_models = {
        name.lower(): SAM2Config(**model_cfg)
        for name, model_cfg in segmentation_raw["models"].items()
    }

    segmentation = SegmentationConfig(
        backend=segmentation_backend,
        models=segmentation_models,
    )

    # Validate backend
    _ = segmentation.active
    # ----------------------------------------------------------------------
    # Models
    # ----------------------------------------------------------------------

    models = ModelsConfig(

        scene_understanding=scene_understanding,

        grounding=grounding,

        segmentation=segmentation,
    )

    # ----------------------------------------------------------------------
    # Pipeline Parameters
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing pipeline parameters."
    )
    pipeline_raw = raw.get(
        "pipeline",
        {},
    )

    if "supported_image_extensions" in pipeline_raw:

        pipeline_raw["supported_image_extensions"] = tuple(
            pipeline_raw["supported_image_extensions"]
        )

    pipeline = PipelineParams(
        **pipeline_raw,
    )

    # ----------------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------------
    logger.info(
        "Initializing logging configuration."
    )
    logging_config = LoggingConfig(
        **raw.get("logging", {})
    )

    # ----------------------------------------------------------------------
    # Final Configuration
    # ----------------------------------------------------------------------

    config = PipelineConfig(

        project=project,

        paths=paths,

        models=models,

        pipeline=pipeline,

        logging=logging_config,
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

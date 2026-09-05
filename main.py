"""Main entry point for the VLM-First annotation pipeline.

Pipeline
--------
Random Frame Preparation
        |
        v
Image Discovery
        |
        v
Scene Understanding
        |
        v
Ontology Reasoning
        |
        v
Locate Anything Grounding
        |          \
        |           -> one retry when final detections exceed the threshold
        v
Annotation
        |
        v
Size-based postprocessing
        |
        v
Semantic Verification (when enabled)
        |
        v
Final Visualization
        |
        v
Pipeline Complete

Semantic verification is an optional pipeline stage controlled by
``run_semantic_verification``. It runs after annotation/postprocessing,
preserves the authoritative Locate Anything bounding boxes, updates only
semantic labels, and writes the complete verified annotation to
``outputs/semantic_verified``. Diagnostic JSON also preserves the complete
PipelineCache snapshot so failures can be traced across stages.

The pipeline also keeps model lifetimes separated through explicit cleanup
boundaries so large VLMs do not unnecessarily remain resident in GPU memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch
import time
import gc
import shutil
import re
from typing import Any

from custom_logger import CustomLogger
from logmod import logs

from annotation_pipeline.cache.pipeline_cache import PipelineCache, PipelineStage
from annotation_pipeline.configs.settings import load_config
from annotation_pipeline.models.vlm.model_loader import (
    create_scene_model,
)
from annotation_pipeline.utils.image_utils import (
    discover_images,
)
from annotation_pipeline.utils.gpu_monitor import GPUMonitor

from annotation_pipeline.pipeline.scene_understanding import (
    SceneUnderstandingEngine,
)

from annotation_pipeline.pipeline.ontology import (
    OntologyEngine,
)

from annotation_pipeline.pipeline.locate_anything import (
    LocateAnythingEngine,
)
from annotation_pipeline.pipeline.annotation import AnnotationEngine
from annotation_pipeline.prompts.load_prompt import load_prompt
# Random frame downloader
from annotation_pipeline.pipeline.random_frame_sampler import get_random_frames_from_common_config
from annotation_pipeline.models.segmentation.sam2_segmenter import (
    SAM2SegmentationEngine,
)
from annotation_pipeline.models.yolopv2 import YOLOPv2SegmentationEngine

logger = CustomLogger(__name__)

# ============================================================================
# Random Frame Preparation
# ============================================================================


def has_images(
    image_dir: Path,
    extensions: tuple[str, ...],
) -> bool:
    """Return whether the configured frame directory has supported images.

    ``discover_images`` is used so the check follows the same extension
    handling as the main image-discovery stage.
    """

    if not image_dir.exists():
        return False

    try:

        image_paths = discover_images(
            image_dir,
            extensions,
        )

        return len(image_paths) > 0

    except ValueError:

        return False


# ============================================================================
# RESOURCE / CACHE HELPERS
# ============================================================================
#
# These helpers are deliberately kept outside the stage implementations.
# They provide shared cleanup, memory diagnostics, semantic-input persistence,
# and grounding-count logic without changing the behavior of the pipeline
# engines themselves.
#
def log_gpu_memory(
    label: str,
) -> None:
    """Log CUDA allocation and reservation at a named pipeline boundary.

    Memory logging is diagnostic only. It never changes model state and
    safely reports CUDA-unavailable environments.
    """

    if not torch.cuda.is_available():
        logger.info(
            "GPU MEMORY [{}] | CUDA unavailable",
            label,
        )
        return

    logger.info(
        "GPU MEMORY [{}] | allocated={:.2f} GB | reserved={:.2f} GB",
        label,
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


def cleanup_stage_resources(
    *objects: Any,
) -> None:
    """Release stage-owned resources and clear Python/CUDA caches.

    Each supplied object is given the opportunity to run its ``unload``
    method. Python garbage collection and CUDA cache cleanup then provide
    a deterministic boundary between memory-heavy pipeline stages.

    This cannot reclaim tensors that are still referenced elsewhere.
    """

    for obj in objects:
        if obj is None:
            continue

        try:
            unload = getattr(
                obj,
                "unload",
                None,
            )

            if callable(unload):
                unload()

        except Exception as exc:
            logger.warning(
                "Stage resource unload failed: {}",
                exc,
            )

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def save_semantic_verification_input(
    config,
    image_path: Path,
    final_detections: list[dict[str, Any]],
) -> Path:
    """Save the final base annotation for semantic verification.

    The saved detections are written only after annotation and size
    postprocessing. Their Locate Anything bounding boxes therefore remain
    the authoritative localization that the separate semantic stage must
    reuse.
    """

    input_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "inputs"
    )

    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        input_dir
        / f"{image_path.stem}.json"
    )

    payload = {
        "image": str(image_path),
        "image_name": image_path.name,
        "detections": final_detections,
    }

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Semantic verification input saved for '{}': {}",
        image_path.name,
        output_path,
    )

    return output_path




def prepare_random_frames(
    config,
) -> None:
    """Populate the random-frame directory only when it is empty.

    Existing frames are deliberately preserved so rerunning the pipeline
    does not redownload the same input set unnecessarily.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("Preparing Random Frames")
    logger.info("=" * 80)

    if has_images(
        config.paths.random_frames,
        config.pipeline.supported_image_extensions,
    ):

        logger.info(
            "Random frame directory already contains images."
        )

        logger.info(
            "Skipping download."
        )

        return

    logger.info(
        "No random frames found."
    )

    logger.info(
        "Downloading random frames..."
    )

    try:

        records, manifest = (
            get_random_frames_from_common_config()
        )

        logger.info(
            "Downloaded {} image(s).",
            len(records),
        )

        if manifest is not None:

            logger.info(
                "Manifest written to '{}'.",
                manifest,
            )

    except Exception as e:

        logger.error(
            "Random frame download failed: {}",
            e,
        )

        raise

def route_defected_annotation(
    annotation_engine: AnnotationEngine,
    image_path: Path,
    visualization_path: Path,
    detections_count: int,
    threshold: int,
) -> Path:
    """Move an over-threshold final visualization to defected_annotations.

    The annotation itself is still produced normally. Only the final
    annotated visualization is routed away from the normal annotations
    directory when the final detection count remains above the configured
    threshold after the single retry.
    """

    defected_dir = (
        annotation_engine.output_dir.parent
        / "defected_annotations"
    )

    defected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    defected_path = (
        defected_dir
        / visualization_path.name
    )

    try:
        shutil.move(
            str(visualization_path),
            str(defected_path),
        )

    except Exception as exc:
        logger.error(
            "Failed to move over-threshold annotation for '{}' "
            "to defected_annotations: {}",
            image_path.name,
            exc,
        )
        # Keep the original path if routing fails.
        return visualization_path

    logger.warning(
        "Final annotation for '{}' has {} detections (> {}). "
        "Moved annotated result to '{}'.",
        image_path.name,
        detections_count,
        threshold,
        defected_path,
    )

    return defected_path


def save_dropped_detections_visualization(
    annotation_engine: AnnotationEngine,
    image_path: Path,
    original_detections: list[dict],
    final_detections: list[dict],
) -> Path | None:
    """Visualize detections removed during postprocessing.

    The dropped visualization is written beside the normal annotations under
    ``outputs/dropped``. Only detections removed by postprocessing are shown.
    The original annotation bbox and label data are preserved unchanged.
    """

    surviving_ids = {
        id(detection)
        for detection in final_detections
        if isinstance(detection, dict)
    }

    dropped_detections = [
        detection
        for detection in original_detections
        if isinstance(detection, dict)
        and id(detection) not in surviving_ids
    ]

    dropped_dir = (
        annotation_engine.output_dir.parent
        / "dropped"
    )

    dropped_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dropped_path = (
        dropped_dir
        / f"{image_path.stem}_dropped.png"
    )

    if not dropped_detections:
        # Avoid leaving a stale dropped image from an earlier run when this
        # run produced no postprocessing removals.
        try:
            if dropped_path.exists():
                dropped_path.unlink()
        except OSError as exc:
            logger.warning(
                "Could not remove stale dropped visualization for '{}': {}",
                image_path.name,
                exc,
            )

        return None

    annotation_engine.visualizer.visualize(
        image_path=image_path,
        detections=dropped_detections,
        output_path=dropped_path,
    )

    logger.info(
        "Dropped annotation visualization for '{}' saved to '{}'. "
        "{} detection(s) were removed during postprocessing.",
        image_path.name,
        dropped_path,
        len(dropped_detections),
    )

    return dropped_path


def run_annotation_stage(
    annotation_engine: AnnotationEngine,
    image_path: Path,
) -> tuple[list[dict], Path]:
    """Run annotation, size filtering, and final visualization.

    The annotation engine first produces the base detections. The
    postprocessing filter then removes only detections that are
    geometrically implausibly tiny or oversized. The visualization is
    regenerated from that filtered list so removed detections do not remain
    visible in the final image.

    The detection dictionaries themselves are not otherwise rewritten.
    """

    original_detections = annotation_engine.annotate_image(
        image_path
    )

    # --------------------------------------------------------------
    # POSTPROCESSING
    # --------------------------------------------------------------

    final_detections = postprocess_detections(
        image_path=image_path,
        detections=original_detections,
    )

    # Save a separate visualization containing only detections removed by
    # postprocessing. The normal annotation visualization remains filtered.
    save_dropped_detections_visualization(
        annotation_engine=annotation_engine,
        image_path=image_path,
        original_detections=original_detections,
        final_detections=final_detections,
    )

    visualization_path = (
        annotation_engine.output_dir
        / f"{image_path.stem}_annotated.png"
    )

    # Re-render using the filtered detections.
    annotation_engine.visualizer.visualize(
        image_path=image_path,
        detections=final_detections,
        output_path=visualization_path,
    )

    return (
        final_detections,
        visualization_path,
    )

def _normalise_detection_label(
    detection: dict,
) -> str:
    """Return the best available semantic label for a detection."""

    for key in (
        "class_name",
        "final_label",
        "object_name",
        "grounding_label",
        "grounding_prompt",
    ):
        value = detection.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return ""


def _is_zebra_crossing_detection(
    detection: dict,
) -> bool:
    """Return whether a detection represents a zebra/crossing marking."""

    label = _normalise_detection_label(detection)

    normalized = re.sub(
        r"[_\-]+",
        " ",
        label,
    )

    return (
        "zebra crossing" in normalized
        or "crosswalk" in normalized
    )


def _is_road_marking_detection(
    detection: dict,
) -> bool:
    """Return whether a detection is a road-marking object.

    Zebra crossings are explicitly excluded. The rule targets named lane
    markings, road arrows, and other explicit road-marking classes.
    """

    label = _normalise_detection_label(detection)

    normalized = re.sub(
        r"[_\-]+",
        " ",
        label,
    )

    if not normalized or _is_zebra_crossing_detection(detection):
        return False

    road_marking_terms = (
        "road marking",
        "lane divider",
        "lane line",
        "solid lane",
        "broken lane",
        "road arrow",
        "turn arrow",
        "direction arrow",
        "stop line",
        "stopline",
        "center line",
        "centre line",
        "cross lane line",
        "traffic marking",
    )

    return any(
        term in normalized
        for term in road_marking_terms
    )


def _bbox_fully_inside(
    inner_bbox: list | tuple,
    outer_bbox: list | tuple,
) -> bool:
    """Return True when inner_bbox is fully contained by outer_bbox."""

    if (
        not isinstance(inner_bbox, (list, tuple))
        or len(inner_bbox) != 4
        or not isinstance(outer_bbox, (list, tuple))
        or len(outer_bbox) != 4
    ):
        return False

    try:
        ix1, iy1, ix2, iy2 = map(float, inner_bbox)
        ox1, oy1, ox2, oy2 = map(float, outer_bbox)
    except (TypeError, ValueError):
        return False

    ix1, ix2 = sorted((ix1, ix2))
    iy1, iy2 = sorted((iy1, iy2))
    ox1, ox2 = sorted((ox1, ox2))
    oy1, oy2 = sorted((oy1, oy2))

    return (
        ix1 >= ox1
        and iy1 >= oy1
        and ix2 <= ox2
        and iy2 <= oy2
    )


def _bbox_containment_ratio(
    inner_bbox: list | tuple,
    outer_bbox: list | tuple,
) -> float:
    """Return the fraction of inner_bbox area covered by outer_bbox."""

    if (
        not isinstance(inner_bbox, (list, tuple))
        or len(inner_bbox) != 4
        or not isinstance(outer_bbox, (list, tuple))
        or len(outer_bbox) != 4
    ):
        return 0.0

    try:
        ix1, iy1, ix2, iy2 = map(float, inner_bbox)
        ox1, oy1, ox2, oy2 = map(float, outer_bbox)
    except (TypeError, ValueError):
        return 0.0

    ix1, ix2 = sorted((ix1, ix2))
    iy1, iy2 = sorted((iy1, iy2))
    ox1, ox2 = sorted((ox1, ox2))
    oy1, oy2 = sorted((oy1, oy2))

    inner_width = max(0.0, ix2 - ix1)
    inner_height = max(0.0, iy2 - iy1)
    inner_area = inner_width * inner_height

    if inner_area <= 0:
        return 0.0

    intersection_width = max(
        0.0,
        min(ix2, ox2) - max(ix1, ox1),
    )
    intersection_height = max(
        0.0,
        min(iy2, oy2) - max(iy1, oy1),
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    return intersection_area / inner_area


def remove_road_markings_inside_zebra_crossing(
    image_path: Path,
    detections: list[dict],
) -> list[dict]:
    """Remove road markings fully contained by zebra-crossing boxes.

    A zebra-crossing detection is authoritative for the crossing area.
    Other road-marking detections are removed when at least 95% of their
    bbox area lies inside a zebra-crossing bbox. This tolerates tiny
    localization differences at the boundary while preserving markings that
    substantially extend outside the crossing.
    """

    zebra_crossings = [
        detection
        for detection in detections
        if (
            isinstance(detection, dict)
            and _is_zebra_crossing_detection(detection)
            and isinstance(detection.get("bbox"), (list, tuple))
            and len(detection.get("bbox")) == 4
        )
    ]

    if not zebra_crossings:
        return detections

    filtered = []
    removed = 0

    for detection in detections:
        if not isinstance(detection, dict):
            filtered.append(detection)
            continue

        if _is_zebra_crossing_detection(detection):
            filtered.append(detection)
            continue

        if not _is_road_marking_detection(detection):
            filtered.append(detection)
            continue

        bbox = detection.get("bbox")

        if any(
            _bbox_containment_ratio(
                inner_bbox=bbox,
                outer_bbox=zebra.get("bbox"),
            ) >= 0.95
            for zebra in zebra_crossings
        ):
            removed += 1

            label = _normalise_detection_label(detection)

            logger.warning(
                "Postprocessing removed road marking '{}' from '{}' "
                "because its bbox is fully inside a zebra-crossing bbox.",
                label or "unknown",
                image_path.name,
            )
            continue

        filtered.append(detection)

    if removed:
        logger.info(
            "Zebra-crossing cleanup for '{}': removed {} "
            "road-marking detection(s) contained inside crossing box(es).",
            image_path.name,
            removed,
        )

    return filtered


def _detection_superclass(
    detection: dict,
) -> str:
    """Return the superclass/category used for containment comparison.

    Prefer an explicit superclass field when present. If the annotation
    schema does not provide one, fall back to the canonical class name so
    containment suppression remains conservative.
    """

    for key in (
        "superclass",
        "super_class",
        "parent_class",
        "category",
    ):
        value = detection.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    for key in (
        "class_name",
        "final_label",
    ):
        value = detection.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return ""


def _detection_class(
    detection: dict,
) -> str:
    """Return the most specific class label available."""

    for key in (
        "class_name",
        "final_label",
    ):
        value = detection.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    return _normalise_detection_label(detection)


def remove_redundant_contained_detections(
    image_path: Path,
    detections: list[dict],
) -> list[dict]:
    """Remove redundant detections fully contained by a larger same-class box.

    Two detections are candidates for suppression only when they belong to
    the same superclass and the same semantic class. If one bbox is fully
    contained inside the other, only the larger-area detection is retained.

    This is intentionally stricter than arbitrary overlap suppression:
    partial overlaps are preserved, and detections from different classes
    are never removed by this rule.
    """

    candidates = []

    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            continue

        bbox = detection.get("bbox")

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue

        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            continue

        width = abs(x2 - x1)
        height = abs(y2 - y1)
        area = width * height

        superclass = _detection_superclass(detection)
        class_name = _detection_class(detection)

        if not superclass or not class_name or area <= 0:
            continue

        candidates.append(
            (
                index,
                detection,
                area,
                superclass,
                class_name,
            )
        )

    # Largest boxes are considered first. This guarantees that when several
    # nested detections exist, the outermost/largest valid detection survives.
    candidates.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    remove_indices = set()

    for outer_index, outer_detection, outer_area, outer_superclass, outer_class in candidates:
        if outer_index in remove_indices:
            continue

        outer_bbox = outer_detection.get("bbox")

        for inner_index, inner_detection, inner_area, inner_superclass, inner_class in candidates:
            if inner_index == outer_index or inner_index in remove_indices:
                continue

            if outer_superclass != inner_superclass:
                continue

            if outer_class != inner_class:
                continue

            # The outer candidate must actually be the larger box.
            if outer_area <= inner_area:
                continue

            if _bbox_fully_inside(
                inner_bbox=inner_detection.get("bbox"),
                outer_bbox=outer_bbox,
            ):
                remove_indices.add(inner_index)

                logger.warning(
                    "Postprocessing removed redundant '{}' detection "
                    "from '{}': bbox is fully contained inside the larger "
                    "same-class '{}' detection.",
                    inner_class,
                    image_path.name,
                    outer_class,
                )

    if not remove_indices:
        return detections

    filtered = [
        detection
        for index, detection in enumerate(detections)
        if index not in remove_indices
    ]

    logger.info(
        "Redundant containment cleanup for '{}': removed {} "
        "contained same-class detection(s).",
        image_path.name,
        len(remove_indices),
    )

    return filtered


def _detection_group(
    detection: dict,
) -> str:
    """Return the ontology superclass/category for a detection."""

    for key in (
        "object_group",
        "group",
        "superclass",
        "super_class",
        "parent_class",
    ):
        value = detection.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    # Some cache/annotation representations retain the ontology prediction
    # as a nested object. Use it when available instead of guessing from the
    # class-name text.
    for ontology_key in (
        "ontology_reasoning",
        "ontology",
    ):
        ontology = detection.get(ontology_key)

        if not isinstance(ontology, dict):
            continue

        prediction = ontology.get("prediction")

        if not isinstance(prediction, dict):
            continue

        for key in (
            "superclass",
            "super_class",
            "parent_class",
            "object_group",
            "group",
        ):
            value = prediction.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    return ""


def _is_vehicle_or_road_user(
    detection: dict,
) -> bool:
    """Return whether the ontology superclass marks this as a vehicle/user.

    The ontology superclass is the source of truth. Do not infer the
    superclass from class-name text because words such as "pedestrian" can
    occur inside unrelated classes such as "pedestrian_crossing_sign".
    """

    group = _detection_group(detection)

    return group in {
        "vehicle",
        "road_user",
        "road users",
        "road_user_group",
    }


def _bbox_fully_in_top_third(
    bbox: list | tuple,
    image_height: int,
) -> bool:
    """Return True when the complete bbox lies in the image's top third."""

    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or image_height <= 0
    ):
        return False

    try:
        _, y1, _, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return False

    y1, y2 = sorted((y1, y2))

    return y2 <= (image_height / 3.0)


def _bbox_overlaps_top_third(
    bbox: list | tuple,
    image_height: int,
) -> bool:
    """Return True when any part of bbox lies in the image's top third."""

    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or image_height <= 0
    ):
        return False

    try:
        _, y1, _, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return False

    y1, y2 = sorted((y1, y2))

    top_third_boundary = image_height / 3.0

    return y1 < top_third_boundary and y2 > 0


def _is_roadside_detection(
    detection: dict,
) -> bool:
    """Return whether detection is a road-surface/roadside object.

    These are objects that should not remain in the top third when their
    bbox enters that region. Traffic signs, traffic signals, and streetlights
    are deliberately not included because they are valid scene objects that
    can naturally occupy the upper part of an image.
    """

    group = _detection_group(detection)

    if group in {
        "road",
        "road_surface",
        "road surface",
        "road_infrastructure",
        "road infrastructure",
        "roadside",
        "road_side",
        "road side",
        "barrier",
        "guardrail",
        "guard rail",
    }:
        return True

    label = _detection_class(detection)

    normalized = re.sub(
        r"[_\-]+",
        " ",
        label,
    )

    normalized = " ".join(
        normalized.split()
    )

    roadside_terms = (
        "guardrail",
        "guard rail",
        "road barrier",
        "roadside barrier",
        "crash barrier",
        "safety barrier",
        "jersey barrier",
        "road divider",
        "median",
        "road surface",
        "road",
        "curb",
        "kerb",
        "pavement",
        "sidewalk",
        "traffic island",
        "road island",
        "rumble strip",
        "speed bump",
        "speed hump",
    )

    return any(
        term == normalized
        or normalized.startswith(term + " ")
        for term in roadside_terms
    )


def remove_top_third_detections(
    image_path: Path,
    detections: list[dict],
) -> list[dict]:
    """Remove inappropriate road-related detections from the top third.

    Vehicles and road users are removed only when their COMPLETE bbox lies
    in the top third. If a vehicle or road user crosses the top-third
    boundary, it is retained because the object is only partially present
    there.

    Road markings and roadside/road-surface objects are removed whenever
    their bbox enters the top third.

    Traffic signs, traffic signals, streetlights, buildings, and other
    non-road objects are left untouched. Vehicle/road-user classification
    comes from the ontology superclass, not from class-name keywords.
    """

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image_height = image.height

    except Exception as exc:
        logger.warning(
            "Could not determine image height for top-third cleanup "
            "of '{}': {}",
            image_path.name,
            exc,
        )
        return detections

    top_third_boundary = image_height / 3.0
    filtered = []
    removed = 0

    for detection in detections:
        if not isinstance(detection, dict):
            filtered.append(detection)
            continue

        bbox = detection.get("bbox")

        is_road_marking = _is_road_marking_detection(
            detection
        )

        is_roadside = _is_roadside_detection(
            detection
        )

        is_vehicle_or_user = _is_vehicle_or_road_user(
            detection
        )

        # --------------------------------------------------------------
        # Road markings / roadside objects:
        # remove them if any part of their bbox enters the top third.
        # --------------------------------------------------------------
        if (
            is_road_marking
            or is_roadside
        ):
            if not _bbox_overlaps_top_third(
                bbox=bbox,
                image_height=image_height,
            ):
                filtered.append(detection)
                continue

            label = _normalise_detection_label(detection)

            logger.warning(
                "Top-third cleanup removed '{}' from '{}': "
                "road-related bbox enters the top third above y={:.1f}.",
                label or "unknown",
                image_path.name,
                top_third_boundary,
            )

            removed += 1
            continue

        # --------------------------------------------------------------
        # Vehicles / road users:
        # remove ONLY when the complete bbox is inside the top third.
        # Partial objects crossing below the boundary are preserved.
        # --------------------------------------------------------------
        if is_vehicle_or_user:
            if not _bbox_fully_in_top_third(
                bbox=bbox,
                image_height=image_height,
            ):
                filtered.append(detection)
                continue

            label = _normalise_detection_label(detection)

            logger.warning(
                "Top-third cleanup removed '{}' from '{}': "
                "full vehicle/road-user bbox lies above y={:.1f}.",
                label or "unknown",
                image_path.name,
                top_third_boundary,
            )

            removed += 1
            continue

        # Signs, signals, streetlights, buildings, etc. are untouched.
        filtered.append(detection)

    if removed:
        logger.info(
            "Top-third cleanup for '{}': removed {} "
            "road-related/vehicle/road-user detection(s).",
            image_path.name,
            removed,
        )

    return filtered


def remove_extreme_aspect_ratio_detections(
    image_path: Path,
    detections: list[dict],
    max_aspect_ratio: float = 10.0,
    min_aspect_ratio: float = 0.10,
) -> list[dict]:
    """Remove detections with an implausibly extreme bbox aspect ratio.

    A bbox is removed when its width is more than ``max_aspect_ratio`` times
    its height, or when its height is more than ``1 / min_aspect_ratio`` times
    its width. This catches extremely thin horizontal/vertical false
    detections while leaving ordinary elongated objects untouched.

    Bboxes with invalid/non-positive dimensions are left unchanged here so
    this filter does not silently take ownership of malformed-bbox handling.
    """

    filtered = []
    removed = 0

    for detection in detections:
        if not isinstance(detection, dict):
            filtered.append(detection)
            continue

        bbox = detection.get("bbox")

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            filtered.append(detection)
            continue

        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            filtered.append(detection)
            continue

        bbox_width = abs(x2 - x1)
        bbox_height = abs(y2 - y1)

        if bbox_width <= 0 or bbox_height <= 0:
            filtered.append(detection)
            continue

        aspect_ratio = bbox_width / bbox_height

        # Zebra crossings are intentionally very wide horizontal road
        # markings. Do not remove a valid zebra-crossing bbox merely because
        # its width/height ratio is extreme.
        if _is_zebra_crossing_detection(detection):
            filtered.append(detection)
            continue

        too_wide = aspect_ratio > max_aspect_ratio
        too_tall = aspect_ratio < min_aspect_ratio

        if not (too_wide or too_tall):
            filtered.append(detection)
            continue

        label = _normalise_detection_label(detection)

        logger.warning(
            "Postprocessing removed extreme-aspect-ratio detection "
            "'{}' from '{}': width/height ratio={:.2f}.",
            label or "unknown",
            image_path.name,
            aspect_ratio,
        )

        removed += 1

    if removed:
        logger.info(
            "Aspect-ratio cleanup for '{}': removed {} "
            "extreme-aspect-ratio detection(s).",
            image_path.name,
            removed,
        )

    return filtered


def limit_detections_per_superclass(
    image_path: Path,
    detections: list[dict],
    max_per_superclass: int = 10,
) -> list[dict]:
    """Keep at most ``max_per_superclass`` detections per superclass.

    Vehicle and road-user detections are explicitly exempt because multiple
    road users/vehicles can legitimately occur in the same frame.

    When a superclass exceeds the limit, the largest bboxes are retained.
    The surviving detections are returned in their original order.
    """

    if max_per_superclass <= 0:
        return detections

    grouped: dict[str, list[tuple[int, dict, float]]] = {}

    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            continue

        if _is_vehicle_or_road_user(detection):
            continue

        superclass = _detection_group(detection)

        if not superclass:
            superclass = _detection_superclass(detection)

        if not superclass:
            continue

        bbox = detection.get("bbox")

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            continue

        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            continue

        area = abs(x2 - x1) * abs(y2 - y1)

        if area <= 0:
            continue

        grouped.setdefault(
            superclass,
            [],
        ).append(
            (
                index,
                detection,
                area,
            )
        )

    remove_indices: set[int] = set()

    for superclass, candidates in grouped.items():
        if len(candidates) <= max_per_superclass:
            continue

        # Keep the largest detections. Stable sorting preserves original
        # order when areas are equal.
        ranked = sorted(
            candidates,
            key=lambda item: item[2],
            reverse=True,
        )

        keep_indices = {
            item[0]
            for item in ranked[:max_per_superclass]
        }

        for index, _detection, _area in candidates:
            if index in keep_indices:
                continue

            remove_indices.add(index)

            label = _normalise_detection_label(
                detections[index]
            )

            logger.warning(
                "Postprocessing removed '{}' from '{}': "
                "superclass '{}' exceeded the maximum of {} detections.",
                label or "unknown",
                image_path.name,
                superclass,
                max_per_superclass,
            )

    if not remove_indices:
        return detections

    filtered = [
        detection
        for index, detection in enumerate(detections)
        if index not in remove_indices
    ]

    logger.info(
        "Superclass-count cleanup for '{}': removed {} "
        "detection(s) above the {}-per-superclass limit.",
        image_path.name,
        len(remove_indices),
        max_per_superclass,
    )

    return filtered


def postprocess_detections(
    image_path: Path,
    detections: list[dict],
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.30,
    max_width_ratio: float = 0.90,
    max_height_ratio: float = 0.50,
    tiny_width_ratio: float = 0.02,
    tiny_height_ratio: float = 0.06,
    max_aspect_ratio: float = 10.0,
    min_aspect_ratio: float = 0.10,
    max_per_superclass: int = 10,
) -> list[dict]:
    """Remove only geometrically implausible detections.

    Small-object filtering is intentionally conservative: a detection is
    removed as tiny only when its area is below ``min_area_ratio`` and both
    dimensions are also below their tiny-object limits. This protects
    legitimate thin or distant objects.

    Oversized detections are removed when their area exceeds
    ``max_area_ratio``, or when both their width and height exceed the
    configured image-relative limits. Extreme bbox aspect ratios and
    superclass-count excesses are also removed by dedicated postprocessing
    stages. Vehicle and road-user detections are exempt from the superclass
    count limit.

    Surviving detection dictionaries are returned unchanged.
    """

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image_width, image_height = image.size

    except Exception as exc:

        logger.warning(
            "Could not determine image dimensions for '{}': {}. "
            "Skipping bbox-size postprocessing.",
            image_path.name,
            exc,
        )

        return detections

    if image_width <= 0 or image_height <= 0:
        return detections

    image_area = image_width * image_height

    filtered_detections = []
    removed_small = 0
    removed_large = 0

    for detection in detections:

        if not isinstance(detection, dict):
            filtered_detections.append(detection)
            continue

        bbox = detection.get("bbox")

        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
        ):
            filtered_detections.append(detection)
            continue

        try:

            x1, y1, x2, y2 = map(
                float,
                bbox,
            )

        except (TypeError, ValueError):

            filtered_detections.append(detection)
            continue

        bbox_width = max(
            0.0,
            x2 - x1,
        )

        bbox_height = max(
            0.0,
            y2 - y1,
        )

        bbox_area = (
            bbox_width
            * bbox_height
        )

        area_ratio = (
            bbox_area
            / image_area
        )

        width_ratio = (
            bbox_width
            / image_width
        )

        height_ratio = (
            bbox_height
            / image_height
        )

        # ==============================================================
        # TOO SMALL
        # ==============================================================

        too_small = (
            area_ratio < min_area_ratio
            and width_ratio < tiny_width_ratio
            and height_ratio < tiny_height_ratio
        )

        # ==============================================================
        # TOO LARGE
        # ==============================================================

        # Class-aware oversized handling
        # Default conservative limit remains, but certain classes
        # (vehicles) may occupy a larger fraction of the image.
        LARGE_OBJECT_AREA_LIMITS = {
            "default": max_area_ratio,
            "vehicle": 0.50,
            "bus": 0.50,
            "city_bus": 0.50,
            "truck": 0.50,
            "van": 0.45,
            "car": 0.45,
            # infrastructure-like classes remain strict
            "bridge": 0.20,
            "building": 0.20,
            "road": 0.20,
        }

        # Determine detection class (prefer ontology class_name then final_label)
        det_class = None
        if isinstance(detection.get("class_name"), str) and detection.get("class_name").strip():
            det_class = detection.get("class_name").strip().lower()
        elif isinstance(detection.get("final_label"), str) and detection.get("final_label").strip():
            det_class = detection.get("final_label").strip().lower()
        else:
            det_class = None

        # Resolve threshold
        chosen_max_area = LARGE_OBJECT_AREA_LIMITS.get("default", max_area_ratio)
        if det_class:
            # exact
            if det_class in LARGE_OBJECT_AREA_LIMITS:
                chosen_max_area = LARGE_OBJECT_AREA_LIMITS[det_class]
            else:
                # token match (e.g., city_bus -> bus)
                for key, val in LARGE_OBJECT_AREA_LIMITS.items():
                    if key == "default":
                        continue
                    if key in det_class:
                        chosen_max_area = val
                        break

        too_large_by_area = (area_ratio > chosen_max_area)

        too_large_by_dimensions = (
            width_ratio > max_width_ratio
            and height_ratio > max_height_ratio
        )

        too_large = (
            too_large_by_area
            or too_large_by_dimensions
        )

        # ==============================================================
        # REMOVE
        # ==============================================================

        if too_small:

            removed_small += 1

            class_name = detection.get(
                "class_name",
                detection.get(
                    "final_label",
                    "unknown",
                ),
            )

            logger.warning(
                "Postprocessing removed tiny detection "
                "'{}' from '{}': "
                "area={:.4f}% width={:.2f}% height={:.2f}%",
                class_name,
                image_path.name,
                area_ratio * 100,
                width_ratio * 100,
                height_ratio * 100,
            )

            continue

        if too_large:

            removed_large += 1

            class_name = detection.get(
                "class_name",
                detection.get(
                    "final_label",
                    "unknown",
                ),
            )

            logger.warning(
                "Postprocessing removed oversized detection "
                "'{}' from '{}': "
                "area={:.2f}% width={:.1f}% height={:.1f}%",
                class_name,
                image_path.name,
                area_ratio * 100,
                width_ratio * 100,
                height_ratio * 100,
            )

            continue

        filtered_detections.append(
            detection
        )

    # --------------------------------------------------------------
    # ZEBRA-CROSSING ROAD-MARKING CLEANUP
    # --------------------------------------------------------------
    filtered_detections = remove_road_markings_inside_zebra_crossing(
        image_path=image_path,
        detections=filtered_detections,
    )

    # Remove road markings and fully-contained vehicles/road users from
    # the top third of the image. Partial objects are preserved.
    filtered_detections = remove_top_third_detections(
        image_path=image_path,
        detections=filtered_detections,
    )

    # Remove implausibly thin/wide or tall/narrow bounding boxes.
    filtered_detections = remove_extreme_aspect_ratio_detections(
        image_path=image_path,
        detections=filtered_detections,
        max_aspect_ratio=max_aspect_ratio,
        min_aspect_ratio=min_aspect_ratio,
    )

    # Generic redundancy rule: for the same superclass + same class,
    # retain the larger bbox when the smaller bbox is fully contained.
    filtered_detections = remove_redundant_contained_detections(
        image_path=image_path,
        detections=filtered_detections,
    )

    # Keep at most 10 detections for each superclass. Vehicles and road
    # users are intentionally exempt because many can legitimately appear.
    filtered_detections = limit_detections_per_superclass(
        image_path=image_path,
        detections=filtered_detections,
        max_per_superclass=max_per_superclass,
    )

    logger.info(
        "Postprocessing '{}': {} → {} detection(s). "
        "Removed {} tiny, {} oversized.",
        image_path.name,
        len(detections),
        len(filtered_detections),
        removed_small,
        removed_large,
    )

    return filtered_detections

def _scene_object_class_name(
    scene_object: dict,
) -> str:
    """Return the ontology class name for one scene object."""

    value = scene_object.get("class_name")

    if isinstance(value, str) and value.strip():
        return value.strip().lower()

    ontology = scene_object.get(
        "ontology_reasoning",
        {},
    )

    if isinstance(ontology, dict):
        prediction = ontology.get(
            "prediction",
            {},
        )

        if isinstance(prediction, dict):
            value = prediction.get("class_name")

            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    return ""


def _grounding_entries_for_object(
    image_cache: dict,
    object_id: int,
) -> list[dict]:
    """Return normalized grounding entries for one cached object."""

    entry = image_cache.get(object_id)

    if not isinstance(entry, dict):
        entry = image_cache.get(str(object_id))

    if not isinstance(entry, dict):
        return []

    grounding = entry.get(
        PipelineStage.GROUNDING.value,
        [],
    )

    if isinstance(grounding, list):
        return [
            item
            for item in grounding
            if isinstance(item, dict)
        ]

    if isinstance(grounding, dict):
        return [grounding]

    return []


def get_missing_grounding_classes(
    grounding_engine: LocateAnythingEngine,
    cache: PipelineCache,
    image_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Find classes with no successful Locate Anything grounding.

    A class is considered processed when at least one scene object of that
    class already has at least one grounding bbox.
    """

    image_name = image_path.name

    scene_objects = grounding_engine._load_scene_objects_for_grounding(
        image_path=image_path,
        cache=cache,
    )

    if not scene_objects:
        logger.warning(
            "No scene objects available for grounding coverage check "
            "for '{}'.",
            image_name,
        )
        return [], []

    image_cache = getattr(cache, "_cache", {}).get(
        image_name,
        {},
    )

    if not isinstance(image_cache, dict):
        return scene_objects, scene_objects

    processed_classes: set[str] = set()

    all_classes: set[str] = set()

    for scene_object in scene_objects:
        class_name = _scene_object_class_name(
            scene_object,
        )

        if not class_name:
            continue

        all_classes.add(class_name)

        try:
            object_id = int(
                scene_object.get("object_id"),
            )
        except (TypeError, ValueError):
            continue

        if _grounding_entries_for_object(
            image_cache=image_cache,
            object_id=object_id,
        ):
            processed_classes.add(class_name)

    missing_by_class: dict[str, dict] = {}

    for scene_object in scene_objects:
        class_name = _scene_object_class_name(
            scene_object,
        )

        if (
            class_name
            and class_name not in processed_classes
        ):
            missing_by_class.setdefault(
                class_name,
                scene_object,
            )

    missing_objects = list(
        missing_by_class.values()
    )

    logger.info(
        "Grounding coverage for '{}': {}/{} classes processed; "
        "{} class(es) missing.",
        image_name,
        len(processed_classes),
        len(all_classes),
        len(missing_objects),
    )

    if missing_objects:
        logger.warning(
            "Missing grounding classes for '{}': {}",
            image_name,
            [
                _scene_object_class_name(obj)
                for obj in missing_objects
            ],
        )

    return scene_objects, missing_objects


def rerun_locate_anything_for_missing_classes(
    grounding_engine: LocateAnythingEngine,
    cache: PipelineCache,
    image_path: Path,
) -> tuple[int, list[dict]]:
    """Run the single targeted LA retry and merge it with existing results.

    Existing grounding is deliberately NOT cleared. Only classes that had no
    grounding result are put into the retry prompt. New boxes are appended to
    the existing class/object grounding results, so annotation sees the
    complete result from both passes.
    """

    scene_objects, missing_objects = get_missing_grounding_classes(
        grounding_engine=grounding_engine,
        cache=cache,
        image_path=image_path,
    )

    if not missing_objects:
        return 0, []

    prompt = grounding_engine.prompt_builder.build_prompt(
        scene_objects=missing_objects,
    )

    logger.warning(
        "Targeted Locate Anything retry for '{}'. Prompt contains only: {}",
        image_path.name,
        [
            _scene_object_class_name(obj)
            for obj in missing_objects
        ],
    )
    logger.info(
        "Targeted retry prompt for '{}': {} ",
        image_path.name,
        prompt,
    )

    # Keep the retry self-contained so it cannot fail because of a missing
    # module-level PIL import in a deployed/copy-pasted main.py.
    from PIL import Image

    image = Image.open(
        image_path,
    ).convert("RGB")

    try:
        raw_outputs = grounding_engine.model.generate_batch(
            images=[image],
            prompts=[prompt],
        )

        if not raw_outputs:
            return 0, missing_objects

        raw_output = raw_outputs[0]

        # Parse against ALL scene objects so the existing parser retains the
        # authoritative object IDs and normal matching hierarchy.
        parsed_detections = grounding_engine.parser.parse(
            raw_output=raw_output,
            scene_objects=scene_objects,
            image_size=image.size,
        )

        missing_object_ids = {
            int(obj["object_id"])
            for obj in missing_objects
            if obj.get("object_id") is not None
        }

        recovered_detections = []

        for detection in parsed_detections:
            try:
                object_id = int(
                    detection.get("object_id"),
                )
            except (TypeError, ValueError):
                continue

            # The targeted prompt is only supposed to recover missing classes.
            # Ignore anything the model unexpectedly returns for a class that
            # already had grounding in the first pass.
            if object_id not in missing_object_ids:
                continue

            recovered_detections.append(
                detection,
            )

        image_cache = getattr(cache, "_cache", {}).get(
            image_path.name,
        )

        if not isinstance(image_cache, dict):
            raise RuntimeError(
                f"No cache entry exists for '{image_path.name}' "
                "while merging targeted grounding retry."
            )

        merged_count = 0

        for detection in recovered_detections:
            object_id = int(
                detection["object_id"],
            )

            entry = image_cache.get(object_id)

            if not isinstance(entry, dict):
                entry = image_cache.get(
                    str(object_id),
                )

            if not isinstance(entry, dict):
                continue

            grounding_key = PipelineStage.GROUNDING.value
            existing = entry.get(
                grounding_key,
                [],
            )

            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []

            bbox = detection.get("bbox")

            duplicate = any(
                isinstance(item, dict)
                and item.get("bbox") == bbox
                for item in existing
            )

            if duplicate:
                continue

            existing.append(
                {
                    key: value
                    for key, value in detection.items()
                    if key != "object_id"
                }
            )

            entry[grounding_key] = existing
            merged_count += 1

        # Keep genuinely unmatched retry output for diagnostics, but do not
        # allow it to replace or erase the successful first-pass results.
        unmatched_retry = [
            detection
            for detection in parsed_detections
            if detection.get("object_id") is None
        ]

        if unmatched_retry:
            image_cache.setdefault(
                "_unmatched_grounding",
                [],
            ).extend(unmatched_retry)

        if grounding_engine.config.pipeline.save_intermediate_cache:
            cache.save_image_cache(
                image_name=image_path.name,
                directory=grounding_engine.config.paths.pipeline_cache,
                stage="grounding",
            )

        retry_dir = (
            grounding_engine.config.paths.outputs
            / "locate_anything"
            / "retry"
        )
        retry_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            retry_dir
            / f"{image_path.stem}_missing_classes_prompt.txt"
        ).write_text(
            prompt,
            encoding="utf-8",
        )

        (
            retry_dir
            / f"{image_path.stem}_missing_classes_raw.txt"
        ).write_text(
            str(raw_output),
            encoding="utf-8",
        )

        (
            retry_dir
            / f"{image_path.stem}_missing_classes_parsed.json"
        ).write_text(
            json.dumps(
                recovered_detections,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Targeted Locate Anything retry for '{}' merged {} "
            "new bbox(es) while preserving the first-pass results.",
            image_path.name,
            merged_count,
        )

        return merged_count, missing_objects

    finally:
        try:
            image.close()
        except Exception:
            pass


def clear_grounding_for_retry(
    cache: PipelineCache,
    image_name: str,
) -> None:
    """Clear one image's grounding state before its single retry.

    Locate Anything appends grounding results to per-object lists and can
    also populate unmatched-detection buckets. Clearing those structures
    first makes the retry a replacement pass rather than an accumulation
    pass.

    Only in-memory cache state is reset here. The grounding engine remains
    responsible for refreshing its configured on-disk intermediate cache.
    """

    image_cache = getattr(cache, "_cache", {}).get(image_name)

    if not isinstance(image_cache, dict):
        logger.warning(
            "clear_grounding_for_retry: no cache entry found for '{}'.",
            image_name,
        )
        return



    cleared_objects = 0

    for key, entry in image_cache.items():

        if key in ("grounding_detections", "_unmatched_grounding"):
            continue

        if not isinstance(entry, dict):
            continue

        if PipelineStage.GROUNDING.value in entry:
            entry[PipelineStage.GROUNDING.value] = []
            cleared_objects += 1

    # Top-level orphan buckets (see PipelineCache.add_unmatched_grounding_result
    # and the legacy "grounding_detections" list written by
    # LocateAnythingEngine._update_pipeline_cache).
    image_cache["grounding_detections"] = []
    image_cache["_unmatched_grounding"] = []

    logger.info(
        "Cleared grounding state for '{}' before retry "
        "({} scene object(s) reset).",
        image_name,
        cleared_objects,
    )



# ============================================================================
# SEMANTIC VERIFICATION
# ============================================================================
#
# Semantic verification is a real pipeline stage, not a second program.
# It runs after annotation/postprocessing and works directly on the final
# in-memory detections. The original Locate Anything bbox is immutable.
#

UNREADABLE_CLASS = "unreadable_traffic_sign"
NONE_LABELS = {"", "none", "null", "unknown"}

SEMANTIC_RECOVERY_PROMPT = 'You are an expert traffic scene understanding model performing SEMANTIC RECOVERY for ONE ALREADY-LOCALIZED candidate object.\n\nThe candidate has already been localized by Locate Anything. Its bounding box is authoritative.\nDO NOT perform object detection.\nDO NOT change, refine, expand, shrink, or replace the bounding box.\nYour only task is to determine what traffic-related object, if any, is actually present inside the provided candidate region.\n\nYou receive:\n1. the ORIGINAL FULL IMAGE, for scene context and visual confirmation;\n2. the CROPPED REGION corresponding to the existing bounding box.\n\nInspect both images before deciding.\n\n==================================================\nPRIMARY TASK\n==================================================\n\nDetermine whether the localized candidate is a real, detectable traffic-related object.\n\nIf it is a real traffic-related object and its identity can be determined from DIRECT VISUAL EVIDENCE:\n- return the most specific reliable traffic-object identity;\n- assign exactly one broad object_group from the allowed list;\n- provide a concise visual description.\n\nIf the candidate is present but its exact semantic identity cannot be determined:\n- use the most specific GENERIC traffic-object description that is actually supported by the image;\n- do NOT guess a fine-grained class from context.\n\nIf the localized region does NOT contain a detectable traffic-related object, or the visible content is not a valid object that should be represented by this traffic annotation pipeline:\n- set "discard": true;\n- set "observed_object": "";\n- do NOT invent a replacement class.\n\nA candidate MUST be discarded when the evidence shows that it is:\n- background, texture, shadow, glare, reflection, compression artifact, or noise;\n- an ordinary non-traffic object;\n- road, road surface, asphalt, pavement, sidewalk, building, vegetation, sky,\n  terrain, or another excluded scene element;\n- an accidental/invalid localization that does not correspond to a distinct\n  traffic-related object;\n- too visually ambiguous to establish that a traffic-related object is actually present.\n\nDo NOT keep a false positive merely because a bounding box exists.\n\n==================================================\nTRAFFIC OBJECT SEARCH\n==================================================\n\nInspect the candidate region and the corresponding area in the full image for:\n\nTRAFFIC CONTROL\n- traffic signs and supplementary plates\n- regulatory, warning, guide, information, parking, mandatory,\n  prohibition, priority and temporary signs\n- overhead guide signs\n- traffic, pedestrian, bicycle and bus signals\n- variable message signs\n\nROAD MARKINGS\n- lane boundaries\n- dashed and solid lane dividers\n- centre and edge lines\n- stop lines\n- crosswalks / zebra crossings\n- straight, turn and merge arrows\n- bicycle and bus symbols\n- painted road text and speed markings\n- chevrons and hatch markings\n\nTRAFFIC INFRASTRUCTURE\n- guardrails, barriers and medians\n- bollards and delineator posts\n- traffic cones and temporary barriers\n- street lights\n- traffic cameras\n- gantries\n- bridges and tunnels\n\nTRAFFIC PARTICIPANTS\n- passenger cars, taxis, vans, buses and trucks\n- motorcycles and bicycles\n- pedestrians, cyclists and riders\n- animals\n\nDo not return ordinary utility poles or other non-traffic structures unless\nthe visible object itself is a traffic-related object.\n\n==================================================\nTRAFFIC SIGN INSPECTION\n==================================================\n\nIf the candidate is a traffic sign, inspect the visible board itself.\n\nConsider:\n- circular, triangular, rectangular and octagonal geometry\n- coloured roadside boards and reflective panels\n- border and background\n- visible pictograms and arrows\n- readable text\n- supplementary plates\n- mounting arrangement\n\nA sign is still a valid traffic object when its text or pictogram cannot be\nidentified.\n\nUse "unreadable" for text ONLY when visible text exists but cannot be\ndeciphered.\n\nUse "unreadable" for symbol ONLY when a visible pictogram exists but cannot\nbe identified.\n\nUse "none" when no text or symbol is visibly present.\n\nNever invent text, symbols, languages, countries, traffic rules, or unseen\nproperties.\n\nUnreadable text does NOT automatically mean that the whole sign is\nunreadable.\n\n==================================================\nTRAFFIC SIGNAL INSPECTION\n==================================================\n\nIf the candidate is a traffic signal:\n- inspect the individual signal head;\n- determine the ACTIVE signal state whenever an illuminated lens is visible;\n- preserve the directly visible active colour and lens position.\n\nIf a red, yellow, or green lens is visibly illuminated, that colour is\nDIRECT visual evidence and MUST be included in the description and,\nwhen useful, distinguishing_features.\n\nExamples:\n- "bottom green lens illuminated"\n- "red lens lit"\n\nDo NOT infer hidden lenses, hidden signal heads, or states that are not\ndirectly visible.\n\nA distant coloured light alone is NOT sufficient evidence for a traffic\nsignal.\n\n==================================================\nSEMANTIC INTERPRETATION\n==================================================\n\nUse semantic interpretation ONLY when the identity is visually supported by\nCLEAR, DIRECT evidence.\n\nEvidence priority:\n1. directly visible text, symbols and pictograms\n2. visible geometry and layout\n3. clearly recognizable semantic identity\n4. generic physical description\n\nDo NOT infer semantic meaning from:\n- shape or colour alone\n- text layout alone\n- mounting position\n- road location\n- surrounding objects\n- common traffic-sign conventions\n- similarity to an ontology class\n\nFor example:\n- a clearly recognizable STOP sign may be identified as a stop sign;\n- a rectangular board with unreadable text should remain a generic\n  rectangular traffic sign with unreadable text unless stronger evidence\n  identifies it.\n\nIf multiple semantic classes remain plausible, DO NOT choose one.\nUse a generic physical traffic-object description instead.\n\nDo NOT assign ontology class IDs.\n\n==================================================\nOBJECT GROUP\n==================================================\n\nEvery kept object must have exactly one object_group.\n\nUse ONLY:\nvehicle\nroad_user\nanimal\ntraffic_sign\ntraffic_signal\nroad_marking\nroad_infrastructure\ninfrastructure\ntemporary_object\ncountry_specific\n\nobject_group is the BROAD PHYSICAL CATEGORY, not the fine-grained ontology\nclass.\n\nExamples:\n- bus/car/truck -> vehicle\n- pedestrian/cyclist/rider -> road_user\n- traffic sign -> traffic_sign\n- traffic light -> traffic_signal\n- crosswalk/lane divider/road arrow -> road_marking\n- guardrail/bollard/delineator/median -> road_infrastructure\n\nDo NOT use fine-grained ontology class names as object_group.\nDo NOT use "person" as object_group.\n\n==================================================\nDESCRIPTION\n==================================================\n\nFor a kept object, write one concise, information-rich description of about\n30–70 words.\n\nDescribe ONLY observable characteristics.\n\nWhere applicable include:\n- colour\n- shape\n- approximate size\n- border and background\n- pictograms\n- arrows\n- active signal state and lens position\n- readable text\n- reflective appearance\n- material only if visually obvious\n- mounting\n- distinctive markings\n- damage\n- partial occlusion\n\nFor signs, describe visible geometry, border, background, pictogram, arrow,\ntext, supplementary plates and mounting.\n\nFor road markings, describe solid/dashed pattern, single/double lines,\ndirection, symbols, continuity, wear and orientation.\n\nFor traffic signals, explicitly describe the illuminated lens when clearly\nvisible.\n\nDo not replace direct visual evidence with "unknown" when something is\nclearly visible.\n\n==================================================\nEXCLUDED OBJECTS\n==================================================\n\nThe following are NOT traffic-object detections and MUST be discarded:\n\n- road\n- road surface\n- asphalt\n- pavement\n- sidewalk\n- buildings\n- houses\n- offices\n- shops\n- vegetation\n- trees\n- sky\n- clouds\n- terrain\n\nAlso discard an invalid localization when the bbox does not actually\ncorrespond to a distinct traffic-related object.\n\n==================================================\nDISCARD DECISION\n==================================================\n\nThe "discard" field is authoritative.\n\nSet:\n"discard": false\nwhen the candidate is a valid traffic-related object that should remain\nrepresented.\n\nSet:\n"discard": true\nwhen the candidate should NOT be represented as a traffic-object\nannotation.\n\nWhen discard is true:\n- "observed_object" MUST be "";\n- "object_group" may be "";\n- "description" may be "";\n- do NOT provide a guessed replacement object.\n\nFor a candidate whose current annotation is "none", this is especially\nimportant: inspect what is actually inside the existing bbox and determine\nwhether it is a valid traffic-related object. If it is, identify it using\nthe same evidence-based rules as the main scene-understanding prompt. If it\nis not, explicitly discard it.\n\n==================================================\nOUTPUT FORMAT\n==================================================\n\nReturn EXACTLY ONE valid JSON object.\n\nReturn ONLY the JSON object.\nNo Markdown.\nNo explanations.\n\nFor a valid object:\n\n{\n    "discard": false,\n    "observed_object": "...",\n    "object_group": "...",\n    "description": "..."\n}\n\nFor an object that must be discarded:\n\n{\n    "discard": true,\n    "observed_object": "",\n    "object_group": "",\n    "description": ""\n}\n\n==================================================\nFINAL VERIFICATION\n==================================================\n\nBefore returning the JSON:\n1. Confirm the candidate region contains a real traffic-related object.\n2. Confirm the proposed identity is supported by direct visual evidence.\n3. Confirm object_group is one of the allowed broad categories.\n4. Confirm no excluded scene element is being returned.\n5. If the candidate is not a valid detectable traffic object, set\n   "discard": true.\n6. Never guess merely because Locate Anything supplied a bbox.\n'


def _semantic_normalize(value: Any) -> str:
    return (
        str(value or "")
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def _semantic_detection_label(
    detection: dict[str, Any],
) -> str:
    value = detection.get("class_name")

    if value is None:
        value = detection.get("final_label")

    return _semantic_normalize(value)


def _semantic_is_recovery_target(
    detection: dict[str, Any],
) -> bool:
    label = _semantic_detection_label(detection)

    return (
        label == _semantic_normalize(UNREADABLE_CLASS)
        or label in NONE_LABELS
    )


def _semantic_is_none_label(
    detection: dict[str, Any],
) -> bool:
    return _semantic_detection_label(detection) in NONE_LABELS




def _semantic_discard_requested(
    semantic_result: dict[str, Any],
) -> bool:
    """Return True when Gemma explicitly instructs the pipeline to discard.

    Accept the canonical boolean field and a few defensive textual forms so
    minor JSON formatting variations cannot accidentally keep a false
    positive. An explicit discard instruction always takes precedence over
    semantic recovery.
    """
    if not isinstance(semantic_result, dict):
        return False

    value = semantic_result.get("discard")

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "yes",
            "discard",
            "drop",
            "remove",
        }

    decision = semantic_result.get("decision")
    if isinstance(decision, str):
        return decision.strip().lower() in {
            "discard",
            "drop",
            "remove",
        }

    action = semantic_result.get("action")
    if isinstance(action, str):
        return action.strip().lower() in {
            "discard",
            "drop",
            "remove",
        }

    return False


def _create_semantic_crop(
    image_path: Path,
    bbox: list | tuple,
    output_dir: Path,
    object_id: Any = None,
) -> Path | None:
    """Create a crop from the existing authoritative LA bbox."""

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
    except Exception as exc:
        logger.warning(
            "Could not open '{}' for semantic verification: {}",
            image_path.name,
            exc,
        )
        return None

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        logger.warning(
            "Invalid semantic-verification bbox for '{}': {}",
            image_path.name,
            bbox,
        )
        image.close()
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        logger.warning(
            "Non-numeric semantic-verification bbox for '{}': {}",
            image_path.name,
            bbox,
        )
        image.close()
        return None

    width, height = image.size

    if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(1, min(x2, width)))
    y2 = int(max(1, min(y2, height)))

    if x2 <= x1 or y2 <= y1:
        image.close()
        return None

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = max(10, int(box_width * 0.15))
    margin_y = max(10, int(box_height * 0.15))

    crop = image.crop(
        (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(width, x2 + margin_x),
            min(height, y2 + margin_y),
        )
    )

    image.close()
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        f"_object_{object_id}"
        if object_id is not None
        else "_unreadable"
    )

    crop_path = (
        output_dir
        / f"{image_path.stem}{suffix}_semantic_verification.png"
    )

    crop.save(crop_path)
    return crop_path


def _parse_semantic_json(
    raw_response: str,
) -> dict[str, Any] | None:
    """Parse Gemma JSON, including fenced/prose-wrapped JSON."""

    text = str(raw_response or "").strip()

    if not text:
        return None

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            return None

        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def _run_semantic_gemma(
    image_path: Path,
    detection: dict[str, Any],
    recovery_model: Any,
    config: Any,
    crop_dir: Path,
    index: int,
    total: int,
) -> dict[str, Any]:
    """Run Gemma recovery for one detection without running Ontology.

    Keeping Gemma and Ontology separate lets the pipeline unload Gemma before
    loading the second Ontology pass, which is important on constrained GPUs.
    """

    is_none_target = _semantic_is_none_label(detection)
    bbox = detection.get("bbox")
    original_bbox = list(bbox) if isinstance(bbox, (list, tuple)) else None

    diagnostic: dict[str, Any] = {
        "attempted": True,
        "recovered": False,
    }

    if bbox is None:
        diagnostic["reason"] = "missing_bbox"
        detection["semantic_verification"] = diagnostic
        return {
            "keep": not is_none_target,
            "scene_object": None,
        }

    crop_path = _create_semantic_crop(
        image_path=image_path,
        bbox=bbox,
        output_dir=crop_dir,
        object_id=detection.get("object_id"),
    )

    if crop_path is None:
        diagnostic["reason"] = "invalid_bbox_or_crop"
        detection["semantic_verification"] = diagnostic
        return {
            "keep": not is_none_target,
            "scene_object": None,
        }

    diagnostic["crop"] = str(crop_path)

    logger.info(
        "Semantic verification %d/%d for '{}'.",
        index,
        total,
        image_path.name,
    )

    try:
        recovery_prompt = SEMANTIC_RECOVERY_PROMPT

        response = recovery_model.infer(
            image_paths=[
                image_path,
                crop_path,
            ],
            prompt=recovery_prompt,
        )

    except Exception as exc:
        diagnostic["reason"] = f"inference_failed: {exc}"
        detection["semantic_verification"] = diagnostic
        logger.warning(
            "Semantic verification inference failed for '{}': {}",
            image_path.name,
            exc,
        )
        return {
            "keep": not is_none_target,
            "scene_object": None,
        }

    responses = (
        response.get("responses", [])
        if isinstance(response, dict)
        else []
    )

    if not responses:
        diagnostic["reason"] = "empty_model_response"
        detection["semantic_verification"] = diagnostic
        return {
            "keep": not is_none_target,
            "scene_object": None,
        }

    raw_response = str(responses[0])
    recovered_scene_object = _parse_semantic_json(raw_response)

    if recovered_scene_object is None:
        diagnostic["reason"] = "invalid_json"
        diagnostic["raw_response"] = raw_response
        detection["semantic_verification"] = diagnostic
        return {
            "keep": not is_none_target,
            "scene_object": None,
        }

    diagnostic["raw_response"] = raw_response
    diagnostic["discard_requested"] = _semantic_discard_requested(
        recovered_scene_object
    )

    if diagnostic["discard_requested"]:
        diagnostic["reason"] = "gemma_discard"
        diagnostic["accounted_for"] = False
        detection["semantic_verification"] = diagnostic

        logger.warning(
            "Gemma instructed semantic verification to discard detection "
            "{} in '{}'. It will not be counted as a final detection.",
            detection.get("object_id", "unknown"),
            image_path.name,
        )

        return {
            "keep": False,
            "scene_object": None,
        }

    object_id = detection.get("object_id")
    recovered_scene_object["object_id"] = object_id

    if not is_none_target:
        recovered_scene_object["object_group"] = "traffic_sign"
    else:
        existing_group = (
            detection.get("object_group")
            or detection.get("superclass")
            or detection.get("group")
        )

        if isinstance(existing_group, str) and existing_group.strip():
            recovered_scene_object["object_group"] = existing_group

    diagnostic["result"] = recovered_scene_object
    detection["semantic_verification"] = diagnostic

    recovered_name = _semantic_normalize(
        recovered_scene_object.get("observed_object")
    )

    if not recovered_name:
        diagnostic["reason"] = "no_class_recovered"
        return {
            "keep": not is_none_target,
            "scene_object": recovered_scene_object,
        }

    if recovered_name == _semantic_normalize(UNREADABLE_CLASS):
        diagnostic["reason"] = "still_unreadable"
        return {
            "keep": not is_none_target,
            "scene_object": recovered_scene_object,
        }

    # Restore the authoritative bbox immediately. Ontology receives the
    # recovered semantics, never a newly inferred localization.
    if original_bbox is not None:
        detection["bbox"] = original_bbox

    return {
        "keep": True,
        "scene_object": recovered_scene_object,
    }


def _apply_semantic_ontology(
    detection: dict[str, Any],
    scene_object: dict[str, Any] | None,
    recovery_ontology: OntologyEngine,
    is_none_target: bool,
) -> bool:
    """Resolve a recovered Scene Understanding object through Ontology."""

    if not isinstance(scene_object, dict):
        return not is_none_target

    try:
        ontology_result = recovery_ontology._reason_object(
            scene_object
        )
    except Exception as exc:
        detection.setdefault(
            "semantic_verification",
            {},
        )["reason"] = f"ontology_failed: {exc}"

        logger.warning(
            "Semantic verification Ontology failed: {}",
            exc,
        )

        return not is_none_target

    detection.setdefault(
        "semantic_verification",
        {},
    )["ontology"] = ontology_result

    prediction = (
        ontology_result.get("prediction", {})
        if isinstance(ontology_result, dict)
        else {}
    )

    if not isinstance(prediction, dict):
        prediction = {}

    recovered_class = str(
        prediction.get("class_name") or ""
    ).strip()

    if (
        not recovered_class
        or _semantic_normalize(recovered_class)
        == _semantic_normalize(UNREADABLE_CLASS)
    ):
        detection["semantic_verification"]["reason"] = (
            "ontology_unresolved"
        )
        return not is_none_target

    # Semantic update ONLY. The original LA bbox remains authoritative.
    original_bbox = list(detection.get("bbox", []))

    detection["bbox"] = original_bbox
    detection["class_id"] = prediction.get("class_id")
    detection["class_name"] = recovered_class
    detection["final_label"] = recovered_class
    detection["grounding_prompt"] = prediction.get("grounding_prompt")
    detection["score"] = prediction.get("score", 0.0)

    if scene_object.get("object_group"):
        detection["object_group"] = scene_object["object_group"]

    detection["semantic_verification"]["recovered"] = True
    detection["semantic_verification"]["accounted_for"] = True

    return True


def _save_semantic_diagnostic(
    config: Any,
    image_path: Path,
    pipeline_cache_snapshot: dict[str, Any],
    pre_semantic_detections: list[dict[str, Any]],
    final_detections: list[dict[str, Any]],
    visualization_path: Path,
    semantic_summary: dict[str, Any],
) -> Path:
    """Save complete PipelineCache + semantic before/after diagnostics."""

    results_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "image": str(image_path),
        "image_name": image_path.name,
        "pipeline_cache": pipeline_cache_snapshot,
        "pre_semantic_detections": pre_semantic_detections,
        "detections": final_detections,
        "semantic_verification_summary": semantic_summary,
        "visualization_path": str(visualization_path),
    }

    output_path = results_dir / f"{image_path.stem}.json"

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return output_path


def run_semantic_verification_stage(
    config: Any,
    cache: PipelineCache,
    annotation_engine: AnnotationEngine,
    image_paths: list[Path],
    detections_by_image: dict[str, list[dict[str, Any]]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    float,
    int,
    int,
    int,
]:
    """Run semantic verification as an in-process pipeline stage.

    Returns:
        updated detections,
        stage wall-clock time,
        target count,
        recovered count,
        dropped-None count.
    """

    stage_start = time.perf_counter()

    semantic_verified_dir = (
        config.paths.outputs
        / "semantic_verified"
    )
    crop_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "crops"
    )

    semantic_verified_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Snapshot the exact post-annotation state before semantic mutation.
    # This is separate from the PipelineCache snapshot because it shows what
    # the annotation stage actually handed to semantic verification.
    pre_semantic_detections_by_image: dict[
        str, list[dict[str, Any]]
    ] = {}

    for image_path in image_paths:
        detections = detections_by_image.get(
            image_path.name,
            [],
        )
        pre_semantic_detections_by_image[image_path.name] = [
            json.loads(
                json.dumps(
                    detection,
                    ensure_ascii=False,
                    default=str,
                )
            )
            for detection in detections
            if isinstance(detection, dict)
        ]

    all_targets: list[
        tuple[Path, dict[str, Any], int, int]
    ] = []

    for image_path in image_paths:
        detections = detections_by_image.get(
            image_path.name,
            [],
        )

        targets = [
            detection
            for detection in detections
            if _semantic_is_recovery_target(detection)
        ]

        for index, detection in enumerate(
            targets,
            start=1,
        ):
            all_targets.append(
                (
                    image_path,
                    detection,
                    index,
                    len(targets),
                )
            )

    target_count = len(all_targets)

    target_counts_by_image: dict[str, int] = {}
    for image_path, _detection, _index, _total in all_targets:
        target_counts_by_image[image_path.name] = (
            target_counts_by_image.get(image_path.name, 0) + 1
        )

    # Every image passing through this enabled stage gets a complete
    # semantic_verified visualization, even when it has zero recovery targets.
    # This makes the stage output unambiguous.
    recovery_model = None
    recovery_ontology = None

    # Cache snapshots are copied before semantic mutation for diagnostics.
    cache_snapshots: dict[str, dict[str, Any]] = {}

    for image_path in image_paths:
        image_cache = getattr(cache, "_cache", {}).get(
            image_path.name,
            {},
        )

        if isinstance(image_cache, dict):
            cache_snapshots[image_path.name] = json.loads(
                json.dumps(
                    image_cache,
                    ensure_ascii=False,
                    default=str,
                )
            )
        else:
            cache_snapshots[image_path.name] = {}

    semantic_results: dict[str, dict[str, Any]] = {}
    dropped_ids_by_image: dict[str, set[int]] = {}

    try:
        if target_count:
            logger.info("")
            logger.info("=" * 80)
            logger.info("SEMANTIC VERIFICATION")
            logger.info("=" * 80)
            logger.info(
                "Found {} recovery target(s) across {} image(s).",
                target_count,
                len(image_paths),
            )

            # ----------------------------------------------------------
            # Pass 1: Gemma only
            # ----------------------------------------------------------
            logger.info(
                "Loading semantic verification Gemma model."
            )

            recovery_model = create_scene_model(config)
            recovery_model.load()

            for (
                image_path,
                detection,
                index,
                total,
            ) in all_targets:
                result = _run_semantic_gemma(
                    image_path=image_path,
                    detection=detection,
                    recovery_model=recovery_model,
                    config=config,
                    crop_dir=crop_dir,
                    index=index,
                    total=total,
                )

                semantic_results.setdefault(
                    image_path.name,
                    {},
                ).setdefault(
                    "gemma",
                    [],
                ).append(result)

                if not result.get("keep", True):
                    dropped_ids_by_image.setdefault(
                        image_path.name,
                        set(),
                    ).add(id(detection))

            # Do not keep Gemma resident while loading Ontology.
            cleanup_stage_resources(
                recovery_model,
            )
            recovery_model = None

            # ----------------------------------------------------------
            # Pass 2: Ontology only for recoverable semantic outputs
            # ----------------------------------------------------------
            ontology_jobs = []

            for image_path, detection, _index, _total in all_targets:
                semantic_info = detection.get(
                    "semantic_verification",
                    {},
                )

                scene_object = semantic_info.get(
                    "result"
                )

                if semantic_info.get("discard_requested", False):
                    continue

                recovered_name = _semantic_normalize(
                    scene_object.get("observed_object")
                    if isinstance(scene_object, dict)
                    else ""
                )

                if (
                    isinstance(scene_object, dict)
                    and recovered_name
                    and recovered_name
                    != _semantic_normalize(UNREADABLE_CLASS)
                ):
                    ontology_jobs.append(
                        (
                            image_path,
                            detection,
                            scene_object,
                        )
                    )

            if ontology_jobs:
                logger.info(
                    "Loading semantic verification Ontology engine "
                    "for {} recovered candidate(s).",
                    len(ontology_jobs),
                )

                recovery_ontology = OntologyEngine(
                    config=config,
                )

                for image_path, detection, scene_object in ontology_jobs:
                    keep = _apply_semantic_ontology(
                        detection=detection,
                        scene_object=scene_object,
                        recovery_ontology=recovery_ontology,
                        is_none_target=_semantic_is_none_label(
                            detection
                        ),
                    )

                    if not keep and _semantic_is_none_label(detection):
                        dropped_ids_by_image.setdefault(
                            image_path.name,
                            set(),
                        ).add(id(detection))

            # ----------------------------------------------------------
            # Unresolved None labels are dropped. Unreadable signs remain.
            # ----------------------------------------------------------
            for image_path, detection, _index, _total in all_targets:
                semantic_info = detection.get(
                    "semantic_verification",
                    {},
                )

                if _semantic_is_none_label(detection):
                    # If it was not successfully resolved, remove it.
                    if not semantic_info.get("recovered", False):
                        dropped_ids_by_image.setdefault(
                            image_path.name,
                            set(),
                        ).add(id(detection))

        else:
            logger.info(
                "Semantic verification enabled, but no unreadable/None "
                "detections require recovery."
            )

        recovered_count = 0
        dropped_count = 0

        # --------------------------------------------------------------
        # Final full-image rendering + diagnostics
        # --------------------------------------------------------------
        for image_path in image_paths:
            detections = detections_by_image.get(
                image_path.name,
                [],
            )

            dropped_ids = dropped_ids_by_image.get(
                image_path.name,
                set(),
            )

            if dropped_ids:
                detections = [
                    detection
                    for detection in detections
                    if id(detection) not in dropped_ids
                ]

                detections_by_image[image_path.name] = detections
                dropped_count += len(dropped_ids)

            recovered_count += sum(
                1
                for detection in detections
                if detection.get(
                    "semantic_verification",
                    {},
                ).get("recovered", False)
            )

            visualization_path = (
                semantic_verified_dir
                / f"{image_path.stem}_annotated.png"
            )

            annotation_engine.visualizer.visualize(
                image_path=image_path,
                detections=detections,
                output_path=visualization_path,
            )

            semantic_summary = {
                "enabled": True,
                "target_count": target_counts_by_image.get(
                    image_path.name,
                    0,
                ),
                "recovered_count": sum(
                    1
                    for detection in detections
                    if detection.get(
                        "semantic_verification",
                        {},
                    ).get("recovered", False)
                ),
                "dropped_none_count": len(dropped_ids),
            }

            diagnostic_path = _save_semantic_diagnostic(
                config=config,
                image_path=image_path,
                pipeline_cache_snapshot=cache_snapshots.get(
                    image_path.name,
                    {},
                ),
                pre_semantic_detections=pre_semantic_detections_by_image.get(
                    image_path.name,
                    [],
                ),
                final_detections=detections,
                visualization_path=visualization_path,
                semantic_summary=semantic_summary,
            )

            semantic_results.setdefault(
                image_path.name,
                {},
            )["diagnostic_path"] = str(diagnostic_path)

            semantic_results[image_path.name][
                "visualization_path"
            ] = str(visualization_path)

        logger.info(
            "Semantic verification summary: {} target(s), "
            "{} recovered, {} None-labeled detection(s) dropped.",
            target_count,
            recovered_count,
            dropped_count,
        )

    finally:
        cleanup_stage_resources(
            recovery_ontology,
            recovery_model,
        )

    return (
        detections_by_image,
        time.perf_counter() - stage_start,
        target_count,
        recovered_count,
        dropped_count,
    )




def load_current_run_final_detections_for_sam(
    config: Any,
    image_paths: list[Path],
    minimum_mtime: float,
) -> dict[str, list[dict[str, Any]]]:
    """Recover current-run final detections if the in-memory map is empty.

    The normal path is the in-memory ``final_detections_by_image`` map.
    This fallback is deliberately limited to JSON artifacts written during
    the CURRENT pipeline run, so an old output cannot silently get segmented.
    Preference:
      1. semantic_verification/results/<stem>.json
      2. semantic_verification/inputs/<stem>.json

    No raw Locate Anything cache is used here because raw grounding has not
    passed the final annotation/postprocessing contract.
    """
    recovered: dict[str, list[dict[str, Any]]] = {}

    results_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "results"
    )
    inputs_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "inputs"
    )

    for image_path in image_paths:
        candidates = [
            results_dir / f"{image_path.stem}.json",
            inputs_dir / f"{image_path.stem}.json",
        ]

        selected = None

        for candidate in candidates:
            try:
                if (
                    candidate.exists()
                    and candidate.stat().st_mtime >= minimum_mtime - 2.0
                ):
                    selected = candidate
                    break
            except OSError:
                continue

        if selected is None:
            continue

        try:
            payload = json.loads(
                selected.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.warning(
                "Could not read current-run SAM fallback JSON '{}': {}",
                selected,
                exc,
            )
            continue

        detections = payload.get("detections")

        if not isinstance(detections, list):
            continue

        clean_detections = [
            detection
            for detection in detections
            if isinstance(detection, dict)
        ]

        recovered[image_path.name] = clean_detections

        logger.warning(
            "SAM 2 fallback recovered {} current-run final detection(s) "
            "for '{}' from '{}'.",
            len(clean_detections),
            image_path.name,
            selected,
        )

    return recovered

# ============================================================================
# FINAL OUTPUT COMPOSITING
# ============================================================================

def _load_rgba_image(path: Path, size: tuple[int, int] | None = None) -> Image.Image | None:
    """Load an image as RGBA, optionally resizing it to the requested size."""
    try:
        image = Image.open(path).convert("RGBA")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image
    except Exception as exc:
        logger.warning("Could not load final-output image '{}': {}", path, exc)
        return None


def _overlay_binary_mask(
    base: Image.Image,
    mask_path: Path,
    rgba: tuple[int, int, int, int],
) -> Image.Image:
    """Overlay a YOLOPv2 binary mask on an RGBA image."""
    if not mask_path.exists():
        return base

    try:
        mask = Image.open(mask_path).convert("L")
        if mask.size != base.size:
            mask = mask.resize(base.size, Image.Resampling.NEAREST)

        alpha = np.asarray(mask, dtype=np.uint8)
        if not np.any(alpha):
            return base

        overlay = np.zeros(
            (base.height, base.width, 4),
            dtype=np.uint8,
        )
        overlay[:, :, 0] = rgba[0]
        overlay[:, :, 1] = rgba[1]
        overlay[:, :, 2] = rgba[2]
        overlay[:, :, 3] = (
            (alpha > 0).astype(np.uint8) * rgba[3]
        )

        return Image.alpha_composite(
            base,
            Image.fromarray(overlay, mode="RGBA"),
        )
    except Exception as exc:
        logger.warning(
            "Could not overlay YOLOPv2 mask '{}': {}",
            mask_path,
            exc,
        )
        return base


def _draw_final_detections(
    base: Image.Image,
    detections: list[dict[str, Any]],
) -> Image.Image:
    """Draw final fused detections on the already segmented base image."""
    rendered = base.copy().convert("RGBA")
    draw = ImageDraw.Draw(rendered)

    for detection in detections:
        if not isinstance(detection, dict):
            continue

        bbox = detection.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            continue

        label = (
            detection.get("class_name")
            or detection.get("final_label")
            or detection.get("observed_object")
            or detection.get("grounding_prompt")
            or detection.get("source")
            or "object"
        )

        try:
            score = float(detection.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        text_label = f"{label} {score:.2f}"

        # Keep YOLOPv2 and VLM/LA visually distinguishable while retaining
        # one unified final image.
        source = str(detection.get("source", "")).lower()
        outline = (255, 80, 80, 255) if source == "yolopv2" else (80, 160, 255, 255)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=outline,
            width=2,
        )

        try:
            text_bbox = draw.textbbox((x1, y1), text_label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except Exception:
            text_width = max(20, len(text_label) * 7)
            text_height = 14

        text_y = max(0, y1 - text_height - 2)

        # Small background improves readability without hiding the image.
        draw.rectangle(
            (x1, text_y, x1 + text_width + 4, text_y + text_height + 2),
            fill=(0, 0, 0, 170),
        )
        draw.text(
            (x1 + 2, text_y + 1),
            text_label,
            fill=(255, 255, 255, 255),
        )

    return rendered


def create_final_output(
    config: Any,
    image_path: Path,
    final_detections: list[dict[str, Any]],
) -> Path:
    """Create the single user-facing final result in outputs/FINAL.

    Composition order:
        1. original image
        2. YOLOPv2 drivable-area mask
        3. YOLOPv2 lane-line mask
        4. SAM 2 segmented visualization, when available
        5. final fused detection boxes/labels

    SAM's segmented image is used as the visual base when available, while
    YOLOPv2 semantic road masks are explicitly composited so both model
    outputs are visible in the same final artifact.
    """

    final_dir = config.paths.outputs / "FINAL"
    final_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    final_path = final_dir / f"{stem}_FINAL.png"

    segmentation_path = (
        config.paths.outputs
        / "segmentation"
        / f"{stem}_segmented.png"
    )

    yolopv2_dir = (
        config.paths.outputs
        / "yolopv2_segmentation"
    )

    drivable_mask_path = (
        yolopv2_dir / "masks" / f"{stem}_drivable.png"
    )
    lane_mask_path = (
        yolopv2_dir / "masks" / f"{stem}_lane.png"
    )

    base = None

    # Prefer the SAM 2 output because it contains the final instance
    # segmentation. If SAM is disabled/failed, use the original image.
    if segmentation_path.exists():
        base = _load_rgba_image(segmentation_path)

    if base is None:
        base = _load_rgba_image(image_path)

    if base is None:
        raise RuntimeError(
            f"Could not load an image for final output: {image_path}"
        )

    # Add YOLOPv2's native semantic road outputs to the same final image.
    base = _overlay_binary_mask(
        base,
        drivable_mask_path,
        (80, 200, 120, 80),
    )

    base = _overlay_binary_mask(
        base,
        lane_mask_path,
        (255, 210, 60, 180),
    )

    # Draw the authoritative final fused detections last, so labels/boxes
    # remain visible above both segmentation layers.
    base = _draw_final_detections(
        base,
        final_detections,
    )

    base.convert("RGB").save(final_path)

    logger.info(
        "FINAL combined output saved for '{}' -> '{}'.",
        image_path.name,
        final_path,
    )

    return final_path

# ============================================================================
# Main Pipeline
# ============================================================================


def main() -> None:
    """
    Execute the VLM-First pipeline.

    Current Pipeline

        Random Frame Download (if required)
                │
                ▼
        Image Discovery
                │
                ▼
        Scene Understanding
                │
                ▼
        Pipeline Cache
                │
                ▼
        Ontology Reasoning
                │
                ▼
        Locate Anything Grounding
                │
                ▼
        Annotation + Postprocessing
                │
                ▼
        Semantic Verification (optional)
                │
                ▼
        Final Visualization
    """

    # ------------------------------------------------------------------
    # End-to-end pipeline timer. Starts before configuration/logging setup.
    # ------------------------------------------------------------------

    pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Load Configuration
    # ------------------------------------------------------------------

    config = load_config()

    config.paths.ensure_output_dirs()

    logs(
        show_level=config.logging.level.lower(),
        save_level=config.logging.level.lower(),
        program_name="vlm_first_pipeline",
        path=config.paths.logs,
    )
    config = load_config()

    config.paths.ensure_output_dirs()

    logger.info("=" * 80)
    logger.info("VLM-First Pipeline")
    logger.info("=" * 80)


    # ------------------------------------------------------------------
    # Command-line Arguments
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="VLM-First Pipeline",
    )

    parser.add_argument(
        "--images",
        type=int,
        default=config.pipeline.num_images,
        help="Number of images to process (0 = all).",
    )

    parser.add_argument(
        "--no-sam-segmentation",
        action="store_true",
        help="Skip the final SAM 2 segmentation stage.",
    )
    parser.add_argument(
        "--no-yolopv2",
        action="store_true",
        help="Skip the auxiliary YOLOPv2 road-perception stage.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Prepare Random Frames
    # ------------------------------------------------------------------

    if config.pipeline.download_random_frames:

        prepare_random_frames(config)

    # ------------------------------------------------------------------
    # Discover Images
    # ------------------------------------------------------------------

    logger.info(
        "Searching for images in '{}'.",
        config.paths.random_frames,
    )

    image_paths = discover_images(
        config.paths.random_frames,
        config.pipeline.supported_image_extensions,
    )

    if not image_paths:

        logger.error(
            "No supported images found in '{}'.",
            config.paths.random_frames,
        )

        sys.exit(1)

    if args.images > 0:

        image_paths = image_paths[: args.images]

    logger.info(
        "Discovered {} image(s).",
        len(image_paths),
    )

    # ------------------------------------------------------------------
    # Initialize Pipeline Cache
    # ------------------------------------------------------------------

    logger.info(
        "Initializing Pipeline Cache."
    )

    cache = PipelineCache()

    monitor = GPUMonitor(
        output_dir=config.paths.outputs,
    )

    monitor.start()

    total_generation_time = 0.0

    total_generated_tokens = 0

    total_images = 0

    total_objects = 0
    total_reasoning_time = 0.0

    total_grounding_time = 0.0
    total_grounding_objects = 0

    # Exclusive wall-clock intervals covering the complete run.
    initialization_stage_time = 0.0
    scene_stage_time = 0.0
    ontology_stage_time = 0.0
    grounding_stage_time = 0.0
    annotation_stage_time = 0.0
    semantic_verification_stage_time = 0.0
    sam_segmentation_stage_time = 0.0
    yolopv2_stage_time = 0.0
    final_output_stage_time = 0.0
    cleanup_stage_time = 0.0

    initialization_stage_start = pipeline_start

    annotation_engine = AnnotationEngine(
        pipeline_cache_dir=config.paths.pipeline_cache,
        output_dir=config.paths.annotations,
    )

    # Stage-owned references are initialized explicitly so later cleanup
    # remains safe when a stage is disabled by configuration.
    engine = None
    model = None
    ontology_engine = None
    grounding_engine = None

    # Annotation results are retained in memory so semantic verification can
    # run immediately after annotation without writing/reloading an input file.
    results: dict[str, dict[str, Any]] = {}
    stage_results: dict[str, dict[str, Any]] = {}
    processed_count = 0

    initialization_stage_time = (
        time.perf_counter()
        - initialization_stage_start
    )

    # ------------------------------------------------------------------
    # Scene Understanding
    # ------------------------------------------------------------------

    scene_stage_start = time.perf_counter()

    failed_images: list[tuple[str, str, str]] = []

    if config.pipeline.run_scene_understanding:

        # --------------------------------------------------------------
        # Create Scene Understanding Model
        # --------------------------------------------------------------

        logger.info(
            "Creating Scene Understanding model."
        )

        try:

            model = create_scene_model(config)

        except Exception:

            logger.error(
                "Failed to create Scene Understanding model."
            )

            raise

        # --------------------------------------------------------------
        # Load Model
        # --------------------------------------------------------------

        logger.info(
            "Loading model '{}'.",
            model.model_name,
        )

        model.load()

        logger.info(
            "Model successfully loaded."
        )

        # --------------------------------------------------------------
        # Create Scene Understanding Engine
        # --------------------------------------------------------------

        engine = SceneUnderstandingEngine(
            config=config,
            model=model,
        )

        # --------------------------------------------------------------
        # Process Images
        # --------------------------------------------------------------

        BATCH_SIZE = config.pipeline.gemma_batch_size

        try:

            for batch_start in range(0, len(image_paths), BATCH_SIZE):

                batch = image_paths[
                    batch_start : batch_start + BATCH_SIZE
                ]

                logger.info("")
                logger.info("=" * 80)
                logger.info(
                    "Processing batch {}-{} of {} ({} image(s))",
                    batch_start + 1,
                    min(batch_start + len(batch), len(image_paths)),
                    len(image_paths),
                    len(batch),
                )
                logger.info(
                    "GPU BEFORE BATCH"
                )

                logger.info(
                    "Allocated : {} GB",
                    round(torch.cuda.memory_allocated() / 1024**3, 2),
                )

                logger.info(
                    "Reserved : {} GB",
                    round(torch.cuda.memory_reserved() / 1024**3, 2),
                )
                logger.info("=" * 80)

                try:

                    objects_per_image = engine.process_images(
                        image_paths=batch,
                        cache=cache,
                    )

                    total_generation_time += engine.last_generation_time

                    total_generated_tokens += engine.last_generated_tokens

                    total_images += len(batch)

                    for image_path, objects in zip(
                        batch,
                        objects_per_image,
                    ):

                        logger.info(
                            "Detected {} traffic object(s) in '{}'.",
                            len(objects),
                            image_path.name,
                        )

                        logger.debug(
                            "Parsed Scene Understanding JSON for '{}':\n{}",
                            image_path.name,
                            json.dumps(
                                objects,
                                indent=4,
                                ensure_ascii=False,
                            ),
                        )

                except Exception as e:

                    logger.error(
                        "Batch failed: {}",
                        e,
                    )

                    for image_path in batch:

                        failed_images.append(
                            (
                                image_path.name,
                                "Scene Understanding",
                                str(e),
                            )
                        )

        finally:

            logger.info(
                "Unloading Scene Understanding model."
            )

            

            model.unload()

            cleanup_stage_resources(
                engine,
                model,
            )

            log_gpu_memory(
                "AFTER SCENE UNDERSTANDING CLEANUP"
            )

            logger.info("")
            logger.info("=" * 80)
            logger.info("SCENE UNDERSTANDING SUMMARY")
            logger.info("=" * 80)

            logger.info(
                "Images Processed      : {}",
                total_images,
            )

            logger.info(
                "Total Generation Time : {:.2f} s",
                total_generation_time,
            )

            if total_images:

                logger.info(
                    "Average/Image        : {:.2f} s",
                    total_generation_time / total_images,
                )

                logger.info(
                    "Images/Hour          : {:.2f}",
                    (3600 * total_images)
                    / total_generation_time,
                )

            logger.info(
                "Generated Tokens      : {}",
                total_generated_tokens,
            )

            if total_generation_time:

                logger.info(
                    "Tokens/sec           : {:.2f}",
                    total_generated_tokens
                    / total_generation_time,
                )

            else:

                logger.info(
                    "Skipping Scene Understanding."
                )

    scene_stage_time = (
        time.perf_counter()
        - scene_stage_start
    )

    # ------------------------------------------------------------------
    # Ontology Reasoning
    # ------------------------------------------------------------------

    ontology_stage_start = time.perf_counter()

    if config.pipeline.run_ontology_reasoning:

        logger.info("")
        logger.info("=" * 80)
        logger.info("ONTOLOGY REASONING")
        logger.info("=" * 80)

        ontology_engine = OntologyEngine(
            config=config,
        )

        total_reasoning_time = 0.0

        total_objects = 0

        for image_path in image_paths:

            try:

                ontology_results = ontology_engine.process_images(
                    image_name=image_path.name,
                    cache=cache,
                )

                total_reasoning_time += (
                    ontology_engine.last_reasoning_time
                )

                total_objects += (
                    ontology_engine.last_objects_processed
                )

                logger.info(
                    "Ontology completed for '{}'.",
                    image_path.name,
                )

                logger.info(
                    "Retrieved {} ontology prediction(s).",
                    len(ontology_results),
                )

            except Exception as e:

                logger.error(
                    "Ontology failed for '{}': {}",
                    image_path.name,
                    e,
                )

                failed_images.append(
                    (
                        image_path.name,
                        "Ontology Reasoning",
                        str(e),
                    )
                )

        logger.info("")
        logger.info("=" * 80)
        logger.info("ONTOLOGY SUMMARY")
        logger.info("=" * 80)

        logger.info(
            "Objects Processed : {}",
            total_objects,
        )

        logger.info(
            "Total Time : {:.2f} s",
            total_reasoning_time,
        )

        if total_objects:

            logger.info(
                "Average/Object : {:.4f} s",
                total_reasoning_time / total_objects,
            )

    ontology_stage_time = (
        time.perf_counter()
        - ontology_stage_start
    )

    # ------------------------------------------------------------------
    # Locate Anything Grounding
    # ------------------------------------------------------------------

    grounding_stage_start = time.perf_counter()

    if config.pipeline.run_grounding:

        logger.info("")
        logger.info("=" * 80)
        logger.info("LOCATE ANYTHING GROUNDING")
        logger.info("=" * 80)

        try:

            logger.info(
                "Creating LocateAnythingEngine."
            )

            grounding_engine = LocateAnythingEngine(
                config=config,
            )

            logger.info(
                "LocateAnythingEngine successfully initialized."
            )

            grounding_start = time.perf_counter()

            grounding_results = grounding_engine.process_images(
                image_paths=image_paths,
                cache=cache,
            )

            total_grounding_time = (
                time.perf_counter()
                - grounding_start
            )

            total_grounding_objects = (
                grounding_engine.last_objects_processed
            )

            logger.info("")
            logger.info(
                "Locate Anything completed successfully."
            )

            logger.info(
                "Images Processed : {}",
                grounding_engine.last_images_processed,
            )

            logger.info(
                "Objects Processed : {}",
                grounding_engine.last_objects_processed,
            )

            logger.info(
                "Grounding Time : {:.2f} s",
                total_grounding_time,
            )

            if grounding_engine.last_images_processed:

                logger.info(
                    "Average/Image : {:.2f} s",
                    total_grounding_time
                    / grounding_engine.last_images_processed,
                )

        except Exception as e:

            logger.error(
                "Locate Anything grounding failed: {}",
                e,
            )

            failed_images.extend(
                (
                    image_path.name,
                    "Locate Anything",
                    str(e),
                )
                for image_path in image_paths
            )

        logger.info("")
        logger.info("=" * 80)
        logger.info("LOCATE ANYTHING SUMMARY")
        logger.info("=" * 80)

        logger.info(
            "Objects Processed : {}",
            total_grounding_objects,
        )

        logger.info(
            "Total Time : {:.2f} s",
            total_grounding_time,
        )

        if total_grounding_objects:

            logger.info(
                "Average/Object : {:.4f} s",
                total_grounding_time
                / total_grounding_objects,
            )

    grounding_stage_time = (
        time.perf_counter()
        - grounding_stage_start
    )

    # ==============================================================
    # YOLOPv2 AUXILIARY ROAD PERCEPTION
    # ==============================================================

    # YOLOPv2 replaces the old BDD100K auxiliary branch.
    # Generic road users come from YOLOPv2; signs/signals, specialized
    # vehicles and other out-of-set traffic objects remain with the VLM.
    if not args.no_yolopv2 and getattr(
        config.pipeline,
        "run_yolopv2",
        True,
    ):
        yolopv2_stage_start = time.perf_counter()
        yolopv2_engine = None

        logger.info("")
        logger.info("=" * 80)
        logger.info("YOLOPv2 AUXILIARY ROAD PERCEPTION")
        logger.info("=" * 80)

        try:
            yolopv2_engine = YOLOPv2SegmentationEngine(
                output_dir=(
                    config.paths.outputs
                    / "yolopv2_segmentation"
                ),
                device=getattr(
                    config.pipeline,
                    "yolopv2_device",
                    None,
                ),
                image_size=getattr(
                    config.pipeline,
                    "yolopv2_image_size",
                    640,
                ),
                confidence_threshold=getattr(
                    config.pipeline,
                    "yolopv2_confidence_threshold",
                    0.30,
                ),
                iou_threshold=getattr(
                    config.pipeline,
                    "yolopv2_iou_threshold",
                    0.45,
                ),
            )

            yolopv2_detections_by_image = (
                yolopv2_engine.process_images(image_paths)
            )

            logger.info(
                "YOLOPv2 completed successfully. Detection counts: {}",
                {
                    image_name: len(detections)
                    for image_name, detections
                    in yolopv2_detections_by_image.items()
                },
            )

        except Exception as exc:
            logger.error(
                "YOLOPv2 auxiliary stage failed: {}",
                exc,
            )
            failed_images.extend(
                (
                    image_path.name,
                    "YOLOPv2",
                    str(exc),
                )
                for image_path in image_paths
            )

        finally:
            cleanup_stage_resources(yolopv2_engine)
            yolopv2_stage_time = (
                time.perf_counter() - yolopv2_stage_start
            )
    else:
        logger.info(
            "YOLOPv2 auxiliary road perception skipped."
        )

    # ==============================================================
    # ANNOTATION
    # ==============================================================

    annotation_stage_start = time.perf_counter()

    final_detections_by_image: dict[str, list[dict[str, Any]]] = {}
    # Independent auxiliary YOLOPv2 results. These bypass the LA
    # annotation/postprocessing/semantic-verification branch.
    yolopv2_detections_by_image: dict[str, list[dict[str, Any]]] = {}

    if config.pipeline.run_annotation:

        RETRY_THRESHOLD = 30

        for image_path in image_paths:

            processed_count += 1
            retry_attempted = False

            try:

                # ------------------------------------------------------
                # First annotation pass
                # ------------------------------------------------------

                final_detections, visualization_path = (
                    run_annotation_stage(
                        annotation_engine=annotation_engine,
                        image_path=image_path,
                    )
                )



                logger.info(
                    "Annotation completed for {}",
                    image_path.name,
                )

                logger.info(
                    "Initial final detections: {}",
                    len(final_detections),
                )

                # ------------------------------------------------------
                # ONE Locate Anything retry.
                #
                # If one or more classes were completely missed by the first
                # grounding pass, retry with ONLY those classes. Existing
                # grounding results are preserved and the recovered boxes
                # are merged into them.
                #
                # If every class already has grounding but the final
                # annotation count is >30, preserve the old full-prompt
                # replacement retry.
                #
                # Exactly one retry is allowed per image.
                # ------------------------------------------------------

                try:
                    (
                        _scene_objects,
                        missing_grounding_objects,
                    ) = get_missing_grounding_classes(
                        grounding_engine=grounding_engine,
                        cache=cache,
                        image_path=image_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not determine missing grounding classes "
                        "for '{}': {}",
                        image_path.name,
                        exc,
                    )
                    missing_grounding_objects = []

                needs_grounding_retry = bool(
                    missing_grounding_objects
                    or len(final_detections) > RETRY_THRESHOLD
                )

                if (
                    needs_grounding_retry
                    and not retry_attempted
                ):

                    retry_attempted = True

                    if missing_grounding_objects:
                        logger.warning(
                            "Running targeted Locate Anything retry for "
                            "'{}' to recover {} missing class(es).",
                            image_path.name,
                            len(missing_grounding_objects),
                        )

                        rerun_locate_anything_for_missing_classes(
                            grounding_engine=grounding_engine,
                            cache=cache,
                            image_path=image_path,
                        )

                    else:
                        logger.warning(
                            "Detection count for '{}' is {} (> {}). "
                            "Running one full Locate Anything retry.",
                            image_path.name,
                            len(final_detections),
                            RETRY_THRESHOLD,
                        )

                        clear_grounding_for_retry(
                            cache=cache,
                            image_name=image_path.name,
                        )

                        grounding_engine.process_images(
                            image_paths=[image_path],
                            cache=cache,
                        )

                    logger.info(
                        "Locate Anything retry completed for '{}'.",
                        image_path.name,
                    )

                    # The annotation engine now sees:
                    #   existing successful grounding
                    #   +
                    #   newly recovered grounding
                    final_detections, visualization_path = (
                        run_annotation_stage(
                            annotation_engine=annotation_engine,
                            image_path=image_path,
                        )
                    )

                    logger.info(
                        "Retry annotation completed for {}",
                        image_path.name,
                    )

                    logger.info(
                        "Retry final detections: {}",
                        len(final_detections),
                    )

                    if len(final_detections) > RETRY_THRESHOLD:
                        logger.warning(
                            "Retry for '{}' still produced {} detections "
                            "(> {}). No second retry will be performed.",
                            image_path.name,
                            len(final_detections),
                            RETRY_THRESHOLD,
                        )

                # ------------------------------------------------------
                # Final result
                # ------------------------------------------------------

                logger.info(
                    "Final detections: {}",
                    len(final_detections),
                )

                # If the FINAL result is still over the threshold after
                # the single retry, route its annotated visualization to
                # defected_annotations instead of annotations.
                if (
                    len(final_detections) > RETRY_THRESHOLD
                    and not config.pipeline.run_semantic_verification
                ):
                    visualization_path = route_defected_annotation(
                        annotation_engine=annotation_engine,
                        image_path=image_path,
                        visualization_path=visualization_path,
                        detections_count=len(final_detections),
                        threshold=RETRY_THRESHOLD,
                    )

                logger.info(
                    "Final visualization: {}",
                    visualization_path,
                )

                final_detections_by_image[image_path.name] = final_detections

                stage_results[image_path.name] = {
                    "status": "completed",
                    "detections": len(final_detections),
                    "visualization_path": str(
                        visualization_path
                    ),
                }

                results[image_path.name] = {
                    "annotation": stage_results[image_path.name],
                }

                # Keep the base annotation input as an intermediate artifact
                # for diagnostics/reprocessing. When semantic verification is
                # enabled, the actual verification below uses the in-memory
                # detections directly, just like the other pipeline stages.
                save_semantic_verification_input(
                    config=config,
                    image_path=image_path,
                    final_detections=final_detections,
                )

            except Exception as exc:

                logger.error(
                    "Annotation failed for {}",
                    image_path.name,
                )

                stage_results[image_path.name] = {
                    "status": "failed",
                    "error": str(exc),
                }

                results[image_path.name] = {
                    "annotation": stage_results[
                        image_path.name
                    ],
                }

                failed_images.append(
                    (
                        image_path.name,
                        "Annotation",
                        str(exc),
                    ),
                )
    annotation_stage_time = (
        time.perf_counter()
        - annotation_stage_start
    )

    # ==============================================================
    # SEMANTIC VERIFICATION
    # ==============================================================

    semantic_target_count = 0
    semantic_recovered_count = 0
    semantic_dropped_count = 0

    if config.pipeline.run_semantic_verification:
        (
            final_detections_by_image,
            semantic_verification_stage_time,
            semantic_target_count,
            semantic_recovered_count,
            semantic_dropped_count,
        ) = run_semantic_verification_stage(
            config=config,
            cache=cache,
            annotation_engine=annotation_engine,
            image_paths=image_paths,
            detections_by_image=final_detections_by_image,
        )

        # Update final annotation result metadata to point to the fully
        # verified visualization. The base annotation remains available in
        # outputs/annotations, while semantic_verified is the post-verification
        # final image.
        for image_path in image_paths:
            image_name = image_path.name
            if image_name not in final_detections_by_image:
                continue

            verified_path = (
                config.paths.outputs
                / "semantic_verified"
                / f"{image_path.stem}_annotated.png"
            )

            if image_name in stage_results:
                stage_results[image_name]["detections"] = len(
                    final_detections_by_image[image_name]
                )
                stage_results[image_name][
                    "visualization_path"
                ] = str(verified_path)
                stage_results[image_name][
                    "semantic_verification"
                ] = {
                    "enabled": True,
                    "targets": sum(
                        1
                        for detection in final_detections_by_image[
                            image_name
                        ]
                        if detection.get(
                            "semantic_verification",
                            {},
                        ).get("attempted", False)
                    ),
                    "recovered": sum(
                        1
                        for detection in final_detections_by_image[
                            image_name
                        ]
                        if detection.get(
                            "semantic_verification",
                            {},
                        ).get("recovered", False)
                    ),
                }

                results[image_name]["annotation"] = stage_results[
                    image_name
                ]

        # Apply the >30 detection routing AFTER semantic verification so
        # the routing decision is based on the true final detection set.
        RETRY_THRESHOLD = 30
        for image_path in image_paths:
            image_name = image_path.name
            detections = final_detections_by_image.get(
                image_name
            )

            if detections is None:
                continue

            if len(detections) <= RETRY_THRESHOLD:
                continue

            verified_path = (
                config.paths.outputs
                / "semantic_verified"
                / f"{image_path.stem}_annotated.png"
            )

            if verified_path.exists():
                final_path = route_defected_annotation(
                    annotation_engine=annotation_engine,
                    image_path=image_path,
                    visualization_path=verified_path,
                    detections_count=len(detections),
                    threshold=RETRY_THRESHOLD,
                )

                if image_name in stage_results:
                    stage_results[image_name][
                        "visualization_path"
                    ] = str(final_path)

                if image_name in results:
                    results[image_name]["annotation"] = stage_results[
                        image_name
                    ]

    # ==============================================================
    # FUSE YOLOPv2 WITH FINAL VLM/LOCATE ANYTHING RESULTS
    # ==============================================================

    # YOLOPv2 is fused only after the LA branch and optional semantic
    # verification are final. Overlapping LA detections take precedence,
    # allowing a specialized VLM label (for example "police car") to replace
    # a generic YOLOPv2 "car" for the same physical object.
    def _bbox_iou(
        bbox_a: list | tuple,
        bbox_b: list | tuple,
    ) -> float:
        if (
            not isinstance(bbox_a, (list, tuple))
            or len(bbox_a) != 4
            or not isinstance(bbox_b, (list, tuple))
            or len(bbox_b) != 4
        ):
            return 0.0

        try:
            ax1, ay1, ax2, ay2 = map(float, bbox_a)
            bx1, by1, bx2, by2 = map(float, bbox_b)
        except (TypeError, ValueError):
            return 0.0

        ax1, ax2 = sorted((ax1, ax2))
        ay1, ay2 = sorted((ay1, ay2))
        bx1, bx2 = sorted((bx1, bx2))
        by1, by2 = sorted((by1, by2))

        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _fuse_yolopv2_detections(
        la_detections: list[dict[str, Any]],
        yolopv2_detections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fused = list(la_detections)

        for yolo_detection in yolopv2_detections:
            if not isinstance(yolo_detection, dict):
                continue

            yolo_bbox = yolo_detection.get("bbox")
            if not isinstance(yolo_bbox, (list, tuple)):
                continue

            # Any IoU >= 0.50 with a final LA detection is treated as the
            # same physical object. LA therefore retains semantic authority.
            duplicate = any(
                _bbox_iou(
                    yolo_bbox,
                    la_detection.get("bbox"),
                ) >= 0.50
                for la_detection in la_detections
                if isinstance(la_detection, dict)
            )

            if duplicate:
                continue

            fused.append({
                **yolo_detection,
                "source": "yolopv2",
                "annotation_postprocessing_applied": False,
                "semantic_verification_applied": False,
            })

        return fused

    if yolopv2_detections_by_image:
        logger.info("")
        logger.info("=" * 80)
        logger.info("FUSING YOLOPv2 ROAD-USER DETECTIONS")
        logger.info("=" * 80)

        for image_path in image_paths:
            image_name = image_path.name

            la_final = final_detections_by_image.get(
                image_name,
                [],
            )
            yolopv2_final = yolopv2_detections_by_image.get(
                image_name,
                [],
            )

            final_detections_by_image[image_name] = (
                _fuse_yolopv2_detections(
                    la_detections=la_final,
                    yolopv2_detections=yolopv2_final,
                )
            )

            logger.info(
                "YOLOPv2 fusion | image='{}' | LA={} | "
                "YOLOPv2 raw={} | final={}",
                image_name,
                len(la_final),
                len(yolopv2_final),
                len(final_detections_by_image[image_name]),
            )

    # ==============================================================
    # FINAL SAM 2 SEGMENTATION
    # ==============================================================

    # Segmentation is deliberately the final model stage. It consumes the
    # final in-memory detections after annotation, postprocessing, optional
    # semantic verification, and >30 routing decisions. SAM 2 receives the
    # existing bbox as a box prompt and never replaces that bbox.
    if not args.no_sam_segmentation:
        sam_segmentation_stage_start = time.perf_counter()

        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL SAM 2 SEGMENTATION")
        logger.info("=" * 80)

        try:
            sam_engine = SAM2SegmentationEngine(
                output_dir=(
                    config.paths.outputs
                    / "segmentation"
                ),
            )

            # SAM 2 must consume the TRUE final detection set.
            # Normally this is the in-memory map. If that map is empty because
            # of a stage/data-flow edge case, recover only artifacts written
            # during THIS run. Never fall back to stale or raw grounding data.
            sam_input_counts = {
                image_path.name: len(
                    final_detections_by_image.get(image_path.name, [])
                )
                for image_path in image_paths
            }

            logger.info(
                "SAM 2 input before model execution: {}",
                sam_input_counts,
            )

            if not any(sam_input_counts.values()):
                fallback_detections = (
                    load_current_run_final_detections_for_sam(
                        config=config,
                        image_paths=image_paths,
                        minimum_mtime=pipeline_start,
                    )
                )

                if fallback_detections:
                    final_detections_by_image.update(
                        fallback_detections
                    )

            sam_input_counts = {
                image_path.name: len(
                    final_detections_by_image.get(image_path.name, [])
                )
                for image_path in image_paths
            }

            logger.info(
                "SAM 2 final input after fallback check: {}",
                sam_input_counts,
            )

            if not any(sam_input_counts.values()):
                raise RuntimeError(
                    "No final detections are available for SAM 2. "
                    "The annotation/semantic stages produced no current-run "
                    "detections, so SAM 2 was not allowed to create a false "
                    "successful output."
                )

            # SAM 2 is loaded, used, and unloaded entirely inside
            # process_images(), so it does not remain resident after the
            # final stage.
            final_detections_by_image = sam_engine.process_images(
                image_paths=image_paths,
                detections_by_image=final_detections_by_image,
            )

            # Update the authoritative final result metadata to point to
            # the SAM-segmented visualization.
            for image_path in image_paths:
                image_name = image_path.name

                if image_name not in final_detections_by_image:
                    continue

                detections = final_detections_by_image[image_name]

                segmentation_path = (
                    config.paths.outputs
                    / "segmentation"
                    / f"{image_path.stem}_segmented.png"
                )

                if image_name in stage_results:
                    stage_results[image_name]["detections"] = len(
                        detections
                    )
                    stage_results[image_name][
                        "visualization_path"
                    ] = str(segmentation_path)
                    stage_results[image_name][
                        "segmentation"
                    ] = {
                        "enabled": True,
                        "model": "SAM 2",
                        "visualization_path": str(
                            segmentation_path
                        ),
                    }

                    results[image_name]["annotation"] = stage_results[
                        image_name
                    ]

            logger.info(
                "Final SAM 2 segmentation completed."
            )

        except Exception as exc:
            logger.error(
                "Final SAM 2 segmentation failed: {}",
                exc,
            )

            failed_images.extend(
                (
                    image_path.name,
                    "SAM 2 Segmentation",
                    str(exc),
                )
                for image_path in image_paths
                if image_path.name in final_detections_by_image
                and final_detections_by_image.get(
                    image_path.name,
                    [],
                )
            )

        finally:
            sam_segmentation_stage_time = (
                time.perf_counter()
                - sam_segmentation_stage_start
            )

    else:
        logger.info(
            "Final SAM 2 segmentation skipped by --no-sam-segmentation."
        )

    # ==============================================================
    # FINAL COMBINED OUTPUT
    # ==============================================================

    # This is the single user-facing output. It combines:
    #   - SAM 2 instance segmentation
    #   - YOLOPv2 drivable-area segmentation
    #   - YOLOPv2 lane-line segmentation
    #   - final fused VLM/Locate Anything + YOLOPv2 detections
    #
    # It is deliberately created AFTER SAM 2 so it represents the true
    # final state of the pipeline.
    final_output_stage_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL COMBINED OUTPUT")
    logger.info("=" * 80)

    for image_path in image_paths:
        image_name = image_path.name
        detections = final_detections_by_image.get(
            image_name,
            [],
        )

        try:
            final_path = create_final_output(
                config=config,
                image_path=image_path,
                final_detections=detections,
            )

            if image_name in stage_results:
                stage_results[image_name]["visualization_path"] = str(
                    final_path
                )
                stage_results[image_name]["final_output_path"] = str(
                    final_path
                )

            if image_name in results:
                results[image_name]["annotation"] = stage_results.get(
                    image_name,
                    results[image_name].get("annotation", {}),
                )

        except Exception as exc:
            logger.error(
                "Final combined output failed for '{}': {}",
                image_name,
                exc,
            )
            failed_images.append(
                (
                    image_name,
                    "FINAL Output",
                    str(exc),
                )
            )

    final_output_stage_time = (
        time.perf_counter() - final_output_stage_start
    )

    # ------------------------------------------------------------------
    # Final stage-resource cleanup
    #
    # This is a defensive boundary for any engine that survived its normal
    # stage-level cleanup. It also makes repeated programmatic execution
    # less likely to retain model references.
    # ------------------------------------------------------------------

    cleanup_stage_start = time.perf_counter()

    cleanup_stage_resources(
        globals().get("grounding_engine"),
        globals().get("ontology_engine"),
        globals().get("engine"),
        globals().get("model"),
    )

    log_gpu_memory(
        "FINAL PIPELINE CLEANUP"
    )

    # ------------------------------------------------------------------
    # Pipeline Complete
    # ------------------------------------------------------------------
    monitor.stop()

    monitor.save()

    cleanup_stage_time = (
        time.perf_counter()
        - cleanup_stage_start
    )

    # Authoritative end-to-end wall-clock time. This includes setup,
    # input preparation, all enabled stages, retries, annotation,
    # postprocessing, visualization, semantic-input saving, and cleanup.
    pipeline_time = time.perf_counter() - pipeline_start

    accounted_pipeline_time = (
        initialization_stage_time
        + scene_stage_time
        + ontology_stage_time
        + grounding_stage_time
        + annotation_stage_time
        + semantic_verification_stage_time
        + yolopv2_stage_time
        + sam_segmentation_stage_time
        + final_output_stage_time
        + cleanup_stage_time
    )

    timing_gap = max(
        0.0,
        pipeline_time - accounted_pipeline_time,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)

    logger.info(
        "Images Processed              : {}",
        total_images,
    )

    logger.info(
        "Objects Processed             : {}",
        total_objects,
    )

    logger.info(
        "Initialization/Input Time     : {:.2f} s",
        initialization_stage_time,
    )

    logger.info(
        "Scene Understanding Time      : {:.2f} s",
        scene_stage_time,
    )

    logger.info(
        "  └─ Generation Time           : {:.2f} s",
        total_generation_time,
    )

    logger.info(
        "Ontology Reasoning Time       : {:.2f} s",
        ontology_stage_time,
    )

    logger.info(
        "  └─ Reasoning Time            : {:.2f} s",
        total_reasoning_time,
    )

    logger.info(
        "Locate Anything Time          : {:.2f} s",
        grounding_stage_time,
    )

    logger.info(
        "  └─ Initial Grounding Time    : {:.2f} s",
        total_grounding_time,
    )

    logger.info(
        "Annotation + Retry + Postproc : {:.2f} s",
        annotation_stage_time,
    )

    logger.info(
        "Semantic Verification Time     : {:.2f} s",
        semantic_verification_stage_time,
    )

    logger.info(
        "SAM 2 Segmentation Time        : {:.2f} s",
        sam_segmentation_stage_time,
    )
    logger.info(
        "YOLOPv2 Auxiliary Time          : {:.2f} s",
        yolopv2_stage_time,
    )

    logger.info(
        "Final Combined Output Time      : {:.2f} s",
        final_output_stage_time,
    )

    logger.info(
        "  └─ Recovery Targets           : {}",
        semantic_target_count,
    )

    logger.info(
        "  └─ Recovered                  : {}",
        semantic_recovered_count,
    )

    logger.info(
        "  └─ Discarded / Unresolved     : {}",
        semantic_dropped_count,
    )

    logger.info(
        "Final Resource Cleanup        : {:.2f} s",
        cleanup_stage_time,
    )

    logger.info(
        "Accounted Pipeline Time       : {:.2f} s",
        accounted_pipeline_time,
    )

    logger.info(
        "Timing Measurement Gap        : {:.2f} s",
        timing_gap,
    )

    logger.info(
        "Total Pipeline Time           : {:.2f} s",
        pipeline_time,
    )

    if total_images:

        logger.info(
            "Average/Image                 : {:.2f} s",
            pipeline_time / total_images,
        )

        if pipeline_time > 0:

            logger.info(
                "Images/Hour                   : {:.2f}",
                3600 * total_images / pipeline_time,
            )
    logger.info("")
    logger.info("=" * 80)
    logger.info("VLM-First Pipeline Complete")
    logger.info("=" * 80)

    if failed_images:

        logger.warning(
            "{} image(s) failed during processing.",
            len(failed_images),
        )

        for image_name, stage, reason in failed_images:

            logger.warning(
                "    {} | Stage: {} | Reason: {}",
                image_name,
                stage,
                reason,
            )

    else:

        logger.info(
            "All images processed successfully."
        )

    # --------------------------------------------------------------
    # Successful annotation outputs
    # --------------------------------------------------------------

    if processed_count > 0:
        logger.info("")
        logger.info("FINAL ANNOTATION OUTPUTS")
        logger.info("-" * 70)

        for image_path in image_paths:

            image_name = image_path.name

            result = results.get(
                image_name
            )

            if not isinstance(
                result,
                dict,
            ):
                continue

            annotation_result = result.get(
                "annotation"
            )

            if not isinstance(
                annotation_result,
                dict,
            ):
                continue

            if (
                annotation_result.get("status")
                != "completed"
            ):
                continue

            detections = annotation_result.get(
                "detections",
                0,
            )

            visualization_path = (
                annotation_result.get(
                    "visualization_path"
                )
            )

            logger.info(
                "  {}",
                image_name,
            )

            logger.info(
                "      Final detections : {}",
                detections,
            )

            if visualization_path:
                logger.info(
                    "      Visualization    : {}",
                    visualization_path,
                )

    # --------------------------------------------------------------
    # Failed images
    # --------------------------------------------------------------

    if failed_images:

        logger.warning("")
        logger.warning(
            "{} image(s) failed during processing.",
            len(failed_images),
        )

        for failure in failed_images:

            if isinstance(
                failure,
                dict,
            ):
                image_name = failure.get(
                    "image",
                    "unknown",
                )

                stage = failure.get(
                    "stage",
                    "unknown",
                )

                reason = failure.get(
                    "reason",
                    "unknown",
                )

                logger.warning(
                    "    {} | Stage: {} | Reason: {}",
                    image_name,
                    stage,
                    reason,
                )

            else:
                logger.warning(
                    "    {}",
                    failure,
                )

    # --------------------------------------------------------------
    # Output directory
    # --------------------------------------------------------------

    if annotation_engine is not None:
        logger.info("")
        logger.info(
            "Final annotation directory: {}",
            annotation_engine.output_dir,
        )

    logger.info(
        "Final SAM 2 segmentation directory: {}",
        config.paths.outputs / "segmentation",
    )
    logger.info(
        "YOLOPv2 auxiliary output directory: {}",
        config.paths.outputs / "yolopv2_segmentation",
    )
    logger.info(
        "FINAL combined output directory: {}",
        config.paths.outputs / "FINAL",
    )

    logger.info("=" * 70)


if __name__ == "__main__":

    main()
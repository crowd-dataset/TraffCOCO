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
Save semantic-verification input
        |
        v
Pipeline Complete

Semantic verification remains a separate process. The annotation output
saved under ``semantic_verification/inputs`` contains the final base
detections and authoritative Locate Anything bounding boxes for that
separate stage.

The pipeline also keeps model lifetimes separated through explicit cleanup
boundaries so large VLMs do not unnecessarily remain resident in GPU memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
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
# Random frame downloader
from annotation_pipeline.pipeline.random_frame_sampler import get_random_frames_from_common_config

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

    final_detections = annotation_engine.annotate_image(
        image_path
    )

    # --------------------------------------------------------------
    # POSTPROCESSING
    # --------------------------------------------------------------

    final_detections = postprocess_detections(
        image_path=image_path,
        detections=final_detections,
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
    """Return the broad semantic group for a detection."""

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

    return ""


def _is_vehicle_or_road_user(
    detection: dict,
) -> bool:
    """Return whether detection is a vehicle or road user."""

    group = _detection_group(detection)

    if group in {
        "vehicle",
        "road_user",
        "road users",
        "road_user_group",
    }:
        return True

    label = _detection_class(detection)
    normalized = label.replace("_", " ").replace("-", " ")

    vehicle_terms = (
        "car",
        "van",
        "bus",
        "truck",
        "motorcycle",
        "motorbike",
        "bicycle",
        "cyclist",
        "pedestrian",
        "vehicle",
        "road user",
        "wheelchair",
    )

    return any(
        term == normalized
        or normalized.startswith(term + " ")
        for term in vehicle_terms
    )


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
    non-road objects are left untouched.
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


def postprocess_detections(
    image_path: Path,
    detections: list[dict],
    min_area_ratio: float = 0.0005,
    max_area_ratio: float = 0.30,
    max_width_ratio: float = 0.90,
    max_height_ratio: float = 0.50,
    tiny_width_ratio: float = 0.02,
    tiny_height_ratio: float = 0.06,
) -> list[dict]:
    """Remove only geometrically implausible detections.

    Small-object filtering is intentionally conservative: a detection is
    removed as tiny only when its area is below ``min_area_ratio`` and both
    dimensions are also below their tiny-object limits. This protects
    legitimate thin or distant objects.

    Oversized detections are removed when their area exceeds
    ``max_area_ratio``, or when both their width and height exceed the
    configured image-relative limits.

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

    # Generic redundancy rule: for the same superclass + same class,
    # retain the larger bbox when the smaller bbox is fully contained.
    filtered_detections = remove_redundant_contained_detections(
        image_path=image_path,
        detections=filtered_detections,
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
        Pipeline Cache
    """

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

    pipeline_start = time.perf_counter()

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

    annotation_engine = AnnotationEngine(
        pipeline_cache_dir=config.paths.pipeline_cache,
        output_dir=config.paths.annotations,
    )


    # ------------------------------------------------------------------
    # Scene Understanding
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Ontology Reasoning
    # ------------------------------------------------------------------

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

                results = ontology_engine.process_images(
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
                    len(results),
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

    # ------------------------------------------------------------------
    # Locate Anything Grounding
    # ------------------------------------------------------------------

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

    # ==============================================================
    # ANNOTATION
    # ==============================================================

    if config.pipeline.run_annotation:

        results: dict[str, dict[str, Any]] = {}
        stage_results: dict[str, dict[str, Any]] = {}
        processed_count = 0

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
                if len(final_detections) > RETRY_THRESHOLD:
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

                # Persist the FINAL postprocessed annotation so the
                # standalone semantic-verification process can reuse the
                # authoritative Locate Anything bbox. No semantic inference
                # is performed by this main process.
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
    # ------------------------------------------------------------------
    # Final stage-resource cleanup
    #
    # This is a defensive boundary for any engine that survived its normal
    # stage-level cleanup. It also makes repeated programmatic execution
    # less likely to retain model references.
    # ------------------------------------------------------------------

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
    pipeline_time = time.perf_counter() - pipeline_start

    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)

    logger.info(
        "Images Processed      : {}",
        total_images,
    )

    logger.info(
        "Objects Processed     : {}",
        total_objects,
    )

    logger.info(
        "Scene Time            : {:.2f} s",
        total_generation_time,
    )

    logger.info(
        "Ontology Time         : {:.2f} s",
        total_reasoning_time,
    )

    logger.info(
        "Total Pipeline Time   : {:.2f} s",
        pipeline_time,
    )

    if total_images:

        logger.info(
            "Average/Image        : {:.2f} s",
            pipeline_time / total_images,
        )

        logger.info(
            "Images/Hour          : {:.2f}",
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
            "Final visualization directory: {}",
            annotation_engine.output_dir,
        )

    logger.info("=" * 70)


if __name__ == "__main__":

    main()
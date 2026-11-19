"""
Shape check for upright objects, using the SAM 2 instance masks.

The scene model often calls a thin orange post a "traffic cone" (a Japanese
test frame, 2026-11-18: five flexible posts on a hatched area), and a
grounding prompt with "lane" in it can put a post box on a lane dash. Both
are visible in the mask:

* a cone narrows towards the top; a post keeps the same width;
* a cone, post or bollard is taller than it is wide; a lane dash is not.

``check_upright_shapes`` runs after SAM 2 and, for each box of an upright
class with a usable mask:

* drops it when the mask is clearly wider than tall;
* relabels a ``traffic_cone`` whose mask does not narrow to
  ``flexible_post`` and keeps ``traffic_cone`` in ``alternative_labels``.

Masks shorter than ``MIN_MASK_HEIGHT_PX`` are left alone: a few rows of
pixels are not enough to measure a taper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from custom_logger import CustomLogger

logger = CustomLogger(__name__)

UPRIGHT_CLASSES = frozenset({
    "traffic_cone",
    "flexible_post",
    "delineator_post",
    "bollard",
})

CONE_CLASS = "traffic_cone"
POST_CLASS = "flexible_post"
POST_LABEL = "flexible post"

MIN_MASK_HEIGHT_PX = 12

# Upright objects are at least this many times as tall as wide.
MIN_HEIGHT_TO_WIDTH = 1.0

# Width near the top (30% of the height from the top) over width lower
# down (65%). A cone tapering linearly to a small top gives about 0.5-0.6;
# a cylinder gives about 1.0.
POST_MIN_TAPER_RATIO = 0.8


def _class_name(detection: dict[str, Any]) -> str:
    for key in ("class_name", "final_label"):
        value = detection.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace(" ", "_")
    return ""


def _load_mask(detection: dict[str, Any]) -> np.ndarray | None:
    segmentation = detection.get("segmentation")
    if not isinstance(segmentation, dict):
        return None
    mask_path = segmentation.get("mask_path")
    if not mask_path or not Path(str(mask_path)).exists():
        return None
    with Image.open(mask_path) as mask_image:
        return np.asarray(mask_image.convert("L")) > 127


def mask_shape(mask: np.ndarray) -> dict[str, float] | None:
    """Height, width and taper of the mask's largest extent."""

    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None

    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1
    width = int(columns[-1] - columns[0] + 1)

    def row_width(fraction: float) -> int:
        return int(mask[top + int(fraction * (height - 1))].sum())

    lower = row_width(0.65)
    return {
        "height": float(height),
        "width": float(width),
        "taper": row_width(0.30) / lower if lower else 0.0,
    }


def check_upright_shapes(
    detections_by_image: dict[str, list[dict[str, Any]]],
    post_class_id: str | None = None,
) -> dict[str, Any]:
    """Drop or relabel upright-object boxes whose mask shape disagrees.

    Changes ``detections_by_image`` in place and returns the counts and
    the names of the images it changed (their SAM overlays need redrawing).
    """

    counts: dict[str, Any] = {"dropped": 0, "relabelled": 0, "changed": []}

    for image_name, detections in detections_by_image.items():
        kept = []
        changed = False
        for detection in detections:
            class_name = _class_name(detection)
            mask = (
                _load_mask(detection)
                if class_name in UPRIGHT_CLASSES
                else None
            )
            shape = mask_shape(mask) if mask is not None else None

            if shape is None or shape["height"] < MIN_MASK_HEIGHT_PX:
                kept.append(detection)
                continue

            detection["shape_check"] = {
                key: round(value, 3) for key, value in shape.items()
            }

            if shape["height"] < MIN_HEIGHT_TO_WIDTH * shape["width"]:
                counts["dropped"] += 1
                changed = True
                logger.info(
                    "Shape check | image='{}' | dropped {} box {} "
                    "(mask {}x{} px is wider than tall)",
                    image_name, class_name, detection.get("bbox"),
                    int(shape["width"]), int(shape["height"]),
                )
                continue

            if (
                class_name == CONE_CLASS
                and shape["taper"] >= POST_MIN_TAPER_RATIO
            ):
                alternatives = list(detection.get("alternative_labels") or [])
                if CONE_CLASS not in alternatives:
                    alternatives.append(CONE_CLASS)
                detection["alternative_labels"] = alternatives
                detection["class_name"] = POST_CLASS
                detection["final_label"] = POST_LABEL
                if post_class_id:
                    detection["class_id"] = post_class_id
                counts["relabelled"] += 1
                changed = True
                logger.info(
                    "Shape check | image='{}' | traffic_cone box {} -> "
                    "flexible_post (mask does not narrow, taper {:.2f})",
                    image_name, detection.get("bbox"), shape["taper"],
                )

            kept.append(detection)

        detections[:] = kept
        if changed:
            counts["changed"].append(image_name)

    return counts


# Two boxes whose SAM masks overlap this much (shared pixels over the
# smaller mask) cover one object. In the 2026-11-18 global test, 35 of 232
# wrong boxes were a second box or a fragment of an object that already
# had one (half a car, a second box on one streetlight or sign).
DUPLICATE_MASK_SHARE = 0.6
MIN_DUPLICATE_MASK_PX = 20


def remove_mask_duplicates(
    detections_by_image: dict[str, list[dict[str, Any]]],
    class_groups: dict[str, str],
) -> dict[str, Any]:
    """Drop the smaller of two same-group boxes whose masks mostly overlap.

    Only boxes in the same ontology super-category are compared, so a rider
    and the motorcycle under them are both kept. Changes
    ``detections_by_image`` in place and returns the counts and the names
    of the images it changed.
    """

    counts: dict[str, Any] = {"dropped": 0, "changed": []}

    for image_name, detections in detections_by_image.items():
        masks = [_load_mask(detection) for detection in detections]
        areas = [int(mask.sum()) if mask is not None else 0 for mask in masks]
        groups = [class_groups.get(_class_name(d)) for d in detections]
        order = sorted(range(len(detections)), key=lambda i: -areas[i])
        drop: set[int] = set()

        for position, i in enumerate(order):
            if i in drop or areas[i] < MIN_DUPLICATE_MASK_PX or not groups[i]:
                continue
            for j in order[position + 1:]:
                if (
                    j in drop
                    or groups[j] != groups[i]
                    or areas[j] < MIN_DUPLICATE_MASK_PX
                    or masks[i].shape != masks[j].shape
                ):
                    continue
                shared = int(np.logical_and(masks[i], masks[j]).sum())
                if shared >= DUPLICATE_MASK_SHARE * areas[j]:
                    drop.add(j)
                    logger.info(
                        "Duplicate check | image='{}' | dropped {} box {} "
                        "({:.0%} of its mask lies on {} box {})",
                        image_name, _class_name(detections[j]),
                        detections[j].get("bbox"), shared / areas[j],
                        _class_name(detections[i]), detections[i].get("bbox"),
                    )

        if drop:
            detections[:] = [d for k, d in enumerate(detections) if k not in drop]
            counts["dropped"] += len(drop)
            counts["changed"].append(image_name)

    return counts

"""
deduplication.py

Remove duplicate final detections of the same physical object.

Several scene objects often ground to the same thing: "black sedan" and
"blue sedan" both become a Locate Anything "car" query, which returns
every car, so each car box is attached to both scene objects. Other
times two different classes claim one box ("car" and "bus shelter" on
the same vehicle). Each copy used to reach SAM 2 and the COCO/YOLO
export as a separate object.

Rules (boxes are [x1, y1, x2, y2]):

1. Any two detections with IoU >= ``same_object_iou`` are one object.
2. Two detections of the same class with IoU >= ``nested_iou`` where
   the smaller box lies almost entirely inside the larger one are one
   object; the tighter box is kept (the larger one usually spans two
   neighbouring objects).

Different boxes of the same class are untouched, so several cars stay
several cars, and a rider overlapping its bicycle stays two objects.

Within a group the kept detection is the one with an ontology class,
then the higher ontology score, scene confidence and grounding
confidence, then the earliest one (the VLM lists the most salient
objects first). Labels of dropped members that differ from the kept
label are stored in ``alternative_labels`` so a later check can review
the conflict.
"""

from __future__ import annotations

from typing import Any

from custom_logger import CustomLogger

logger = CustomLogger(__name__)

SAME_OBJECT_IOU = 0.85
NESTED_IOU = 0.6
NESTED_CONTAINMENT = 0.9


def _box(detection: dict[str, Any]) -> tuple[float, float, float, float] | None:

    bbox = detection.get("bbox")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _area(box: tuple[float, float, float, float]) -> float:

    return (box[2] - box[0]) * (box[3] - box[1])


def _intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:

    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])

    return max(0.0, width) * max(0.0, height)


def _label(detection: dict[str, Any]) -> str:

    return str(
        detection.get("class_name")
        or detection.get("final_label")
        or detection.get("grounding_label")
        or ""
    ).strip().lower()


def _number(value: Any) -> float:

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _rank(detection: dict[str, Any], order: int) -> tuple:
    """Sort key: smaller is better."""

    return (
        0 if detection.get("class_name") else 1,
        -_number(detection.get("ontology_score")),
        -_number(detection.get("scene_confidence")),
        -_number(detection.get("grounding_confidence")),
        order,
    )


def deduplicate_detections(
    detections: list[dict[str, Any]],
    same_object_iou: float = SAME_OBJECT_IOU,
    nested_iou: float = NESTED_IOU,
    nested_containment: float = NESTED_CONTAINMENT,
) -> tuple[list[dict[str, Any]], int]:
    """Return (detections without duplicates, number removed)."""

    if len(detections) < 2:
        return list(detections), 0

    boxes = [_box(d) for d in detections]
    labels = [_label(d) for d in detections]
    count = len(detections)

    # Union-find over "same object" pairs.
    parent = list(range(count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # For nested same-class pairs, remember which member is the larger box.
    larger_in_nested: set[int] = set()

    for i in range(count):
        if boxes[i] is None:
            continue

        for j in range(i + 1, count):
            if boxes[j] is None:
                continue

            inter = _intersection(boxes[i], boxes[j])

            if inter <= 0.0:
                continue

            area_i = _area(boxes[i])
            area_j = _area(boxes[j])
            iou = inter / (area_i + area_j - inter)

            same = iou >= same_object_iou

            if not same and labels[i] and labels[i] == labels[j]:
                smaller = min(area_i, area_j)
                if iou >= nested_iou and inter / smaller >= nested_containment:
                    same = True
                    larger_in_nested.add(i if area_i > area_j else j)

            if same:
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}

    for i in range(count):
        groups.setdefault(find(i), []).append(i)

    kept: list[tuple[int, dict[str, Any]]] = []
    removed = 0

    for members in groups.values():
        if len(members) == 1:
            kept.append((members[0], detections[members[0]]))
            continue

        # A box that only joined because it encloses a tighter box of the
        # same class loses to that tighter box.
        candidates = [m for m in members if m not in larger_in_nested] or members
        best = min(candidates, key=lambda m: _rank(detections[m], m))

        winner = dict(detections[best])
        alternatives = sorted(
            {
                labels[m]
                for m in members
                if m != best and labels[m] and labels[m] != labels[best]
            }
        )

        if alternatives:
            winner["alternative_labels"] = alternatives

        logger.info(
            "Duplicate detections merged: kept '{}' {} over {}",
            labels[best],
            [round(v) for v in boxes[best]] if boxes[best] else None,
            [
                (labels[m], [round(v) for v in boxes[m]] if boxes[m] else None)
                for m in members
                if m != best
            ],
        )

        removed += len(members) - 1
        kept.append((min(members), winner))

    kept.sort(key=lambda item: item[0])

    return [detection for _, detection in kept], removed


def deduplicate_detections_by_image(
    detections_by_image: dict[str, list[dict[str, Any]]],
) -> int:
    """Deduplicate every image in place; return the number removed."""

    total = 0

    for image_name, detections in list(detections_by_image.items()):
        if not isinstance(detections, list):
            continue

        unique, removed = deduplicate_detections(detections)

        if removed:
            logger.info(
                "Removed {} duplicate detection(s) from '{}' ({} -> {}).",
                removed,
                image_name,
                len(detections),
                len(unique),
            )

        detections_by_image[image_name] = unique
        total += removed

    return total

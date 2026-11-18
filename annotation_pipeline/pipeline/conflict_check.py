"""
conflict_check.py

Ask the scene VLM to settle boxes that two descriptions claim.

Grounding runs one text query per scene object, and a query often boxes a
neighbouring object of another class as well. In a Bengaluru test frame
(2026-11-18) "auto-rickshaw" also boxed a motorcyclist, and in another run
"motorcycle with rider" also boxed the auto-rickshaw; which label won
de-duplication depended on scores, not on the pixels. Counting boxes per
description did not help either (it fixed one box and broke another), so
the pixels decide.

A box is in conflict when

* de-duplication merged it with a box of another label (the losing labels
  are in ``merged_labels``), or
* a box of another label overlaps it with IoU >= ``CONFLICT_IOU`` and the
  smaller of the two lies at least ``CONFLICT_CONTAINMENT`` inside the
  larger. A rider and the two-wheeler or pedal vehicle they ride are two
  objects and are not a conflict.

For each conflict box the VLM gets a crop of the frame with the box drawn
in red and picks one of the competing labels or "none of these":

* its own label: kept;
* a label it was merged from: relabelled (old label kept in
  ``alternative_labels``);
* the label of an overlapping box that the VLM confirmed for that box:
  dropped as a duplicate of it;
* anything else (none, no answer, partner not confirmed): unchanged.

Each decision is stored in ``detection["conflict_check"]`` and boxes that
have one are not asked again. Crops are written to
``outputs/conflict_check/crops`` for review.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from custom_logger import CustomLogger

logger = CustomLogger(__name__)

CONFLICT_IOU = 0.5
CONFLICT_CONTAINMENT = 0.9

RIDER_CLASSES = frozenset({"cyclist", "motorcyclist", "delivery_rider"})
RIDDEN_CLASSES = frozenset({
    "motorcycle",
    "scooter",
    "bicycle",
    "boda_boda",
    "habal_habal",
    "cycle_rickshaw",
    "becak",
    "xich_lo",
})

NONE_OPTION = "none of these"
ANSWER_TOKENS = 8
CROP_CONTEXT = 0.75
CROP_MIN_SIDE = 512

PROMPT = (
    "This is part of a dashcam street scene. The red rectangle marks one "
    "object. Which of these best describes the object inside the red "
    "rectangle?\n{options}\n"
    "Judge only the object inside the red rectangle, not objects next to "
    "it. Answer with the letter only."
)


def _label(detection: dict[str, Any]) -> str:

    return str(
        detection.get("class_name") or detection.get("final_label") or ""
    ).strip().lower().replace(" ", "_")


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


def _overlaps(a: tuple, b: tuple) -> bool:

    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return False
    inter = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    iou = inter / (area_a + area_b - inter)
    return iou >= CONFLICT_IOU and inter / min(area_a, area_b) >= CONFLICT_CONTAINMENT


def _rider_pair(a: str, b: str) -> bool:

    return (a in RIDER_CLASSES and b in RIDDEN_CLASSES) or (
        b in RIDER_CLASSES and a in RIDDEN_CLASSES
    )


def find_conflicts(detections: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Return {index: {"candidates": [...], "partners": {label: [index]}}}."""

    labels = [_label(d) for d in detections]
    boxes = [_box(d) for d in detections]
    conflicts: dict[int, dict[str, Any]] = {}

    for i, detection in enumerate(detections):
        if not labels[i] or boxes[i] is None or "conflict_check" in detection:
            continue

        candidates = [labels[i]]
        partners: dict[str, list[int]] = {}

        for merged in detection.get("merged_labels") or []:
            merged = str(merged).strip().lower().replace(" ", "_")
            if merged and merged not in candidates:
                candidates.append(merged)

        for j, other in enumerate(detections):
            if (
                j == i
                or boxes[j] is None
                or not labels[j]
                or labels[j] == labels[i]
                or _rider_pair(labels[i], labels[j])
                or not _overlaps(boxes[i], boxes[j])
            ):
                continue
            partners.setdefault(labels[j], []).append(j)
            if labels[j] not in candidates:
                candidates.append(labels[j])

        if len(candidates) > 1:
            conflicts[i] = {"candidates": candidates, "partners": partners}

    return conflicts


def make_crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
    out_path: Path,
) -> Path:
    """Save a context crop around ``box`` with the box drawn in red."""

    x1, y1, x2, y2 = box
    margin = CROP_CONTEXT * max(x2 - x1, y2 - y1)
    left = max(0, int(x1 - margin))
    top = max(0, int(y1 - margin))
    right = min(image.width, int(x2 + margin))
    bottom = min(image.height, int(y2 + margin))

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    scale = max(1.0, CROP_MIN_SIDE / max(crop.width, crop.height))
    if scale > 1.0:
        crop = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.LANCZOS,
        )

    draw = ImageDraw.Draw(crop)
    width = max(2, round(max(crop.size) / 150))
    draw.rectangle(
        (
            (x1 - left) * scale,
            (y1 - top) * scale,
            (x2 - left) * scale,
            (y2 - top) * scale,
        ),
        outline=(255, 0, 0),
        width=width,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def build_prompt(options: list[str]) -> str:

    lines = [
        f"{letter}) {option.replace('_', ' ')}"
        for letter, option in zip(string.ascii_uppercase, options)
    ]
    return PROMPT.format(options="\n".join(lines))


def parse_answer(response: str, options: list[str]) -> str | None:
    """Map the model's reply to one of ``options`` (None if unclear)."""

    text = (response or "").strip()
    match = re.match(r"^\W*([A-Z])\b", text)
    if match:
        index = string.ascii_uppercase.index(match.group(1))
        if index < len(options):
            return options[index]

    lowered = text.lower()
    found = [
        option for option in options
        if option.replace("_", " ") in lowered
    ]
    return found[0] if len(found) == 1 else None


def _response_text(result: Any) -> str:

    responses = result.get("responses") if isinstance(result, dict) else None
    if not responses:
        return ""
    first = responses[0]
    return first if isinstance(first, str) else str(first)


def decide(
    detections: list[dict[str, Any]],
    conflicts: dict[int, dict[str, Any]],
    answers: dict[int, str | None],
    class_ids: dict[str, str],
) -> list[dict[str, Any]]:
    """Apply the VLM answers; return the detections that are kept."""

    labels = [_label(d) for d in detections]
    drop: set[int] = set()

    for i, conflict in conflicts.items():
        own = labels[i]
        answer = answers.get(i)
        action = "unchanged"

        if answer == own:
            action = "confirmed"
        elif answer and answer in conflict["partners"]:
            confirmed = [
                j for j in conflict["partners"][answer]
                if answers.get(j, labels[j]) == labels[j]
            ]
            if confirmed:
                action = "dropped_duplicate"
                drop.add(i)
        elif answer and answer != NONE_OPTION:
            action = "relabelled"
            detection = detections[i]
            alternatives = [
                a for a in (detection.get("alternative_labels") or [])
                if a != answer
            ]
            if own not in alternatives:
                alternatives.append(own)
            detection["alternative_labels"] = alternatives
            detection["class_name"] = answer
            detection["final_label"] = answer.replace("_", " ")
            if class_ids.get(answer):
                detection["class_id"] = class_ids[answer]

        detections[i]["conflict_check"] = {
            "candidates": conflict["candidates"],
            "answer": answer,
            "action": action,
        }
        logger.info(
            "Conflict check | {} box {} | options {} | answer {} -> {}",
            own,
            [round(v) for v in _box(detections[i]) or ()],
            conflict["candidates"],
            answer,
            action,
        )

    return [d for i, d in enumerate(detections) if i not in drop]


def run_conflict_check(
    config: Any,
    image_paths: list[Path],
    detections_by_image: dict[str, list[dict[str, Any]]],
    class_ids: dict[str, str],
) -> dict[str, int]:
    """Check every conflict box with the scene VLM; change detections in place."""

    targets = {
        image_path: find_conflicts(detections_by_image.get(image_path.name, []))
        for image_path in image_paths
    }
    targets = {path: found for path, found in targets.items() if found}
    total = sum(len(found) for found in targets.values())
    counts = {"checked": 0, "dropped": 0}

    if not total:
        logger.info("Conflict check: no conflicting boxes.")
        return counts

    from annotation_pipeline.models.vlm.model_loader import create_scene_model

    out_dir = Path(config.paths.outputs) / "conflict_check"
    model = create_scene_model(config)
    logger.info("Conflict check: {} box(es) in {} image(s).", total, len(targets))

    try:
        model.load()
        model.max_new_tokens = ANSWER_TOKENS

        for image_path, conflicts in targets.items():
            detections = detections_by_image[image_path.name]
            answers: dict[int, str | None] = {}

            with Image.open(image_path) as image:
                for i, conflict in conflicts.items():
                    options = conflict["candidates"] + [NONE_OPTION]
                    crop_path = make_crop(
                        image,
                        _box(detections[i]),
                        out_dir / "crops" / f"{image_path.stem}_box{i}.png",
                    )
                    try:
                        result = model.infer(
                            image_paths=[crop_path],
                            prompt=build_prompt(options),
                        )
                        answers[i] = parse_answer(_response_text(result), options)
                    except Exception as exc:
                        logger.warning(
                            "Conflict check failed for a box in '{}': {}",
                            image_path.name,
                            exc,
                        )
                        answers[i] = None

            kept = decide(detections, conflicts, answers, class_ids)
            counts["checked"] += len(conflicts)
            counts["dropped"] += len(detections) - len(kept)
            detections_by_image[image_path.name] = kept

            (out_dir / f"{image_path.stem}.json").write_text(
                json.dumps(
                    [
                        {
                            "label": _label(d),
                            "bbox": d.get("bbox"),
                            **d["conflict_check"],
                        }
                        for d in detections
                        if "conflict_check" in d
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        model.unload()

    return counts

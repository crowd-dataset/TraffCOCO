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

A road user or vehicle box is also asked about when its scene object
grounded to more than one box (counting boxes merged away by
de-duplication). Each scene object is one thing the VLM saw, so its extra
boxes are often other objects: in the same frame "auto-rickshaw" also
boxed a motorcycle and "motorcycle with rider" also boxed a small white
car, with no competing box to flag them. The options are its own label
and every other road-user or vehicle label in the frame.

Every cyclist or motorcyclist box is asked too (cyclist, motorcyclist or
pedestrian): the scene VLM called a motorcyclist a "bicycle rider" in the
same frame, and a rider's description has only one box.

Every sign, road-marking, roadside-infrastructure or temporary-object box
is verified as well: the options are its own label, the classes the
ontology lists as commonly confused with it, and "none of these". A box
whose own label Qwen picks in neither option order, answering "none of
these" at least once, is dropped (a sign box on
foliage, a "zebra crossing" on a canal edge, a "construction ahead sign"
on an excavator, a "streetlight" on empty sky in the 2026-11-18 check).
Traffic lights are not verified: small night lights are often right but
hard to judge from a crop. Cones, posts and bollards are not verified
either: shape_check.py settles them from their SAM masks.

Options are always ontology class names; a grounding label that is not a
class (a garbled token such as ``sh<561>``) is never offered.

For each conflict box the VLM gets a crop of the frame with the box drawn
in red and picks one of the competing labels or "none of these":

* its own label: kept;
* another label: relabelled (old label kept in ``alternative_labels``),
  unless a box that already has that label lies at least half inside this
  one: then the VLM was most likely naming that neighbour (a motorcyclist
  riding in front of an SUV) and the box is left unchanged;
* the label of an overlapping box that the VLM confirmed for that box:
  dropped as a duplicate of it;
* anything else (none, no answer, partner not confirmed): unchanged.

An answer that would change a box is asked a second time with the
options in reverse order and only used when both answers agree, because
small VLMs favour options by position.

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

from annotation_pipeline.pipeline.shape_check import UPRIGHT_CLASSES
from custom_logger import CustomLogger

logger = CustomLogger(__name__)

CONFLICT_IOU = 0.5
CONFLICT_CONTAINMENT = 0.9

# Ontology super-categories checked for extra boxes of one scene object.
MOVER_GROUPS = frozenset({"vehicle", "road_user"})

# Ontology super-categories whose boxes are verified one by one.
VERIFY_GROUPS = frozenset({
    "traffic_sign",
    "road_marking",
    "road_infrastructure",
    "temporary_object",
})
MAX_CONFUSIONS = 3

# A box labelled as a traffic sign is often a shop board, an advert, a
# building plaque or the back of a sign (27 of 77 wrong sign boxes in the
# 2026-11-18 global test). With only its own label and "none of these"
# to choose from, the VLM mostly confirmed the label, so sign boxes are
# also offered these look-alike classes and a "not a sign" answer.
SIGN_GROUP = "traffic_sign"
SIGN_LOOKALIKES = ("shop_sign", "billboard")
NOT_A_SIGN = "not_a_sign"
# Without this choice a real sign with the wrong subtype (a stop sign
# boxed as "red route sign") was answered "not a sign" and dropped (8 of
# 22 such drops in the 2026-11-19 sign test). It keeps the box as it is.
OTHER_SIGN = "other_traffic_sign"

# How a class is named in the question when its name reads badly.
OPTION_TEXT = {
    "unreadable_traffic_sign": "traffic sign (text not readable)",
    "shop_sign": "shop or business sign (shop name, logo, phone number)",
    "billboard": "advertisement board or billboard",
    NOT_A_SIGN: "back of a sign, a window or wall, or no sign at all",
    OTHER_SIGN: "a different traffic sign or road post (not the one named)",
}

RIDER_CLASSES = frozenset({"cyclist", "motorcyclist", "delivery_rider"})
# Options for the rider-type question, in this order after the box's own.
RIDER_TYPES = ("cyclist", "motorcyclist", "pedestrian")

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
# Verify answers that reject the box.
DROP_ANSWERS = frozenset({NONE_OPTION, NOT_A_SIGN})
ANSWER_TOKENS = 8
CROP_CONTEXT = 0.75
CROP_MIN_SIDE = 512

PROMPT = (
    "This is part of a dashcam street scene. The red rectangle marks one "
    "object. Which of these best describes the object inside the red "
    "rectangle?\n{options}\n"
    "Judge only the object inside the red rectangle, not objects next to "
    "it. If the rectangle is mainly around a vehicle, name the vehicle, "
    "even if someone is riding it. Answer with the letter only."
)


def _same_class(a: str, b: str) -> bool:
    """Ontology spellings of one class (``guardrail``, ``guard_rail``)."""

    return a.replace("_", "") == b.replace("_", "")


def _inside_fraction(inner: tuple, outer: tuple) -> float:
    """Share of ``inner``'s area that lies inside ``outer``."""

    width = min(inner[2], outer[2]) - max(inner[0], outer[0])
    height = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if width <= 0 or height <= 0:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return width * height / area


NEIGHBOUR_INSIDE = 0.5


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


def _is_mover(label: str, class_groups: dict[str, str]) -> bool:

    return class_groups.get(label) in MOVER_GROUPS


def find_conflicts(
    detections: list[dict[str, Any]],
    class_groups: dict[str, str] | None = None,
    class_confusions: dict[str, list[str]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Return {index: {"kind", "candidates": [...], "partners": {label: [index]}}}.

    ``class_groups`` maps ontology class names to super-categories; without
    it, extra boxes of one scene object are not checked.
    """

    class_groups = class_groups or {}

    def known(label: str) -> bool:
        return not class_groups or label in class_groups

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
            if (
                merged
                and known(merged)
                and not any(_same_class(merged, c) for c in candidates)
            ):
                candidates.append(merged)

        for j, other in enumerate(detections):
            if (
                j == i
                or boxes[j] is None
                or not labels[j]
                or _same_class(labels[j], labels[i])
                or not known(labels[j])
                or _rider_pair(labels[i], labels[j])
                or not _overlaps(boxes[i], boxes[j])
            ):
                continue
            partners.setdefault(labels[j], []).append(j)
            if not any(_same_class(labels[j], c) for c in candidates):
                candidates.append(labels[j])

        if len(candidates) > 1:
            conflicts[i] = {
                "kind": "competing_labels",
                "candidates": candidates,
                "partners": partners,
            }

    # Extra boxes of one scene object (road users and vehicles only).
    boxes_per_object: dict[Any, int] = {}
    for detection in detections:
        for object_id in [detection.get("object_id")] + list(
            detection.get("merged_object_ids") or []
        ):
            if object_id is not None:
                boxes_per_object[object_id] = boxes_per_object.get(object_id, 0) + 1

    mover_labels = sorted(
        {label for label in labels if label and _is_mover(label, class_groups)}
    )

    for i, detection in enumerate(detections):
        if (
            i in conflicts
            or "conflict_check" in detection
            or boxes[i] is None
            or not _is_mover(labels[i], class_groups)
            or boxes_per_object.get(detection.get("object_id"), 0) < 2
        ):
            continue

        candidates = [labels[i]] + [
            m for m in mover_labels if not _same_class(m, labels[i])
        ]
        if len(candidates) > 1:
            conflicts[i] = {
                "kind": "extra_box",
                "candidates": candidates,
                "partners": {},
            }

    # Rider type (cyclist vs motorcyclist) is checked for every rider box.
    for i, detection in enumerate(detections):
        if (
            i in conflicts
            or "conflict_check" in detection
            or boxes[i] is None
            or labels[i] not in {"cyclist", "motorcyclist"}
        ):
            continue
        conflicts[i] = {
            "kind": "rider_type",
            "candidates": [labels[i]] + [
                label for label in RIDER_TYPES if label != labels[i]
            ],
            "partners": {},
        }

    # Signs, markings and roadside objects: verify the label itself.
    for i, detection in enumerate(detections):
        if (
            i in conflicts
            or "conflict_check" in detection
            or boxes[i] is None
            or class_groups.get(labels[i]) not in VERIFY_GROUPS
            or labels[i] in UPRIGHT_CLASSES
        ):
            continue
        confusions = [
            c for c in (class_confusions or {}).get(labels[i], [])
            if known(c) and not _same_class(c, labels[i])
        ][:MAX_CONFUSIONS]
        extra: list[str] = []
        if class_groups.get(labels[i]) == SIGN_GROUP:
            confusions += [
                c for c in SIGN_LOOKALIKES
                if known(c) and c != labels[i] and c not in confusions
            ]
            extra = [OTHER_SIGN, NOT_A_SIGN]
        conflicts[i] = {
            "kind": "verify",
            "candidates": [labels[i]] + confusions,
            "extra": extra,
            "partners": {},
        }

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


def _option_text(option: str) -> str:

    return OPTION_TEXT.get(option, option.replace("_", " "))


def build_prompt(options: list[str]) -> str:

    lines = [
        f"{letter}) {_option_text(option)}"
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
        if _option_text(option) in lowered
    ]
    return found[0] if len(found) == 1 else None


def _response_text(result: Any) -> str:

    responses = result.get("responses") if isinstance(result, dict) else None
    if not responses:
        return ""
    first = responses[0]
    return first if isinstance(first, str) else str(first)


def _ask(model: Any, crop_path: Path, options: list[str]) -> str | None:

    result = model.infer(image_paths=[crop_path], prompt=build_prompt(options))
    return parse_answer(_response_text(result), options)


def _consistent_answer(
    model: Any,
    crop_path: Path,
    candidates: list[str],
    kind: str | None = None,
    extra: list[str] | None = None,
) -> tuple[str | None, list[str | None]]:
    """Ask once; if the answer would change the box, ask again with the
    options reversed and keep the answer only when both agree.

    Small VLMs favour an option by its position (a clear motorcycle was
    answered "auto rickshaw" as option B in a Bengaluru test, 2026-11-18),
    so a change needs the same answer from both orders. A confirmation of
    the box's own label (the first candidate) costs one question.
    """

    options = candidates + list(extra or []) + [NONE_OPTION]
    first = _ask(model, crop_path, options)
    if first is None or first == candidates[0]:
        return first, [first]

    second = _ask(model, crop_path, list(reversed(options)))
    if second == first:
        return first, [first, second]
    # Verify: the box's own label was picked in neither order and "none"
    # (or "not a sign") in at least one, so the label is rejected (an
    # excavator boxed as a "construction ahead sign": none, then roadworks
    # sign).
    if (
        kind == "verify"
        and DROP_ANSWERS & {first, second}
        and candidates[0] not in (first, second)
        and OTHER_SIGN not in (first, second)
    ):
        return NONE_OPTION, [first, second]
    return None, [first, second]


def decide(
    detections: list[dict[str, Any]],
    conflicts: dict[int, dict[str, Any]],
    answers: dict[int, str | None],
    class_ids: dict[str, str],
) -> list[dict[str, Any]]:
    """Apply the VLM answers; return the detections that are kept."""

    labels = [_label(d) for d in detections]
    boxes = [_box(d) for d in detections]
    drop: set[int] = set()

    # Labels after this round's answers, for the neighbour test: a box
    # relabelled in the same round (an "auto-rickshaw" box that is really
    # the motorcycle in front of an SUV) counts with its new label.
    settled = list(labels)
    for j, answer in answers.items():
        if answer and answer not in DROP_ANSWERS | {OTHER_SIGN} and j in conflicts:
            settled[j] = answer

    for i, conflict in conflicts.items():
        own = labels[i]
        answer = answers.get(i)
        action = "unchanged"

        if answer == own:
            action = "confirmed"
        elif answer == OTHER_SIGN:
            action = "unchanged_other_sign"
        elif answer in DROP_ANSWERS and conflict.get("kind") == "verify":
            action = "dropped_not_seen"
            drop.add(i)
        elif answer and answer in conflict["partners"]:
            # A partner box counts as confirmed unless the VLM gave it a
            # different label (an unsure answer does not contradict it).
            confirmed = [
                j for j in conflict["partners"][answer]
                if answers.get(j) in (labels[j], None)
            ]
            if confirmed:
                action = "dropped_duplicate"
                drop.add(i)
        elif answer and answer not in DROP_ANSWERS and any(
            j != i
            and settled[j] == answer
            and boxes[j] is not None
            and boxes[i] is not None
            and _inside_fraction(boxes[j], boxes[i]) >= NEIGHBOUR_INSIDE
            for j in range(len(detections))
        ):
            action = "unchanged_neighbour_answer"
        elif answer and answer not in DROP_ANSWERS:
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
            "kind": conflict.get("kind"),
            "candidates": conflict["candidates"],
            "answer": answer,
            "action": action,
        }
        logger.info(
            "Conflict check | {} | {} box {} | options {} | answer {} -> {}",
            conflict.get("kind"),
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
    class_groups: dict[str, str] | None = None,
    class_confusions: dict[str, list[str]] | None = None,
) -> dict[str, int]:
    """Check every conflict box with the scene VLM; change detections in place."""

    targets = {
        image_path: find_conflicts(
            detections_by_image.get(image_path.name, []),
            class_groups,
            class_confusions,
        )
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
            raw: dict[int, list[str | None]] = {}

            with Image.open(image_path) as image:
                for i, conflict in conflicts.items():
                    crop_path = make_crop(
                        image,
                        _box(detections[i]),
                        out_dir / "crops" / f"{image_path.stem}_box{i}.png",
                    )
                    try:
                        answers[i], raw[i] = _consistent_answer(
                            model,
                            crop_path,
                            conflict["candidates"],
                            conflict.get("kind"),
                            conflict.get("extra"),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Conflict check failed for a box in '{}': {}",
                            image_path.name,
                            exc,
                        )
                        answers[i] = None

            kept = decide(detections, conflicts, answers, class_ids)
            for i, replies in raw.items():
                detections[i]["conflict_check"]["replies"] = replies
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


# A rider's class follows the vehicle they ride. The scene VLM sometimes
# calls a motorcyclist a "bicycle rider" (Bengaluru test frame,
# 2026-11-18), and that description has only one box, so nothing above
# checks it; the motorcycle under the rider is checked, so it decides.
RIDER_FOR_VEHICLE = {
    "motorcycle": "motorcyclist",
    "scooter": "motorcyclist",
    "boda_boda": "motorcyclist",
    "habal_habal": "motorcyclist",
    "bicycle": "cyclist",
}
RIDER_OVERLAP = 0.4


def align_rider_labels(
    detections_by_image: dict[str, list[dict[str, Any]]],
    class_ids: dict[str, str],
) -> int:
    """Relabel cyclist/motorcyclist boxes to match the vehicle under them.

    A rider box matches a vehicle box when their overlap is at least
    ``RIDER_OVERLAP`` of the smaller box and the vehicle's centre is lower
    than the rider's. With several matches the largest overlap wins.
    Changes detections in place and returns the number relabelled.
    """

    changed = 0

    for image_name, detections in detections_by_image.items():
        labels = [_label(d) for d in detections]
        boxes = [_box(d) for d in detections]

        for i, detection in enumerate(detections):
            if labels[i] not in {"cyclist", "motorcyclist"} or boxes[i] is None:
                continue
            rider = boxes[i]
            best, best_overlap = None, 0.0

            for j, vehicle in enumerate(boxes):
                if labels[j] not in RIDER_FOR_VEHICLE or vehicle is None:
                    continue
                width = min(rider[2], vehicle[2]) - max(rider[0], vehicle[0])
                height = min(rider[3], vehicle[3]) - max(rider[1], vehicle[1])
                if width <= 0 or height <= 0:
                    continue
                smaller = min(
                    (rider[2] - rider[0]) * (rider[3] - rider[1]),
                    (vehicle[2] - vehicle[0]) * (vehicle[3] - vehicle[1]),
                )
                overlap = width * height / smaller
                below = (vehicle[1] + vehicle[3]) > (rider[1] + rider[3])
                if overlap >= RIDER_OVERLAP and below and overlap > best_overlap:
                    best, best_overlap = j, overlap

            if best is None:
                continue
            rider_label = RIDER_FOR_VEHICLE[labels[best]]
            if rider_label == labels[i]:
                continue

            alternatives = list(detection.get("alternative_labels") or [])
            if labels[i] not in alternatives:
                alternatives.append(labels[i])
            detection["alternative_labels"] = alternatives
            detection["class_name"] = rider_label
            detection["final_label"] = rider_label.replace("_", " ")
            if class_ids.get(rider_label):
                detection["class_id"] = class_ids[rider_label]
            detection["rider_of"] = labels[best]
            labels[i] = rider_label
            changed += 1
            logger.info(
                "Rider label | image='{}' | box {} -> {} (rides the {} at {})",
                image_name,
                [round(v) for v in rider],
                rider_label,
                labels[best],
                [round(v) for v in boxes[best]],
            )

    return changed

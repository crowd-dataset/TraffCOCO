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
car, with no competing box to flag them. The options are its own label,
every other road-user or vehicle label in the frame, the common classes
of its kind, "several objects" and "not a vehicle or person". Every other
vehicle and road-user box gets the same question without the frame's
labels.

The box's own label is listed first, and small VLMs favour the first
option, so a vehicle or person box is asked in both orders even when the
first answer keeps its label. When the two answers differ and one of them
is its own rare subtype (a hatchback boxed as "kei car" or "pickup truck"
in the 2026-11-19 class test), the common class from the other order is
used.

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
Every vehicle traffic-light box is asked which lamp is lit, or whether
it is a pedestrian signal or no traffic light at all, in both option
orders (see ``LIGHT_OPTIONS``). Every sign box that is kept is then
described by the VLM, and the sign classes nearest that description are
offered as its type (see ``check_sign_types``). Cones, posts and bollards
are not verified: shape_check.py settles them from their SAM masks.

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

A box from the street-scene detectors (see detector_fusion.py) keeps its
detector group: the VLM may give it a finer class of the same group (a
"car" box may become a taxi, a traffic light may get its state or turn out
to be a pedestrian signal), but an
answer of another group (a crosswalk called a stop line, a person called a
car) leaves the box unchanged, and a "not seen" answer does not drop it
when the detector was at least ``DETECTOR_KEEP_CONFIDENCE`` sure. On 160
held-out Mapillary Vistas images (2026-11-23) the check had dropped or
moved many real detector boxes (sign recall fell from 56% to 40%, crosswalks
from 56% to 26%); with this rule F1 rose from 0.510 to 0.541 (precision
70.6% -> 69.8%, recall 39.9% -> 44.2%), and from 0.501 to 0.533 on the 40
dev images. Restoring boxes dropped as duplicates did not help.

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

from annotation_pipeline.pipeline.detector_fusion import fusion_group
from annotation_pipeline.pipeline.shape_check import UPRIGHT_CLASSES
from custom_logger import CustomLogger

logger = CustomLogger(__name__)

CONFLICT_IOU = 0.5
# A detector box at least this confident is not dropped on a "not seen"
# answer (0.5 and 0.7 gave the same held-out F1; 0.7 kept more precision).
DETECTOR_KEEP_CONFIDENCE = 0.7
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
# A real sign with the wrong subtype (a stop sign boxed as "red route
# sign") was answered "not a sign" and dropped (8 of 22 such drops in the
# first 2026-11-19 sign test). A "different traffic sign" answer that kept
# the box was rarely picked: in the class tests real direction, parking and
# warning signs were still answered "not a sign" or "shop sign" (5 of 9
# wrong drops), with the 8B and the 32B model alike. So every sign box is
# also offered the general class, named by the kinds of sign it covers; it
# is the ontology's class for a sign whose exact type is not known. Asked
# on the 35 reviewed sign boxes of both class tests, it dropped 4 more
# name boards but saved no real sign: the general class picked in one
# order only kept a real parking sign and two non-sign boxes alike, so
# such a split still drops the box.
GENERAL_SIGN = "unreadable_traffic_sign"

# How a class is named in the question when its name reads badly.
OPTION_TEXT = {
    GENERAL_SIGN: (
        "a traffic sign of another type, or whose type cannot be read "
        "(warning, regulatory, direction or place names, parking, information)"
    ),
    "shop_sign": "shop or business sign (shop name, logo, phone number)",
    "billboard": "advertisement board or billboard",
    NOT_A_SIGN: "back of a sign, a window or wall, or no sign at all",
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

# Vehicle traffic lights: the state comes from the scene description of
# the light and is shared by every box grounded to it, so a red light
# ahead and a green one across the junction got the same label (light
# boxes 51% right in the 2026-11-18 global test). Each box is asked which
# lamp is lit; the stateless class is chosen when no lit lamp shows.
LIGHT_STATE_CLASSES = (
    "traffic_light_red",
    "traffic_light_yellow",
    "traffic_light_green",
)
LIGHT_NO_STATE = "traffic_light_3_phase"
# In the 2026-11-20 global run (3rd) 22 of 79 light boxes were wrong: a
# pedestrian signal (orange hand) called a vehicle light, the back or side
# of a head given a state or an arrow, and shop lights, barrels and
# building fronts called lights. The state question had no answer for
# these, and the box's own label, offered first, was taken after one
# question. So the options now come in a fixed order (not the box's own
# label first), add a pedestrian signal and "not a traffic light", and
# every light box is asked in both orders.
PEDESTRIAN_SIGNAL = "pedestrian_signal"
NOT_A_LIGHT = "not_a_traffic_light"
LIGHT_OPTIONS = LIGHT_STATE_CLASSES + (LIGHT_NO_STATE, PEDESTRIAN_SIGNAL, NOT_A_LIGHT)
LIGHT_CLASSES = frozenset(LIGHT_STATE_CLASSES + (
    LIGHT_NO_STATE,
    "traffic_light_off",
    "traffic_light_2_phase",
    "traffic_light_arrow_left",
    "traffic_light_arrow_right",
    "traffic_light_arrow_straight",
))
LIGHT_QUESTION_TEXT = {
    "traffic_light_red": "traffic light with the RED lamp lit",
    "traffic_light_yellow": "traffic light with the YELLOW (amber) lamp lit",
    "traffic_light_green": "traffic light with the GREEN lamp lit",
    LIGHT_NO_STATE: "traffic light with no lit lamp visible (seen from the back or side, or dark)",
    "traffic_light_off": "traffic light with no lit lamp visible (seen from the back or side, or dark)",
    PEDESTRIAN_SIGNAL: "pedestrian signal (lit walking figure or hand)",
    NOT_A_LIGHT: "not a traffic light (a street lamp, sign, shop light, vehicle or building)",
}

# Every vehicle and road-user box is checked against common classes of
# its kind. In the 2026-11-18 global test, 53 of 232 wrong boxes were
# vehicles or people with the wrong class (a trailer as "suv", an ordinary
# hatchback as "kei car", a rider as "pedestrian") or on no vehicle at
# all (a window, a fence, a poster); a general class such as "car" is
# right for the rare subtypes.
MOVER_OPTIONS = {
    "vehicle": (
        "car", "suv", "van", "pickup_truck", "truck", "city_bus",
        "motorcycle", "bicycle", "pedestrian",
    ),
    "road_user": (
        "pedestrian", "cyclist", "motorcyclist", "car", "motorcycle", "bicycle",
    ),
}
NOT_A_MOVER = "not_a_vehicle_or_person"
# One box over a row of cars, two streetlights or two stacked signs (22
# of 232 wrong boxes) is dropped.
SEVERAL_OBJECTS = "several_objects"
STREETLIGHT = "streetlight"
NOT_A_LAMP = "pole_without_lamp"

OPTION_TEXT.update({
    "suv": "SUV",
    "city_bus": "bus",
    NOT_A_MOVER: "not a vehicle or person (a building, sign, picture, pole, plant or empty road)",
    SEVERAL_OBJECTS: "two or more separate objects (the box covers several)",
    NOT_A_LAMP: "a pole without a lamp, a tree or a building",
})

NONE_OPTION = "none of these"
# Verify answers that reject the box.
DROP_ANSWERS = frozenset({
    NONE_OPTION, NOT_A_SIGN, NOT_A_MOVER, SEVERAL_OBJECTS, NOT_A_LAMP, NOT_A_LIGHT,
})
# Vehicle and road-user questions; their boxes are asked in both orders.
MOVER_KINDS = frozenset({"extra_box", "mover_verify"})
COMMON_MOVERS = frozenset(c for options in MOVER_OPTIONS.values() for c in options)
VERIFY_KINDS = frozenset({"verify"}) | MOVER_KINDS
ANSWER_TOKENS = 8
CROP_CONTEXT = 0.75
# A sign crop shows at least this many frame pixels across. With the
# margin alone, a 20 px sign became a 35 px crop blown up to 512 px, and
# the scene VLM answered "not a sign" for real direction, warning and
# regulatory signs (2026-11-19 class tests) that the pole and the road
# around them make plain. On the 35 reviewed sign boxes of those tests a
# 200 px window kept a real parking sign and changed nothing else; 400 px
# saved none and dropped two right boxes.
SIGN_MIN_WINDOW = 200
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
        # The frame's labels alone kept wrong subtypes (an ordinary
        # hatchback confirmed as "kei car" when only kei car, van and
        # pickup were offered; 14 of 15 wrong vehicle types in the
        # 2026-11-19 class test came through this question).
        candidates += [
            c for c in MOVER_OPTIONS.get(class_groups.get(labels[i]), ())
            if known(c) and not any(_same_class(c, k) for k in candidates)
        ]
        if len(candidates) > 1:
            conflicts[i] = {
                "kind": "extra_box",
                "candidates": candidates,
                "extra": [SEVERAL_OBJECTS, NOT_A_MOVER],
                "partners": {},
            }

    # Traffic-light state is checked for every vehicle traffic-light box.
    for i, detection in enumerate(detections):
        if (
            i in conflicts
            or "conflict_check" in detection
            or boxes[i] is None
            or labels[i] not in LIGHT_CLASSES
        ):
            continue
        conflicts[i] = {
            "kind": "light_state",
            "candidates": [labels[i]] + [
                c for c in LIGHT_OPTIONS
                if c != labels[i] and (known(c) or c == NOT_A_LIGHT)
            ],
            "partners": {},
        }

    # Every other vehicle and road-user box: its class, or no single mover.
    for i, detection in enumerate(detections):
        group = class_groups.get(labels[i])
        if (
            i in conflicts
            or "conflict_check" in detection
            or boxes[i] is None
            or group not in MOVER_OPTIONS
            or labels[i] in RIDER_CLASSES
        ):
            continue
        conflicts[i] = {
            "kind": "mover_verify",
            "candidates": [labels[i]] + [
                c for c in MOVER_OPTIONS[group]
                if c != labels[i] and known(c)
            ],
            "extra": [SEVERAL_OBJECTS, NOT_A_MOVER],
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
        extra: list[str] = [SEVERAL_OBJECTS]
        min_window = 0
        if class_groups.get(labels[i]) == SIGN_GROUP:
            confusions += [
                c for c in SIGN_LOOKALIKES + (GENERAL_SIGN,)
                if known(c) and c != labels[i] and c not in confusions
            ]
            extra = [NOT_A_SIGN, SEVERAL_OBJECTS]
            min_window = SIGN_MIN_WINDOW
        elif labels[i] == STREETLIGHT:
            extra = [NOT_A_LAMP, SEVERAL_OBJECTS]
        conflicts[i] = {
            "kind": "verify",
            "candidates": [labels[i]] + confusions,
            "extra": extra,
            "min_window": min_window,
            "partners": {},
        }

    return conflicts


def make_crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
    out_path: Path,
    min_window: float = 0,
) -> Path:
    """Save a context crop around ``box`` with the box drawn in red.

    The crop is at least ``min_window`` frame pixels wide and high (less
    at the frame edge).
    """

    x1, y1, x2, y2 = box
    margin = CROP_CONTEXT * max(x2 - x1, y2 - y1)
    margin_x = max(margin, (min_window - (x2 - x1)) / 2)
    margin_y = max(margin, (min_window - (y2 - y1)) / 2)
    left = max(0, int(x1 - margin_x))
    top = max(0, int(y1 - margin_y))
    right = min(image.width, int(x2 + margin_x))
    bottom = min(image.height, int(y2 + margin_y))

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


def _option_text(option: str, kind: str | None = None) -> str:

    if kind == "light_state" and option in LIGHT_QUESTION_TEXT:
        return LIGHT_QUESTION_TEXT[option]
    return OPTION_TEXT.get(option, option.replace("_", " "))


def build_prompt(options: list[str], kind: str | None = None) -> str:

    lines = [
        f"{letter}) {_option_text(option, kind)}"
        for letter, option in zip(string.ascii_uppercase, options)
    ]
    return PROMPT.format(options="\n".join(lines))


def parse_answer(
    response: str,
    options: list[str],
    kind: str | None = None,
) -> str | None:
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
        if _option_text(option, kind).lower() in lowered
    ]
    return found[0] if len(found) == 1 else None


def _response_text(result: Any) -> str:

    responses = result.get("responses") if isinstance(result, dict) else None
    if not responses:
        return ""
    first = responses[0]
    return first if isinstance(first, str) else str(first)


def _ask(
    model: Any,
    crop_path: Path,
    options: list[str],
    kind: str | None = None,
) -> str | None:

    result = model.infer(image_paths=[crop_path], prompt=build_prompt(options, kind))
    return parse_answer(_response_text(result), options, kind)


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

    if kind == "light_state":
        return _light_answer(model, crop_path, candidates)

    options = candidates + list(extra or []) + [NONE_OPTION]
    first = _ask(model, crop_path, options, kind)
    if first is None or (first == candidates[0] and kind not in MOVER_KINDS):
        return first, [first]

    second = _ask(model, crop_path, list(reversed(options)), kind)
    if second == first:
        return first, [first, second]
    # A rare vehicle subtype picked in one order only gives way to a
    # common class picked in the other.
    if (
        kind in MOVER_KINDS
        and candidates[0] in (first, second)
        and candidates[0] not in COMMON_MOVERS
    ):
        other = second if first == candidates[0] else first
        if other in COMMON_MOVERS:
            return other, [first, second]
    # Verify: the box's own label was picked in neither order and "none"
    # (or "not a sign") in at least one, so the label is rejected (an
    # excavator boxed as a "construction ahead sign": none, then roadworks
    # sign).
    if (
        kind in VERIFY_KINDS
        and DROP_ANSWERS & {first, second}
        and candidates[0] not in (first, second)
        # A vehicle or person is dropped only when both orders reject it.
        and (kind not in MOVER_KINDS or {first, second} <= DROP_ANSWERS)
    ):
        return NONE_OPTION, [first, second]
    return None, [first, second]


VEHICLE_LIGHTS = frozenset(LIGHT_STATE_CLASSES + (LIGHT_NO_STATE,))

# Sign type. The checks above only offer a sign box its own label, the
# classes the ontology lists as confusable with it and the general sign
# class, so a sign named with the wrong type keeps it (a keep-right sign
# as "roundabout sign", a yield sign as "traffic signal ahead sign") and
# readable speed-limit, stop and priority signs stay "unreadable" (11 of
# 30 wrong sign boxes in the 2026-11-20 global run, 3rd). Each kept sign
# box is now described by the scene VLM in its own words, the sign
# classes nearest that description are found by embedding, and the VLM
# picks among the box's label and those classes in both option orders;
# the box is relabelled only when both orders pick the same other class.
SIGN_DESCRIBE_PROMPT = (
    "This is part of a dashcam street scene. The red rectangle marks a "
    "traffic sign. Describe that sign in one sentence: its shape, its "
    "colours, and the symbol, number or words on it."
)
SIGN_DESCRIBE_TOKENS = 48
SIGN_TYPE_CANDIDATES = 4


def _sign_option_text(option: str, hints: dict[str, str]) -> str:

    text = _option_text(option)
    hint = hints.get(option)
    return f"{text} ({hint})" if hint else text


def _ask_sign_type(
    model: Any,
    crop_path: Path,
    options: list[str],
    hints: dict[str, str],
) -> str | None:

    lines = [
        f"{letter}) {_sign_option_text(option, hints)}"
        for letter, option in zip(string.ascii_uppercase, options)
    ]
    result = model.infer(
        image_paths=[crop_path],
        prompt=PROMPT.format(options="\n".join(lines)),
    )
    return parse_answer(_response_text(result), options)


def nearest_sign_classes(
    embedder: Any,
    description: str,
    names: list[str],
    vectors: Any,
    exclude: set[str],
    k: int = SIGN_TYPE_CANDIDATES,
) -> list[str]:
    """The ``k`` sign classes whose ontology text is nearest
    ``description`` (vectors are normalised, so a dot product ranks)."""

    query = embedder.encode(description)
    scores = vectors @ query
    ranked = [names[j] for j in scores.argsort()[::-1]]
    return [name for name in ranked if name not in exclude][:k]


def _sign_type_answer(
    model: Any,
    crop_path: Path,
    own: str,
    nearest: list[str],
    hints: dict[str, str],
) -> tuple[str | None, list[str | None]]:
    """Own label, the nearest classes and the general class, asked in
    both orders; a change needs the same answer from both."""

    options = [own] + [c for c in nearest if c != own]
    if own != GENERAL_SIGN:
        options.append(GENERAL_SIGN)
    first = _ask_sign_type(model, crop_path, options, hints)
    second = _ask_sign_type(model, crop_path, list(reversed(options)), hints)
    if first == second and first not in (None, own):
        return first, [first, second]
    return None, [first, second]


def _sign_index(
    config: Any,
    sign_classes: dict[str, dict[str, str]],
) -> tuple[Any, list[str], Any]:
    """Embedding model, sign class names and their normalised vectors."""

    from annotation_pipeline.models.ontology.embedding_model import EmbeddingModel

    embedder = EmbeddingModel(
        model_name=config.models.embedding_model_id,
        device=config.models.embedding_device,
        normalize_embeddings=True,
    )
    names = list(sign_classes)
    vectors = embedder.encode_batch([sign_classes[n].get("text", n) for n in names])
    return embedder, names, vectors


def check_sign_types(
    model: Any,
    embedder: Any,
    image: Image.Image,
    image_stem: str,
    detections: list[dict[str, Any]],
    class_groups: dict[str, str],
    sign_classes: dict[str, dict[str, str]],
    class_ids: dict[str, str],
    out_dir: Path,
    vectors: Any,
) -> int:
    """Ask the type of every sign box; relabel in place. Returns the
    number of boxes relabelled."""

    names = list(sign_classes)
    hints = {
        name: entry.get("hint", "")
        for name, entry in sign_classes.items()
    }
    changed = 0

    for i, detection in enumerate(detections):
        own = _label(detection)
        box = _box(detection)
        if (
            box is None
            or "sign_type" in detection
            or class_groups.get(own) != SIGN_GROUP
            or own not in sign_classes
        ):
            continue

        crop_path = make_crop(
            image,
            box,
            out_dir / "crops" / f"{image_stem}_sign{i}.png",
            SIGN_MIN_WINDOW,
        )
        model.max_new_tokens = SIGN_DESCRIBE_TOKENS
        try:
            description = _response_text(
                model.infer(image_paths=[crop_path], prompt=SIGN_DESCRIBE_PROMPT)
            ).strip()
        finally:
            model.max_new_tokens = ANSWER_TOKENS
        if not description:
            continue

        nearest = nearest_sign_classes(
            embedder, description, names, vectors, {own, GENERAL_SIGN},
        )
        answer, replies = _sign_type_answer(model, crop_path, own, nearest, hints)
        detection["sign_type"] = {
            "label_before": own,
            "description": description,
            "candidates": nearest,
            "replies": replies,
            "answer": answer,
        }
        if answer:
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
            changed += 1
        logger.info(
            "Sign type | {} box {} | '{}' | options {} | replies {} -> {}",
            own,
            [round(v) for v in box],
            description,
            nearest,
            replies,
            answer or "unchanged",
        )

    return changed


def _light_answer(
    model: Any,
    crop_path: Path,
    candidates: list[str],
) -> tuple[str | None, list[str | None]]:
    """Ask a traffic-light box in both orders of the fixed option list.

    Both orders agree: that answer. Otherwise the box's own label picked
    in one order keeps it; two different vehicle-light answers mean the
    state cannot be told, so the box gets the stateless class.
    """

    own = candidates[0]
    order = [c for c in LIGHT_OPTIONS if c in candidates]
    order += [c for c in candidates if c not in order]
    first = _ask(model, crop_path, order, "light_state")
    second = _ask(model, crop_path, list(reversed(order)), "light_state")
    replies = [first, second]
    if own == "traffic_light_off":
        # The stateless answer confirms a light that is off.
        replies_seen = [own if r == LIGHT_NO_STATE else r for r in replies]
    else:
        replies_seen = replies
    a, b = replies_seen
    if a is None or b is None:
        return None, replies
    if a == b:
        return a, replies
    if own in (a, b):
        return own, replies
    if {a, b} <= VEHICLE_LIGHTS:
        return LIGHT_NO_STATE, replies
    return None, replies


def decide(
    detections: list[dict[str, Any]],
    conflicts: dict[int, dict[str, Any]],
    answers: dict[int, str | None],
    class_ids: dict[str, str],
    class_groups: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Apply the VLM answers; return the detections that are kept."""

    labels = [_label(d) for d in detections]
    boxes = [_box(d) for d in detections]
    drop: set[int] = set()
    class_groups = class_groups or {}

    def leaves_detector_group(i: int, answer: str | None) -> bool:
        """True when an answer would move a detector box to another group."""

        if not detections[i].get("detector") or not answer or answer in DROP_ANSWERS:
            return False
        if class_groups.get(answer) == class_groups.get(labels[i]) == "traffic_signal":
            return False  # a detector "traffic light" may be a pedestrian signal
        return fusion_group(answer, class_groups.get(answer)) != fusion_group(
            labels[i], class_groups.get(labels[i])
        )

    def detector_sure(i: int) -> bool:

        try:
            confidence = float(detections[i].get("detector_confidence") or 0)
        except (TypeError, ValueError):
            return False
        return bool(detections[i].get("detector")) and confidence >= DETECTOR_KEEP_CONFIDENCE

    # Labels after this round's answers, for the neighbour test: a box
    # relabelled in the same round (an "auto-rickshaw" box that is really
    # the motorcycle in front of an SUV) counts with its new label.
    settled = list(labels)
    for j, answer in answers.items():
        if (
            answer and answer not in DROP_ANSWERS and j in conflicts
            and not leaves_detector_group(j, answer)
        ):
            settled[j] = answer

    for i, conflict in conflicts.items():
        own = labels[i]
        answer = answers.get(i)
        action = "unchanged"

        if answer == own:
            action = "confirmed"
        elif (
            answer in DROP_ANSWERS
            and conflict.get("kind") in VERIFY_KINDS | {"light_state"}
            and detector_sure(i)
        ):
            action = "kept_detector_box"
        elif answer in DROP_ANSWERS and conflict.get("kind") in VERIFY_KINDS | {"light_state"}:
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
        elif leaves_detector_group(i, answer):
            action = "kept_detector_group"
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
            "label_before": own,
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
    sign_classes: dict[str, dict[str, str]] | None = None,
) -> dict[str, int]:
    """Check every conflict box with the scene VLM; change detections in place.

    ``sign_classes`` maps each traffic-sign class to its ontology
    ``text`` (for the embedding search) and a short visual ``hint``; when
    given, every kept sign box is also asked its type.
    """

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
    counts["sign_type_changed"] = 0
    embedder = None
    sign_names: list[str] = []
    sign_vectors = None
    class_groups = class_groups or {}
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
                        conflict.get("min_window", 0),
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

            kept = decide(detections, conflicts, answers, class_ids, class_groups)
            for i, replies in raw.items():
                detections[i]["conflict_check"]["replies"] = replies
            counts["checked"] += len(conflicts)
            counts["dropped"] += len(detections) - len(kept)
            detections_by_image[image_path.name] = kept

            if sign_classes and any(
                class_groups.get(_label(d)) == SIGN_GROUP for d in kept
            ):
                if embedder is None:
                    embedder, sign_names, sign_vectors = _sign_index(
                        config, sign_classes,
                    )
                try:
                    with Image.open(image_path) as image:
                        counts["sign_type_changed"] += check_sign_types(
                            model,
                            embedder,
                            image,
                            image_path.stem,
                            kept,
                            class_groups,
                            {n: sign_classes[n] for n in sign_names},
                            class_ids,
                            out_dir,
                            sign_vectors,
                        )
                except Exception as exc:
                    logger.warning(
                        "Sign type check failed in '{}': {}",
                        image_path.name,
                        exc,
                    )

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

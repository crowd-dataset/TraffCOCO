"""
detector_fusion.py

Merge detector proposals (see models/detection/detector_proposals.py) with
the boxes of the VLM -> LocateAnything chain.

Classes are compared by coarse group (car, person, traffic light, traffic
sign, ...), since a detector says "car" where the scene VLM may say "taxi"
or "kei car". For every group the detectors cover:

* a grounding box that matches a proposal of its group (IoU >= 0.5) keeps
  its own, finer label and takes the detector's box;
* a proposal that matches no grounding box is added with its general
  class;
* in ``replace`` mode a grounding box of the group that matches no
  proposal is dropped; in ``add`` mode it is kept.

Replace mode gave the best F1 on held-out Mapillary Vistas images
(2026-11-22): the grounding boxes the detectors do not confirm were mostly
wrong. Groups the detectors do not cover (pedestrian signals, cones,
streetlights, road markings, rare vehicles, ...) are left as they are.

A person or rider box lying at least ``BILLBOARD_INSIDE`` inside a
billboard or banner from the panoptic model is dropped: on the 200 Vistas
images most confident false person boxes were printed people on
billboards and posters (33 of 39 such boxes were false).

Streetlight boxes come from grounding only. When the panoptic model also
looked for lamp heads (Vistas "Street Light", the lamp head only), a
streetlight box is dropped if no lamp head lies at least ``LAMP_INSIDE``
inside it, or if it is less than ``LAMP_MIN_ASPECT`` times as tall as wide
(a lamp head alone or a lamp over a billboard). On the 100 reviewed CROWD
frames (2026-11-25) 32 of 103 streetlight boxes were wrong: part of a pole
with the lamp outside the box, trees, several poles in one box, a lamp head
alone. The check dropped 31 of the wrong boxes and 26 of the 67 right ones
(mostly distant lamps whose heads the panoptic model missed), so the
streetlight boxes left were 98% right.

Dashcam videos often carry a channel logo, watermark or title burned into
a corner of the picture, and the panoptic model takes it for a billboard
or sign. When the image size is given, a traffic-sign or commercial-sign
box is dropped if it lies in the top or bottom ``OVERLAY_BAND`` of the
frame, reaches into the outer ``OVERLAY_SIDE`` at the left or right, and
is at least ``OVERLAY_MIN_HEIGHT`` of the frame tall (small real signs at
the frame edge are kept). On the two reviewed sets of 100 CROWD frames
(2026-11-25) it dropped 29 of 38 overlay boxes and no real one; on the
200 Vistas images, which have no overlays, it dropped 1 of 1,093 right
sign boxes and 3 false ones. Overlays elsewhere (captions across the
middle) are not caught.

Added boxes go through de-duplication and the conflict check like any
other box, so the scene VLM still decides their finer class, light state
or sign type. The conflict check keeps a detector box in its group and
does not drop a confident one (see conflict_check.py).
"""

from __future__ import annotations

import re
from typing import Any

MATCH_IOU = 0.5
BILLBOARD_INSIDE = 0.8
BILLBOARD_LABELS = frozenset({"Billboard", "Banner"})
LAMP_HEAD_LABEL = "Street Light"
LAMP_INSIDE = 0.5
LAMP_MIN_ASPECT = 0.7
OVERLAY_BAND = 0.2
OVERLAY_SIDE = 0.05
OVERLAY_MIN_HEIGHT = 0.03
OVERLAY_GROUPS = frozenset({"traffic_sign", "commercial_sign"})

CAR = frozenset({
    "car", "suv", "van", "taxi", "kei_car", "london_black_cab", "police_vehicle",
    "willys_jeep", "bakkie", "karwa_taxi", "pesero", "pickup_truck", "ambulance",
})
TRUCK = frozenset({
    "truck", "semi_trailer", "tanker_truck", "concrete_mixer", "tow_truck",
    "fire_truck", "jingle_truck", "road_train", "mining_truck", "snowplow",
})
BUS = frozenset({
    "city_bus", "school_bus", "double_decker_bus", "transmilenio_bus", "chiva_bus",
    "diablo_rojo", "car_rapide", "matatu", "danfo", "dala_dala", "jeepney",
    "minibus_taxi_sa", "songthaew", "angkot",
})
MOTORCYCLE = frozenset({"motorcycle", "scooter", "boda_boda", "habal_habal"})
PERSON = frozenset({
    "pedestrian", "child_pedestrian", "elderly_pedestrian", "traffic_police_officer",
    "wheelchair_user", "road_maintenance_worker", "street_vendor", "animal_herder",
    "traffic_controller",
})
RIDER = frozenset({"cyclist", "motorcyclist", "delivery_rider"})
COMMERCIAL = frozenset({"billboard", "shop_sign"})
CROSSWALK = frozenset({
    "zebra_crossing", "pedestrian_crossing_controlled", "school_crossing_marking",
    "ladder_crossing",
})
PEDESTRIAN_SIGNALS = re.compile(
    r"^(pedestrian_signal|bicycle_signal|ampelmann_signal"
    r"|traffic_light_(pedestrian|bicycle|toucan).*)$"
)


def fusion_group(class_name: str, super_category: str | None) -> str | None:
    """Coarse group of an ontology class, or None if no detector covers it."""

    name = str(class_name or "").strip().lower().replace(" ", "_")
    if name in CAR:
        return "car"
    if name in TRUCK:
        return "truck"
    if name in BUS:
        return "bus"
    if name in MOTORCYCLE:
        return "motorcycle"
    if name in PERSON:
        return "person"
    if name in RIDER:
        return "rider"
    if name in COMMERCIAL:
        return "commercial_sign"
    if name in CROSSWALK:
        return "crosswalk"
    if name == "manhole_cover":
        return "manhole"
    if super_category == "traffic_signal" and not PEDESTRIAN_SIGNALS.match(name):
        return "traffic_light"
    if super_category == "traffic_sign":
        return "traffic_sign"
    return None


def _inside(inner: Any, outer: Any) -> float:
    """Share of ``inner`` that lies inside ``outer``."""

    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in inner)
        bx1, by1, bx2, by2 = (float(v) for v in outer)
    except (TypeError, ValueError):
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area = (ax2 - ax1) * (ay2 - ay1)
    return iw * ih / area if area > 0 else 0.0


def _iou(a: Any, b: Any) -> float:

    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
    except (TypeError, ValueError):
        return 0.0
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def fuse_proposals(
    detections: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    class_groups: dict[str, str],
    class_ids: dict[str, Any],
    mode: str = "replace",
    image_name: str = "",
    check_lamps: bool = False,
    image_size: tuple[float, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (fused detections, counts) for one image.

    ``class_groups`` maps ontology class names to super-categories and
    ``class_ids`` to ontology class ids. ``check_lamps`` turns on the
    streetlight check; set it only when the proposals include lamp heads.
    ``image_size`` (width, height) turns on the video-overlay check.
    """

    def group_of(name: str) -> str | None:
        return fusion_group(name, class_groups.get(name))

    def name_of(detection: dict[str, Any]) -> str:
        return str(
            detection.get("class_name") or detection.get("final_label") or ""
        ).strip().lower().replace(" ", "_")

    billboards = [p["bbox"] for p in proposals if p.get("detector_label") in BILLBOARD_LABELS]
    lamp_heads = [p["bbox"] for p in proposals if p.get("detector_label") == LAMP_HEAD_LABEL]

    def in_billboard(name: str, bbox: Any) -> bool:
        return group_of(name) in ("person", "rider") and any(
            _inside(bbox, b) >= BILLBOARD_INSIDE for b in billboards
        )

    proposals = [p for p in proposals if group_of(p.get("class_name", ""))]
    covered = {group_of(p["class_name"]) for p in proposals}
    counts = {
        "matched": 0, "added": 0, "dropped": 0, "kept_unmatched": 0, "in_billboard": 0,
        "streetlight_no_lamp": 0, "video_overlay": 0,
    }

    # Greedy one-to-one matching, best IoU first, within a group.
    pairs = []
    for i, detection in enumerate(detections):
        group = group_of(name_of(detection))
        if group not in covered:
            continue
        for j, proposal in enumerate(proposals):
            if group_of(proposal["class_name"]) != group:
                continue
            iou = _iou(detection.get("bbox"), proposal["bbox"])
            if iou >= MATCH_IOU:
                pairs.append((iou, i, j))
    matched_detection: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in sorted(pairs, reverse=True):
        if i in matched_detection or j in used:
            continue
        matched_detection[i] = j
        used.add(j)

    fused = []
    for i, detection in enumerate(detections):
        group = group_of(name_of(detection))
        if i in matched_detection:
            proposal = proposals[matched_detection[i]]
            fused.append({
                **detection,
                "grounding_bbox": detection.get("bbox"),
                "bbox": list(proposal["bbox"]),
                "detector": proposal.get("detector"),
                "detector_label": proposal.get("detector_label"),
                "detector_confidence": proposal.get("detector_confidence"),
            })
            counts["matched"] += 1
        elif group in covered and mode == "replace":
            counts["dropped"] += 1
        else:
            if group in covered:
                counts["kept_unmatched"] += 1
            fused.append(detection)

    stem = image_name.rsplit(".", 1)[0]
    for j, proposal in enumerate(proposals):
        if j in used:
            continue
        name = proposal["class_name"]
        fused.append({
            "object_id": f"det_{stem}_{j}",
            "bbox": list(proposal["bbox"]),
            "class_name": name,
            "final_label": name.replace("_", " "),
            "class_id": class_ids.get(name),
            "source": proposal.get("detector") or "detector",
            "detector": proposal.get("detector"),
            "detector_label": proposal.get("detector_label"),
            "detector_confidence": proposal.get("detector_confidence"),
            "annotation_postprocessing_applied": False,
            "semantic_verification_applied": False,
        })
        counts["added"] += 1

    kept = [d for d in fused if not in_billboard(name_of(d), d.get("bbox"))]
    counts["in_billboard"] = len(fused) - len(kept)
    if check_lamps:
        before = len(kept)
        kept = [
            d for d in kept
            if name_of(d) != "streetlight" or holds_lamp(d.get("bbox"), lamp_heads)
        ]
        counts["streetlight_no_lamp"] = before - len(kept)
    if image_size:
        before = len(kept)
        kept = [
            d for d in kept
            if group_of(name_of(d)) not in OVERLAY_GROUPS
            or not in_overlay_corner(d.get("bbox"), *image_size)
        ]
        counts["video_overlay"] = before - len(kept)
    return kept, counts


def in_overlay_corner(bbox: Any, width: float, height: float) -> bool:
    """Whether a box sits where videos put channel logos and titles."""

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        width, height = float(width), float(height)
    except (TypeError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    in_band = y2 <= OVERLAY_BAND * height or y1 >= (1 - OVERLAY_BAND) * height
    at_side = x1 <= OVERLAY_SIDE * width or x2 >= (1 - OVERLAY_SIDE) * width
    return in_band and at_side and y2 - y1 >= OVERLAY_MIN_HEIGHT * height


def holds_lamp(bbox: Any, lamp_heads: list[Any]) -> bool:
    """Whether a streetlight box is tall enough and holds a lamp head."""

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    if y2 - y1 < LAMP_MIN_ASPECT * (x2 - x1):
        return False
    return any(_inside(head, bbox) >= LAMP_INSIDE for head in lamp_heads)

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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (fused detections, counts) for one image.

    ``class_groups`` maps ontology class names to super-categories and
    ``class_ids`` to ontology class ids.
    """

    def group_of(name: str) -> str | None:
        return fusion_group(name, class_groups.get(name))

    def name_of(detection: dict[str, Any]) -> str:
        return str(
            detection.get("class_name") or detection.get("final_label") or ""
        ).strip().lower().replace(" ", "_")

    billboards = [p["bbox"] for p in proposals if p.get("detector_label") in BILLBOARD_LABELS]

    def in_billboard(name: str, bbox: Any) -> bool:
        return group_of(name) in ("person", "rider") and any(
            _inside(bbox, b) >= BILLBOARD_INSIDE for b in billboards
        )

    proposals = [p for p in proposals if group_of(p.get("class_name", ""))]
    covered = {group_of(p["class_name"]) for p in proposals}
    counts = {
        "matched": 0, "added": 0, "dropped": 0, "kept_unmatched": 0, "in_billboard": 0,
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
    return kept, counts

"""
Write a high-precision copy of a TraffCOCO COCO export.

The full export keeps every box. This tool drops the boxes that, on
human-labelled Mapillary Vistas images, were too often wrong for a chosen
target precision, and boxes smaller than a minimum side, and writes the rest
next to the input (``instances_precise.json`` by default).

A box is judged by its group and its source:

* group: the detector group of its class (car, person, traffic_sign, ...,
  see annotation_pipeline/pipeline/detector_fusion.py), or the class name
  for classes no detector covers (streetlight, stop_line, ...);
* source: ``matched`` (a detector box that a grounding box also found),
  ``added`` (a detector box only) or ``grnd`` (a grounding box only).

Groups with little evidence in the fitting data share one ``rare|<source>``
row. The table (``quality_gate_vistas.json``) gives, per minimum side and
target precision, the lowest detector confidence a box of each group needs
(0 keeps all, null drops all). Boxes of a group the table never saw are kept.
The table also records the cross-validated precision and recall it reached.

A veto then drops the boxes that a small gradient-boosted model scores below
``--veto`` (0.2 by default, 0 turns it off). The model
(``quality_gate_veto_vistas.json``, made by ``tools/fit_quality_gate.py``)
was fitted on the same kind of Vistas data and reads only fields the export
keeps (see ``FEATURES``): the group and source, which detector gave the box,
the detector, SAM, scene and ontology scores, the box size and place in the
image, and how the box overlaps the others. Used alone it lost many real
CROWD boxes of classes Vistas does not label (bollards, guardrails,
streetlights), so it only removes boxes it is nearly sure are wrong.

Scoring needs no extra packages; fitting the model needs scikit-learn.

Usage:
  uv run python -m tools.quality_gate \
      --instances outputs/COCO_FORMAT/annotations/instances_final.json \
      --target 0.92 --min-side 0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from annotation_pipeline.pipeline.detector_fusion import fusion_group

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "tools" / "quality_gate_vistas.json"
DEFAULT_VETO_MODEL = ROOT / "tools" / "quality_gate_veto_vistas.json"
DEFAULT_ONTOLOGY = ROOT / "annotation_pipeline" / "models" / "ontology" / "traffic_ontology_v2.json"

GROUPS = (
    "car", "truck", "bus", "motorcycle", "person", "rider", "traffic_light", "traffic_sign",
    "commercial_sign", "crosswalk", "manhole", "streetlight",
)
SOURCES = ("matched", "added", "grnd")
DETECTORS = ("yolo", "mask2former")
FEATURES = (
    [f"group_{g}" for g in GROUPS] + ["group_other"]
    + [f"source_{s}" for s in SOURCES] + [f"detector_{d}" for d in DETECTORS]
    + [
        "detector_confidence", "sam_score", "scene_confidence", "ontology_score", "alternative_labels",
        "log_width", "log_height", "centre_y", "centre_x_offset", "top", "bottom",
        "boxes_in_image", "same_group_in_image", "iou_same_group", "iou_other_group",
        "inside_other_group", "contains_other_group",
    ]
)


def box_source(annotation: dict[str, Any]) -> str:
    """matched / added / grnd, from the fields the export keeps."""

    if str(annotation.get("object_id") or "").startswith("det_"):
        return "added"
    if annotation.get("detector_confidence") is not None:
        return "matched"
    return "grnd"


def box_group(class_name: str, super_category: str | None) -> str:

    return fusion_group(class_name, super_category) or class_name


def gate_key(class_name: str, super_category: str | None, source: str, rare: set[str]) -> str:

    key = f"{box_group(class_name, super_category)}|{source}"
    return f"rare|{source}" if key in rare else key


def passes(annotation: dict[str, Any], key: str, table: dict[str, Any], min_side: float) -> bool:

    _, _, width, height = annotation["bbox"]
    if min(width, height) < min_side:
        return False
    if key not in table:
        return True
    threshold = table[key]
    if threshold is None:
        return False
    return float(annotation.get("detector_confidence") or 0.0) >= threshold


def _corners(annotation: dict[str, Any]) -> list[float]:

    x, y, w, h = (float(v) for v in annotation["bbox"])
    return [x, y, x + w, y + h]


def _overlap(a: list[float], b: list[float]) -> float:

    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def _area(a: list[float]) -> float:

    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _number(value: Any) -> float:
    """A score, or -1 when the box has none."""

    return -1.0 if value is None else float(value)


def image_features(
    annotations: list[dict[str, Any]],
    width: float,
    height: float,
    class_names: dict[int, str],
    super_category: dict[str, str | None],
) -> list[list[float]]:
    """Feature vectors (in ``FEATURES`` order) for the boxes of one image."""

    boxes = [_corners(a) for a in annotations]
    groups = []
    for a in annotations:
        name = class_names.get(a["category_id"], "")
        groups.append(box_group(name, super_category.get(name)))
    rows = []
    for i, a in enumerate(annotations):
        box, group = boxes[i], groups[i]
        same = [j for j in range(len(boxes)) if j != i and groups[j] == group]
        other = [j for j in range(len(boxes)) if j != i and groups[j] != group]

        def iou(j: int) -> float:
            inter = _overlap(box, boxes[j])
            union = _area(box) + _area(boxes[j]) - inter
            return inter / union if union > 0 else 0.0

        def share(inner: list[float], outer: list[float]) -> float:
            area = _area(inner)
            return _overlap(inner, outer) / area if area > 0 else 0.0

        source = box_source(a)
        detector = str(a.get("source") or "").lower() if source == "added" else ""
        w, h = max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)
        row = [1.0 if group == g else 0.0 for g in GROUPS] + [0.0 if group in GROUPS else 1.0]
        row += [1.0 if source == s else 0.0 for s in SOURCES]
        row += [1.0 if detector.startswith(d) else 0.0 for d in DETECTORS]
        row += [
            _number(a.get("detector_confidence")),
            _number(a.get("sam_score")),
            _number(a.get("scene_confidence")),
            _number(a.get("ontology_score")),
            float(len(a.get("alternative_labels") or [])),
            math.log(w / width),
            math.log(h / height),
            (box[1] + box[3]) / 2 / height,
            abs((box[0] + box[2]) / 2 / width - 0.5),
            box[1] / height,
            box[3] / height,
            float(len(boxes)),
            float(len(same)),
            max([iou(j) for j in same] or [0.0]),
            max([iou(j) for j in other] or [0.0]),
            max([share(box, boxes[j]) for j in other] or [0.0]),
            max([share(boxes[j], box) for j in other] or [0.0]),
        ]
        rows.append(row)
    return rows


def export_features(coco: dict[str, Any], ontology: dict[str, Any]) -> dict[Any, list[float]]:
    """{annotation id: feature vector} for a whole COCO export."""

    super_category = {c["class_name"]: c.get("super_category") for c in ontology["classes"]}
    names = {c["id"]: c["name"].replace(" ", "_") for c in coco["categories"]}
    by_image: dict[Any, list[dict[str, Any]]] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    features = {}
    for image in coco["images"]:
        annotations = by_image.get(image["id"], [])
        rows = image_features(annotations, float(image["width"]), float(image["height"]), names, super_category)
        for a, row in zip(annotations, rows):
            features[a["id"]] = row
    return features


def score(model: dict[str, Any], row: list[float]) -> float:
    """Probability that a box is right, from the boosted trees in ``model``."""

    raw = model["baseline"]
    for tree in model["trees"]:
        node = tree[0]
        while not node["leaf"]:
            value = row[node["feature"]]
            node = tree[node["left"] if value <= node["threshold"] else node["right"]]
        raw += node["value"]
    return 1.0 / (1.0 + math.exp(-raw))


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--veto-model", type=Path, default=DEFAULT_VETO_MODEL)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--target", default="0.92", help="target precision, a key of the table")
    parser.add_argument(
        "--min-side",
        default="0",
        help="minimum box side in pixels, a key of the table (0, 16, 24 or 32). CROWD frames "
        "are smaller than Vistas images, so 0 kept far more real objects there at the same precision",
    )
    parser.add_argument(
        "--veto", type=float, default=0.2,
        help="drop boxes the veto model scores below this (0 turns the veto off)",
    )
    args = parser.parse_args()

    fits = json.loads(args.table.read_text())["fits"]
    if args.min_side not in fits or args.target not in fits[args.min_side]["targets"]:
        raise SystemExit(
            f"No fit for min side {args.min_side} and target {args.target}; "
            f"the table has min sides {sorted(fits)} and targets "
            f"{sorted(next(iter(fits.values()))['targets'])}."
        )
    fit = fits[args.min_side]
    chosen = fit["targets"][args.target]
    rare = set(fit["rare_groups"])
    min_side = float(args.min_side)

    ontology = json.loads(args.ontology.read_text())
    super_category = {c["class_name"]: c.get("super_category") for c in ontology["classes"]}
    coco = json.loads(args.instances.read_text())
    names = {c["id"]: c["name"].replace(" ", "_") for c in coco["categories"]}

    kept = []
    for annotation in coco["annotations"]:
        name = names.get(annotation["category_id"], "")
        key = gate_key(name, super_category.get(name), box_source(annotation), rare)
        if passes(annotation, key, chosen["table"], min_side):
            kept.append(annotation)

    vetoed = 0
    veto_model = None
    if args.veto > 0:
        veto_model = json.loads(args.veto_model.read_text())
        features = export_features(coco, ontology)
        before = len(kept)
        kept = [a for a in kept if score(veto_model, features[a["id"]]) >= args.veto]
        vetoed = before - len(kept)

    out = args.out or args.instances.with_name("instances_precise.json")
    out.write_text(json.dumps({
        **coco,
        "annotations": kept,
        "quality_gate": {
            "target_precision": float(args.target),
            "min_side": min_side,
            "cv_precision": chosen["cv_precision"],
            "cv_recall": chosen["cv_recall"],
            "fitted_on": fit.get("fitted_on"),
            "veto": args.veto,
            "veto_fitted_on": veto_model.get("fitted_on") if veto_model else None,
        },
    }))
    print(f"Kept {len(kept)} of {len(coco['annotations'])} boxes ({vetoed} dropped by the veto) -> {out}")


if __name__ == "__main__":
    main()

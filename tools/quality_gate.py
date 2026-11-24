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

Usage:
  uv run python -m tools.quality_gate \
      --instances outputs/COCO_FORMAT/annotations/instances_final.json \
      --target 0.92 --min-side 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from annotation_pipeline.pipeline.detector_fusion import fusion_group

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = ROOT / "tools" / "quality_gate_vistas.json"
DEFAULT_ONTOLOGY = ROOT / "annotation_pipeline" / "models" / "ontology" / "traffic_ontology_v2.json"


def box_source(annotation: dict[str, Any]) -> str:
    """matched / added / grnd, from the fields the export keeps."""

    if str(annotation.get("object_id") or "").startswith("det_"):
        return "added"
    if annotation.get("detector_confidence") is not None:
        return "matched"
    return "grnd"


def gate_key(class_name: str, super_category: str | None, source: str, rare: set[str]) -> str:

    key = f"{fusion_group(class_name, super_category) or class_name}|{source}"
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


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--target", default="0.92", help="target precision, a key of the table")
    parser.add_argument(
        "--min-side",
        default="0",
        help="minimum box side in pixels, a key of the table (0, 16, 24 or 32). CROWD frames "
        "are smaller than Vistas images, so 0 kept far more real objects there at the same precision",
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
        },
    }))
    print(f"Kept {len(kept)} of {len(coco['annotations'])} boxes -> {out}")


if __name__ == "__main__":
    main()

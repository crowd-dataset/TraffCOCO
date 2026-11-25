"""
Fit the veto model used by ``tools/quality_gate.py``.

Input is a JSON file of scored boxes from a TraffCOCO export checked against
human labels (for Mapillary Vistas, the scorer in the evaluation folder
writes it):

  {"objects": {"0": <labelled objects>, "16": ..., "24": ..., "32": ...},
   "rows": [{"image": <image name>, "tp": true/false, "side": <box min side>,
             "object_side": <min side of the matched object, or null>,
             "features": <tools.quality_gate.image_features row>}, ...]}

A gradient-boosted classifier (scikit-learn) learns from the features
whether a box is right. The gate uses it only as a veto on low scores.
The file also records, for every minimum side and target precision, what
the model would reach used alone: the cut-off on its score comes from
out-of-fold scores, and 5-fold
cross-validation over images (with the cut-off chosen inside each training
part) gives the precision and recall written next to it. Minimum sides
follow the COCO area-range rule: a box matched to a smaller object and an
unmatched smaller box are left out, and recall counts only objects at least
that large.

Usage:
  uv run --with scikit-learn python -m tools.fit_quality_gate \
      --rows rows.json --out tools/quality_gate_veto_vistas.json --fitted-on "..."
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from tools.quality_gate import FEATURES

MIN_SIDES = ("0", "16", "24", "32")
TARGETS = ("0.85", "0.88", "0.9", "0.91", "0.92", "0.93", "0.95")
FOLDS = 5


def new_model() -> HistGradientBoostingClassifier:

    return HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0,
    )


def out_of_fold(x: np.ndarray, y: np.ndarray, images: np.ndarray, folds: list[list[str]]) -> np.ndarray:

    scores = np.zeros(len(y))
    for fold in folds:
        test = np.isin(images, fold)
        scores[test] = new_model().fit(x[~test], y[~test]).predict_proba(x[test])[:, 1]
    return scores


def cut_for(scores: np.ndarray, y: np.ndarray, target: float) -> float:
    """Lowest score that keeps the precision of the boxes above it at the target."""

    order = np.argsort(-scores)
    precision = np.cumsum(y[order]) / np.arange(1, len(order) + 1)
    ok = np.where(precision >= target)[0]
    return float(scores[order][ok[-1]]) if len(ok) else float("inf")


def in_range(rows: list[dict[str, Any]], side: float) -> np.ndarray:
    """COCO area range: which rows count for a minimum side."""

    return np.array([
        (r["object_side"] >= side) if r["tp"] else (r["side"] >= side) for r in rows
    ])


def export_trees(model: HistGradientBoostingClassifier) -> list[list[dict[str, Any]]]:

    trees = []
    for (predictor,) in model._predictors:
        tree = []
        for node in predictor.nodes:
            if node["is_leaf"]:
                tree.append({"leaf": True, "value": float(node["value"])})
            else:
                tree.append({
                    "leaf": False, "feature": int(node["feature_idx"]),
                    "threshold": float(node["num_threshold"]),
                    "left": int(node["left"]), "right": int(node["right"]),
                })
        trees.append(tree)
    return trees


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fitted-on", required=True)
    args = parser.parse_args()

    data = json.loads(args.rows.read_text())
    rows = data["rows"]
    x = np.array([r["features"] for r in rows], float)
    if x.shape[1] != len(FEATURES):
        raise SystemExit(f"Rows have {x.shape[1]} features, the gate expects {len(FEATURES)}.")
    y = np.array([r["tp"] for r in rows], float)
    images = np.array([r["image"] for r in rows])
    names = sorted(set(images))
    random.seed(0)
    random.shuffle(names)
    folds = [names[i::FOLDS] for i in range(FOLDS)]

    # Cross-validated precision and recall: the cut-off of each outer fold comes
    # from out-of-fold scores inside its training part.
    outer = np.zeros(len(y))
    inner_cut: dict[tuple[int, str, str], float] = {}
    for i, fold in enumerate(folds):
        test = np.isin(images, fold)
        train_names = [n for j, f in enumerate(folds) if j != i for n in f]
        inner = [train_names[k::FOLDS - 1] for k in range(FOLDS - 1)]
        train_scores = out_of_fold(x[~test], y[~test], images[~test], inner)
        train_rows = [r for r, t in zip(rows, test) if not t]
        for side in MIN_SIDES:
            keep = in_range(train_rows, float(side))
            for target in TARGETS:
                inner_cut[i, side, target] = cut_for(train_scores[keep], y[~test][keep], float(target))
        outer[test] = new_model().fit(x[~test], y[~test]).predict_proba(x[test])[:, 1]

    final_scores = out_of_fold(x, y, images, folds)
    model = new_model().fit(x, y)
    cuts: dict[str, dict[str, Any]] = {}
    for side in MIN_SIDES:
        keep = in_range(rows, float(side))
        cuts[side] = {}
        for target in TARGETS:
            passed = np.zeros(len(y), bool)
            for i, fold in enumerate(folds):
                test = np.isin(images, fold)
                passed[test] = outer[test] >= inner_cut[i, side, target]
            passed &= keep
            tp = float(y[passed].sum())
            precision = tp / passed.sum() if passed.sum() else 0.0
            recall = tp / data["objects"][side]
            cuts[side][target] = {
                "cut": round(cut_for(final_scores[keep], y[keep], float(target)), 6),
                "cv_precision": round(precision, 3),
                "cv_recall": round(recall, 3),
            }
            print(f"min side {side:>2}, target {target}: CV precision {precision:.1%}, recall {recall:.1%}")

    args.out.write_text(json.dumps({
        "fitted_on": args.fitted_on,
        "features": list(FEATURES),
        "baseline": float(np.ravel(model._baseline_prediction)[0]),
        "trees": export_trees(model),
        "cuts": cuts,
    }))
    check = [i for i in range(0, len(y), max(1, len(y) // 200))]
    from tools.quality_gate import score
    gap = max(abs(score({"baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": export_trees(model)},
                        list(x[i])) - model.predict_proba(x[i:i + 1])[0, 1]) for i in check)
    print(f"Wrote {args.out} ({len(model._predictors)} trees; largest score gap to scikit-learn {gap:.2e}).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


TRAINING_DIR = Path(__file__).resolve().parent
ANNOTATION_PIPELINE_DIR = TRAINING_DIR.parent

DEFAULT_DATASET_YAML = (
    ANNOTATION_PIPELINE_DIR
    / "outputs"
    / "YOLO_FORMAT"
    / "data.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a trained TraffCOCO YOLO model."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained YOLO .pt checkpoint.",
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATASET_YAML),
        help="Path to data.yaml.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    args = parser.parse_args()

    model = YOLO(args.model)

    kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
    }

    if args.device is not None:
        kwargs["device"] = args.device

    results = model.val(**kwargs)

    print("\nValidation complete.")

    if hasattr(results, "box"):
        print(f"mAP50:     {results.box.map50:.4f}")
        print(f"mAP50-95:  {results.box.map:.4f}")


if __name__ == "__main__":
    main()
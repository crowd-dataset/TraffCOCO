"""Validate a trained TraffCOCO YOLO detection model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


TRAINING_DIR = Path(__file__).resolve().parent
ANNOTATION_PIPELINE_DIR = TRAINING_DIR.parent

DEFAULT_MODEL = TRAINING_DIR / "weights" / "best.pt"
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
        default=str(DEFAULT_MODEL),
        help="Path to trained YOLO .pt model. Defaults to training/weights/best.pt.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATASET_YAML),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model does not exist: {model_path}"
        )

    model = YOLO(str(model_path))

    kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "split": args.split,
    }

    if args.device is not None:
        kwargs["device"] = args.device

    results = model.val(**kwargs)

    print("\nValidation complete.")

    if hasattr(results, "box"):
        print(f"Precision:  {results.box.mp:.4f}")
        print(f"Recall:     {results.box.mr:.4f}")
        print(f"mAP50:      {results.box.map50:.4f}")
        print(f"mAP50-95:   {results.box.map:.4f}")


if __name__ == "__main__":
    main()

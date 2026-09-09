#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


TRAINING_DIR = Path(__file__).resolve().parent
ANNOTATION_PIPELINE_DIR = TRAINING_DIR/ "annotation_pipeline"

DEFAULT_DATASET_YAML = (
    ANNOTATION_PIPELINE_DIR
    / "outputs"
    / "YOLO_FORMAT"
    / "data.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO object detection model on the TraffCOCO dataset."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the pretrained YOLO .pt model.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATASET_YAML),
        help="Path to YOLO dataset YAML.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device, e.g. '0', '1', or 'cpu'.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=str(TRAINING_DIR / "runs"),
        help="Directory where training runs are saved.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="traffcoco_yolo",
        help="Training run name.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images in memory.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the supplied model checkpoint.",
    )

    return parser.parse_args()


def validate_dataset(data_yaml: Path) -> dict:
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{data_yaml}"
        )

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("data.yaml must contain a YAML mapping.")

    required_keys = {"train", "val", "names"}

    missing = required_keys - set(data.keys())

    if missing:
        raise ValueError(
            f"data.yaml is missing required fields: {sorted(missing)}"
        )

    return data


def resolve_dataset_paths(data_yaml: Path, data: dict) -> None:
    dataset_root = Path(data.get("path", data_yaml.parent))

    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    for split in ("train", "val"):
        split_path = Path(data[split])

        if not split_path.is_absolute():
            split_path = dataset_root / split_path

        if not split_path.exists():
            raise FileNotFoundError(
                f"{split.upper()} dataset path does not exist:\n"
                f"{split_path}"
            )

        print(f"{split.upper()} images: {split_path}")

    if "test" in data:
        test_path = Path(data["test"])

        if not test_path.is_absolute():
            test_path = dataset_root / test_path

        if test_path.exists():
            print(f"TEST images: {test_path}")


def main() -> int:
    args = parse_args()

    data_yaml = Path(args.data).resolve()

    print("=" * 70)
    print("TraffCOCO YOLO Training")
    print("=" * 70)

    print(f"Model:       {args.model}")
    print(f"Dataset:     {data_yaml}")
    print(f"Epochs:      {args.epochs}")
    print(f"Image size:  {args.imgsz}")
    print(f"Batch:       {args.batch}")
    print(f"Device:      {args.device or 'auto'}")
    print(f"Workers:     {args.workers}")
    print("=" * 70)

    data = validate_dataset(data_yaml)

    print("\nDataset classes:")
    for class_id, class_name in enumerate(data["names"]):
        print(f"  {class_id}: {class_name}")

    print()

    resolve_dataset_paths(data_yaml, data)

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO model checkpoint not found:\n{model_path}"
        )

    model = YOLO(str(model_path))

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "patience": args.patience,
        "cache": args.cache,
        "resume": args.resume,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    print("\nStarting training...\n")

    results = model.train(**train_kwargs)

    print("\n" + "=" * 70)
    print("Training complete")
    print("=" * 70)

    if hasattr(results, "save_dir"):
        print(f"Results saved to: {results.save_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


TRAINING_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TraffCOCO YOLO inference."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained YOLO .pt model.",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Image, directory, video, or other supported source.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--project",
        default=str(TRAINING_DIR / "predictions"),
    )

    parser.add_argument(
        "--name",
        default="predict",
    )

    parser.add_argument(
        "--save-txt",
        action="store_true",
    )

    parser.add_argument(
        "--save-conf",
        action="store_true",
    )

    args = parser.parse_args()

    model = YOLO(args.model)

    kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "project": args.project,
        "name": args.name,
        "save": True,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
    }

    if args.device is not None:
        kwargs["device"] = args.device

    model.predict(**kwargs)


if __name__ == "__main__":
    main()
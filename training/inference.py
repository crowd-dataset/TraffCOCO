"""Run TraffCOCO YOLO inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = TRAINING_DIR / "weights" / "best.pt"
DEFAULT_PROJECT = TRAINING_DIR.parent / "outputs" / "inference"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TraffCOCO YOLO inference."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to trained YOLO .pt model. Defaults to training/weights/best.pt.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image, directory, video, or other supported source.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT),
    )
    parser.add_argument("--name", default="predict")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO-format text predictions.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence values in saved text predictions.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming inference for very large sources.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model does not exist: {model_path}"
        )

    model = YOLO(str(model_path))

    kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "project": args.project,
        "name": args.name,
        "save": True,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "batch": args.batch,
        "workers": args.workers,
        "stream": args.stream,
    }

    if args.device is not None:
        kwargs["device"] = args.device

    results = model.predict(**kwargs)

    if args.stream:
        # Consume the generator so inference actually runs.
        for _ in results:
            pass


if __name__ == "__main__":
    main()

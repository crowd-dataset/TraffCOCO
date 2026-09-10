"""Run TraffCOCO YOLO inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO


TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = TRAINING_DIR / "weights" / "best.pt"
DEFAULT_PROJECT = TRAINING_DIR.parent / "outputs" / "inference"


def run_yolo_inference(
    model_path: Path | str,
    source: str | Path | list[Path | str],
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | int | None = None,
    project: Path | str = DEFAULT_PROJECT,
    name: str = "predict",
    batch: int = 1,
    workers: int = 8,
    save_txt: bool = False,
    save_conf: bool = False,
    stream: bool = False,
) -> Any:
    """Run YOLO inference using the supplied trained model and source."""
    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model does not exist: {model_path}"
        )

    model = YOLO(str(model_path))

    kwargs = {
        "source": source,
        "imgsz": imgsz,
        "conf": conf,
        "project": str(project),
        "name": name,
        "save": True,
        "save_txt": save_txt,
        "save_conf": save_conf,
        "batch": batch,
        "workers": workers,
        "stream": stream,
    }

    if device is not None:
        kwargs["device"] = device

    results = model.predict(**kwargs)

    if stream:
        # Consume the generator so inference actually runs.
        results = list(results)

    return results


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

    run_yolo_inference(
        model_path=args.model,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        batch=args.batch,
        workers=args.workers,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()

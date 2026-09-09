"""Train the final TraffCOCO YOLO detection model.

The training run is resumable. If an interrupted run has a valid
``last.pt`` checkpoint in the canonical run directory, the next invocation
resumes that run with Ultralytics ``resume=True`` instead of starting from
epoch 0.

Canonical layout
----------------
training/
├── pretrained/
│   └── yolo26n.pt          # base/pretrained model
├── train.py
├── inference.py
├── validate.py
└── ...

outputs/
├── YOLO_FORMAT/
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── training/
    ├── training_state.json
    ├── weights/
    │   ├── last.pt
    │   └── best.pt
    ├── results.csv
    └── args.yaml

The ``training/pretrained`` directory contains only the original pretrained
model. All artifacts produced by YOLO training are kept under
``outputs/training``. Nothing is copied back into ``training/`` after the
run completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


RUN_NAME = "train"
# Kept as the logical run name in training_state for compatibility. The
# physical Ultralytics run directory is outputs/training itself.
STATE_FILENAME = "training_state.json"
LOCK_FILENAME = ".training.lock"
CHECKPOINT_RELATIVE_PATH = Path("weights") / "last.pt"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRETRAINED_DIR = PROJECT_ROOT / "training" / "pretrained"
DEFAULT_PRETRAINED_MODEL = PRETRAINED_DIR / "yolo26n.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_path(output_dir: Path) -> Path:
    return output_dir / STATE_FILENAME


def _run_dir(output_dir: Path) -> Path:
    # The canonical YOLO run directory is outputs/training itself.
    return output_dir


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / CHECKPOINT_RELATIVE_PATH


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _build_training_state(
    *,
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    patience: int,
    device: str | int | None,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "status": "running",
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": _sha256_file(data_yaml),
        "model_name": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "patience": patience,
        "device": str(device) if device is not None else None,
        "seed": seed,
        "run_name": RUN_NAME,
        "checkpoint": str(CHECKPOINT_RELATIVE_PATH),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


def _acquire_lock(output_dir: Path) -> Path:
    """Prevent two processes from training the same run simultaneously."""
    lock_path = output_dir / LOCK_FILENAME
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at_utc": _utc_now(),
    }

    try:
        fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        existing = _read_json(lock_path)
        detail = ""
        if existing:
            detail = (
                f" Existing lock: host={existing.get('host')!r}, "
                f"pid={existing.get('pid')!r}, "
                f"created_at={existing.get('created_at_utc')!r}."
            )
        raise RuntimeError(
            "Another TraffCOCO YOLO training process appears to be using "
            f"{output_dir}.{detail} Remove {lock_path} only after confirming "
            "that no training process is still running."
        ) from exc

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return lock_path


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _validate_dataset(data_yaml: Path) -> None:
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"YOLO dataset YAML does not exist: {data_yaml}"
        )

    dataset_root = data_yaml.parent
    required_dirs = (
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    )

    missing_dirs = [path for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(
            "YOLO dataset is incomplete. Missing directories: "
            + ", ".join(str(path) for path in missing_dirs)
        )


def _validate_resume_state(
    state: dict[str, Any],
    *,
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    patience: int,
    seed: int,
) -> None:
    expected = {
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": _sha256_file(data_yaml),
        "model_name": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "patience": patience,
        "seed": seed,
        "run_name": RUN_NAME,
    }

    mismatches = []
    for key, expected_value in expected.items():
        actual_value = state.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: saved={actual_value!r}, requested={expected_value!r}"
            )

    if mismatches:
        raise RuntimeError(
            "An existing TraffCOCO training run was found, but its saved "
            "configuration/dataset fingerprint does not match this request. "
            "Refusing to start a potentially inconsistent resume.\n"
            + "\n".join(f"  - {item}" for item in mismatches)
        )


def _checkpoint_is_usable(checkpoint: Path) -> bool:
    if not checkpoint.is_file():
        return False
    try:
        YOLO(str(checkpoint))
    except Exception as exc:
        raise RuntimeError(
            f"Found checkpoint but it could not be loaded: {checkpoint}"
        ) from exc
    return True


def _validate_completed_checkpoints(output_dir: Path) -> None:
    """Verify that the completed run left both canonical checkpoints in place."""
    best_checkpoint = output_dir / "weights" / "best.pt"
    last_checkpoint = output_dir / "weights" / "last.pt"

    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"Training completed but best.pt was not found: {best_checkpoint}"
        )
    if not last_checkpoint.is_file():
        raise FileNotFoundError(
            f"Training completed but last.pt was not found: {last_checkpoint}"
        )

    print(f"Best model: {best_checkpoint}")
    print(f"Last model: {last_checkpoint}")

def _mark_state(output_dir: Path, status: str) -> None:
    state_path = _state_path(output_dir)
    state = _read_json(state_path) or {}
    state["status"] = status
    state["updated_at_utc"] = _utc_now()
    _write_json_atomic(state_path, state)


def _reset_training_run(output_dir: Path) -> None:
    """Remove stale training artifacts while preserving the active lock."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    for path in output_dir.iterdir():
        if path.name == LOCK_FILENAME:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()



def _ensure_pretrained_model(model_path: Path) -> Path:
    """Ensure the configured pretrained model exists in training/pretrained."""
    if model_path.is_file():
        return model_path

    if model_path.name != "yolo26n.pt":
        raise FileNotFoundError(
            f"Pretrained YOLO model does not exist: {model_path}. "
            "Automatic download is only configured for yolo26n.pt."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
    temporary = model_path.with_suffix(model_path.suffix + ".download")

    print(f"Pretrained model not found: {model_path}")
    print(f"Downloading pretrained model from: {url}")

    try:
        urllib.request.urlretrieve(url, temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("Downloaded pretrained model is empty.")
        os.replace(temporary, model_path)
    except Exception as exc:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"Failed to download pretrained model to {model_path}"
        ) from exc

    print(f"Pretrained model downloaded to: {model_path}")
    return model_path


def train_yolo(
    data_yaml: Path | str,
    output_dir: Path | str | None = None,
    model_name: str = "yolo26n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    workers: int = 8,
    patience: int = 20,
    device: str | int | None = None,
    seed: int = 42,
) -> Any:
    """Train YOLO, automatically resuming an interrupted canonical run.

    Resume policy:
      * No existing run -> start fresh from the configured pretrained model.
      * Valid last.pt -> resume it with resume=True.
      * Missing/invalid training_state.json -> delete stale training artifacts
        and restart from the configured pretrained model.
      * Existing artifacts without a checkpoint -> delete stale training
        artifacts and restart from the configured pretrained model.
      * A valid interrupted state plus last.pt -> resume the exact checkpoint,
        including optimizer/scheduler/scaler state, via Ultralytics resume=True.
      * Dataset/config mismatch in a valid state -> fail instead of silently
        mixing runs.
      * A run marked completed -> fail instead of silently restarting it.
    """

    data_yaml = Path(data_yaml).resolve()
    _validate_dataset(data_yaml)

    dataset_root = data_yaml.parent
    output_dir = (
        DEFAULT_OUTPUT_DIR
        if output_dir is None
        else Path(output_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    run_dir = _run_dir(output_dir)
    checkpoint = _checkpoint_path(output_dir)
    state_path = _state_path(output_dir)
    results_csv = run_dir / "results.csv"
    best_checkpoint = run_dir / "weights" / "best.pt"

    print(f"YOLO dataset : {data_yaml}")
    print(f"YOLO model   : {model_name}")
    print(f"Device       : {device}")
    print(f"Epochs       : {epochs}")
    print(f"Image size   : {imgsz}")
    print(f"Batch size   : {batch}")
    print(f"Output       : {output_dir}")

    state = _read_json(state_path)
    existing_artifacts = [
        path for path in run_dir.iterdir()
        if path.name != LOCK_FILENAME
    ] if run_dir.exists() else []
    has_run_artifacts = bool(existing_artifacts)

    # A stale/incomplete training directory is not a reason to get stuck.
    # Once the lock is acquired below, it is safe to remove the old run and
    # start again from the original pretrained model.
    if state is not None:
        _validate_resume_state(
            state,
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            patience=patience,
            seed=seed,
        )

    if state is not None and state.get("status") == "completed":
        if best_checkpoint.is_file() or checkpoint.is_file():
            raise RuntimeError(
                f"TraffCOCO training is already marked completed in "
                f"{state_path}. Refusing to restart it. Use a new output "
                "directory for a separate training run."
            )

    lock_path = _acquire_lock(output_dir)

    try:
        # Re-read after locking so concurrent invocations cannot race.
        state = _read_json(state_path)
        existing_artifacts = [
            path for path in run_dir.iterdir()
            if path.name != LOCK_FILENAME
        ] if run_dir.exists() else []

        if state is None and existing_artifacts:
            logger_message = (
                f"Existing training artifacts found at {run_dir}, but "
                f"{state_path.name} is missing or invalid. Restarting the "
                "YOLO training run from the configured pretrained model."
            )
            print(logger_message)
            _reset_training_run(output_dir)
            state = None

        if state is not None:
            _validate_resume_state(
                state,
                data_yaml=data_yaml,
                model_name=model_name,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                patience=patience,
                seed=seed,
            )

        checkpoint = _checkpoint_path(output_dir)
        run_dir = _run_dir(output_dir)
        results_csv = run_dir / "results.csv"

        if checkpoint.is_file():
            _checkpoint_is_usable(checkpoint)

            print(f"RESUMING YOLO TRAINING FROM: {checkpoint}")

            # Ultralytics resume restores the checkpoint's model weights,
            # optimizer state, scheduler state, scaler/EMA state and epoch.
            model = YOLO(str(checkpoint))
            results = model.train(resume=True)

            _validate_completed_checkpoints(output_dir)
            _mark_state(output_dir, "completed")
            return results

        remaining_artifacts = [
            path for path in run_dir.iterdir()
            if path.name != LOCK_FILENAME
        ] if run_dir.exists() else []
        if remaining_artifacts:
            print(
                f"Incomplete YOLO training artifacts found in {run_dir} "
                "without a resumable last.pt. Restarting from the configured "
                "pretrained model."
            )
            _reset_training_run(output_dir)

        state = _build_training_state(
            data_yaml=data_yaml,
            model_name=model_name,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            patience=patience,
            device=device,
            seed=seed,
        )
        _write_json_atomic(state_path, state)

        print("STARTING NEW YOLO TRAINING RUN")

        model_path = Path(model_name)
        if not model_path.is_absolute():
            model_path = PRETRAINED_DIR / model_path

        model_path = _ensure_pretrained_model(model_path)
        model = YOLO(str(model_path))

        try:
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                patience=patience,
                device=device,
                seed=seed,
                # Use outputs/training itself as the Ultralytics run directory.
                # This avoids the unwanted outputs/training/train nesting.
                project=str(output_dir.parent),
                name=output_dir.name,
                exist_ok=True,
                pretrained=True,
                save=True,
                save_period=1,
                verbose=True,
            )
        except BaseException:
            _mark_state(output_dir, "interrupted")
            raise

        _validate_completed_checkpoints(output_dir)
        _mark_state(output_dir, "completed")
        return results

    finally:
        _release_lock(lock_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/resume the TraffCOCO YOLO model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "outputs"
            / "YOLO_FORMAT"
            / "data.yaml"
        ),
    )
    parser.add_argument("--model", default=str(DEFAULT_PRETRAINED_MODEL))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

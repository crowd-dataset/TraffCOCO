"""YOLOPv2 auxiliary road-perception integration.

YOLOPv2 is used as an independent auxiliary source for generic road-user
detection plus its native drivable-area and lane-line semantic masks.

The VLM/Locate Anything branch remains responsible for traffic signs,
traffic signals, specialized vehicles, and other objects outside YOLOPv2's
closed detection set.

The implementation below follows the official YOLOPv2 inference flow:
TorchScript model -> split_for_trace_model -> non_max_suppression ->
driving_area_mask / lane_line_mask -> scale_coords.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from custom_logger import CustomLogger

logger = CustomLogger(__name__)

OFFICIAL_REPO_URL = "https://github.com/CAIC-AD/YOLOPv2.git"
OFFICIAL_WEIGHTS_URL = (
    "https://github.com/CAIC-AD/YOLOPv2/releases/download/"
    "V0.0.1/yolopv2.pt"
)

# YOLOPv2 uses the BDD100K road-object detection task. BDD100K defines the
# detection set as person, rider, car, truck, bus, train, motorcycle,
# bicycle, traffic light, traffic sign. We deliberately import only the
# generic road-user portion (the first eight).
#
# IMPORTANT: this ordering matches the BDD100K/YOLOPv2 detection taxonomy.
YOLOPV2_CLASSES = (
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

YOLOPV2_IMPORTED_CLASS_IDS = tuple(range(8))
YOLOPV2_EXCLUDED_CLASS_IDS = (8, 9)
YOLOPV2_EXCLUDED_CLASSES = frozenset({
    "traffic light",
    "traffic sign",
})

# Generic classes owned by this auxiliary model.
YOLOPV2_GENERIC_CLASSES = frozenset({
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
})


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def model_root() -> Path:
    return project_root() / "annotation_pipeline" / "models" / "yolopv2"


def repo_root() -> Path:
    return model_root() / "yolopv2_repo"


def weights_path() -> Path:
    return model_root() / "yolopv2.pt"


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    logger.info("Downloading YOLOPv2 weights from official release.")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            if total:
                logger.info(
                    "YOLOPv2 weights: {:.1f}/{:.1f} MiB",
                    downloaded / 1024**2,
                    total / 1024**2,
                )

    partial.replace(destination)
    return destination


def ensure_repo() -> Path:
    repo = repo_root()
    repo.parent.mkdir(parents=True, exist_ok=True)

    if (repo / "demo.py").is_file() and (repo / "utils" / "utils.py").is_file():
        return repo

    import subprocess

    logger.info("Cloning official YOLOPv2 repository.")
    subprocess.run(
        ["git", "clone", "--depth", "1", OFFICIAL_REPO_URL, str(repo)],
        check=True,
    )
    return repo


def ensure_weights() -> Path:
    custom = os.environ.get("YOLOPV2_WEIGHTS")
    if custom:
        path = Path(custom).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLOPV2_WEIGHTS does not exist: {path}"
            )
        return path

    return _download(
        OFFICIAL_WEIGHTS_URL,
        weights_path(),
    )


def ensure_assets() -> tuple[Path, Path]:
    return ensure_repo(), ensure_weights()


def _import_official_utils(repo: Path):
    """Import only the inference utilities from the official repository."""
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from utils.utils import (
        driving_area_mask,
        lane_line_mask,
        letterbox,
        non_max_suppression,
        scale_coords,
        split_for_trace_model,
    )

    return (
        driving_area_mask,
        lane_line_mask,
        letterbox,
        non_max_suppression,
        scale_coords,
        split_for_trace_model,
    )


def _json_safe(value: Any) -> Any:
    """Recursively convert Torch/NumPy values into JSON-serializable types."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class YOLOPv2SegmentationEngine:
    """Run official YOLOPv2 detection + road/lane segmentation."""

    def __init__(
        self,
        output_dir: Path,
        weights: str | Path | None = None,
        device: str | None = None,
        image_size: int = 640,
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        overlay_alpha: int = 90,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.mask_dir = self.output_dir / "masks"

        custom_weights = (
            str(weights)
            if weights is not None
            else os.environ.get("YOLOPV2_WEIGHTS")
        )
        self.weights = (
            Path(custom_weights).expanduser()
            if custom_weights
            else None
        )

        self.device_name = (
            device
            or os.environ.get("YOLOPV2_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "YOLOPV2 requested CUDA but CUDA is unavailable. Using CPU."
            )
            self.device_name = "cpu"

        self.device = torch.device(self.device_name)
        self.image_size = int(image_size)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.overlay_alpha = int(overlay_alpha)

        self.model = None
        self._official_utils = None

    def load(self) -> None:
        if self.model is not None:
            return

        repo = ensure_repo()
        if self.weights is None:
            self.weights = ensure_weights()

        self._official_utils = _import_official_utils(repo)

        self.model = torch.jit.load(
            str(self.weights),
            map_location=self.device,
        )
        self.model.to(self.device)

        # Follow the official demo: FP16 is used on CUDA.
        if self.device.type != "cpu":
            self.model.half()

        self.model.eval()

        logger.info(
            "YOLOPv2 loaded | weights='{}' | device='{}'",
            self.weights,
            self.device,
        )

    def unload(self) -> None:
        self.model = None
        self._official_utils = None
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Safely move Torch tensors to CPU before NumPy/OpenCV processing."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _binary_mask(cls, mask: Any) -> np.ndarray:
        # YOLOPv2 may return mask tensors on CUDA. NumPy cannot consume a
        # CUDA tensor directly, so always detach and move tensors to CPU.
        return cls._to_numpy(mask).astype(np.uint8) > 0

    def _save_binary_mask(self, mask: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            (self._binary_mask(mask).astype(np.uint8) * 255),
            mode="L",
        ).save(path)

    def _render(
        self,
        image_bgr: np.ndarray,
        detections: list[dict[str, Any]],
        drivable_mask: np.ndarray,
        lane_mask: np.ndarray,
        output_path: Path,
    ) -> None:
        rgb = Image.fromarray(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        ).convert("RGBA")

        pixels = np.zeros(
            (rgb.height, rgb.width, 4),
            dtype=np.uint8,
        )

        # These are auxiliary semantic masks, not object-instance masks.
        pixels[drivable_mask > 0] = [80, 200, 120, self.overlay_alpha]
        pixels[lane_mask > 0] = [255, 210, 60, self.overlay_alpha]

        overlay = Image.fromarray(pixels, mode="RGBA")
        rendered = Image.alpha_composite(rgb, overlay)

        draw = ImageDraw.Draw(rendered)

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            label = (
                f'{detection["class_name"]} '
                f'{float(detection["score"]):.2f}'
            )

            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(255, 80, 80, 255),
                width=2,
            )

            draw.text(
                (x1, max(0, y1 - 14)),
                label,
                fill=(255, 80, 80, 255),
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.convert("RGB").save(output_path)

    def _prepare_image(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        (
            _driving_area_mask,
            _lane_line_mask,
            letterbox,
            _nms,
            _scale_coords,
            _split_for_trace_model,
        ) = self._official_utils

        # The official YOLOPv2 LoadImages preprocessing first normalizes
        # frames to 1280x720, then letterboxes to 640.
        model_source = cv2.resize(
            image_bgr,
            (1280, 720),
            interpolation=cv2.INTER_LINEAR,
        )

        padded = letterbox(
            model_source,
            self.image_size,
            stride=32,
        )[0]

        rgb = padded[:, :, ::-1].transpose(2, 0, 1)
        rgb = np.ascontiguousarray(rgb)

        tensor = torch.from_numpy(rgb).to(self.device)
        tensor = tensor.half() if self.device.type != "cpu" else tensor.float()
        tensor /= 255.0

        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        return tensor, (model_source.shape[1], model_source.shape[0])

    def process_image(
        self,
        image_path: Path,
    ) -> list[dict[str, Any]]:
        self.load()

        (
            driving_area_mask,
            lane_line_mask,
            _letterbox,
            non_max_suppression,
            scale_coords,
            split_for_trace_model,
        ) = self._official_utils

        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            raise FileNotFoundError(
                f"YOLOPv2 could not read image: {image_path}"
            )

        original_height, original_width = image_bgr.shape[:2]
        tensor, model_source_size = self._prepare_image(image_bgr)

        with torch.inference_mode():
            # Exact output structure used by official demo.py:
            # [pred, anchor_grid], seg, ll
            [pred, anchor_grid], seg, ll = self.model(tensor)

        pred = split_for_trace_model(
            pred,
            anchor_grid,
        )

        predictions = non_max_suppression(
            pred,
            self.confidence_threshold,
            self.iou_threshold,
        )

        detections: list[dict[str, Any]] = []

        if predictions:
            det = predictions[0]

            if len(det):
                # First scale from 640-model coordinates back to the official
                # 1280x720 model source, exactly as the official demo does.
                det[:, :4] = scale_coords(
                    tensor.shape[2:],
                    det[:, :4],
                    (model_source_size[1], model_source_size[0]),
                ).round()

                # Then map 1280x720 coordinates back to the actual input image
                # if it was not already 1280x720.
                sx = original_width / model_source_size[0]
                sy = original_height / model_source_size[1]

                for row in det.detach().float().cpu().numpy():
                    x1, y1, x2, y2, score, class_id = row[:6]

                    class_id = int(class_id)

                    if class_id not in YOLOPV2_IMPORTED_CLASS_IDS:
                        # Traffic light/sign are deliberately excluded.
                        continue

                    label = YOLOPV2_CLASSES[class_id]

                    x1 *= sx
                    x2 *= sx
                    y1 *= sy
                    y2 *= sy

                    x1 = max(0.0, min(float(original_width), x1))
                    y1 = max(0.0, min(float(original_height), y1))
                    x2 = max(0.0, min(float(original_width), x2))
                    y2 = max(0.0, min(float(original_height), y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    detections.append({
                        "source": "yolopv2",
                        "class_name": label,
                        "class_id": class_id,
                        "score": float(score),
                        "bbox": [
                            int(round(x1)),
                            int(round(y1)),
                            int(round(x2)),
                            int(round(y2)),
                        ],
                        "bbox_normalized": [
                            x1 / original_width,
                            y1 / original_height,
                            x2 / original_width,
                            y2 / original_height,
                        ],
                        "object_group": (
                            "road_user"
                            if label in {"person", "rider"}
                            else "vehicle"
                        ),
                        "annotation_postprocessing_applied": False,
                        "semantic_verification_applied": False,
                    })

        # Official YOLOPv2 mask decoding.
        da_mask = self._to_numpy(driving_area_mask(seg))
        ll_mask = self._to_numpy(lane_line_mask(ll))

        # Official mask functions produce the model-space masks. The official
        # demo overlays these on its 1280x720 frame. Map them to our actual
        # input dimensions for saved outputs.
        da_mask = cv2.resize(
            da_mask.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        ll_mask = cv2.resize(
            ll_mask.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        drivable_path = self.mask_dir / f"{stem}_drivable.png"
        lane_path = self.mask_dir / f"{stem}_lane.png"
        visualization_path = (
            self.output_dir / f"{stem}_annotated.png"
        )
        result_path = self.results_dir / f"{stem}.json"

        self._save_binary_mask(
            da_mask,
            drivable_path,
        )
        self._save_binary_mask(
            ll_mask,
            lane_path,
        )

        self._render(
            image_bgr=image_bgr,
            detections=detections,
            drivable_mask=da_mask,
            lane_mask=ll_mask,
            output_path=visualization_path,
        )

        payload = {
            "image": str(image_path),
            "image_name": image_path.name,
            "model": "YOLOPv2",
            "weights": str(self.weights),
            "device": str(self.device),
            "object_detection_closed_set": list(YOLOPV2_CLASSES),
            "imported_detection_classes": [
                YOLOPV2_CLASSES[i]
                for i in YOLOPV2_IMPORTED_CLASS_IDS
            ],
            "excluded_detection_classes": [
                YOLOPV2_CLASSES[i]
                for i in YOLOPV2_EXCLUDED_CLASS_IDS
            ],
            "detections": detections,
            "drivable_area_mask": str(drivable_path),
            "lane_mask": str(lane_path),
            "visualization_path": str(visualization_path),
            "note": (
                "YOLOPv2 detections are auxiliary road-user evidence. "
                "Traffic signs, traffic signals, specialized vehicles and "
                "other out-of-set objects remain owned by the VLM/Locate "
                "Anything branch. Drivable/lane masks are semantic masks."
            ),
        }

        # Convert NumPy/Torch values to native Python values before JSON.
        payload = _json_safe(payload)

        result_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "YOLOPv2 | image='{}' | detections={} | "
            "drivable_pixels={} | lane_pixels={}",
            image_path.name,
            len(detections),
            int((da_mask > 0).sum()),
            int((ll_mask > 0).sum()),
        )

        return detections

    def process_images(
        self,
        image_paths: list[Path],
    ) -> dict[str, list[dict[str, Any]]]:
        self.load()

        results: dict[str, list[dict[str, Any]]] = {}

        try:
            for image_path in image_paths:
                results[Path(image_path).name] = self.process_image(
                    Path(image_path)
                )
        finally:
            self.unload()

        return results

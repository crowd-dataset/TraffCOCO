"""
detector_proposals.py

Object boxes from two street-scene detectors, used as proposals next to
the VLM -> LocateAnything boxes.

LocateAnything finds few instances per prompt: on 200 human-labelled
Mapillary Vistas validation images (2026-11-22) its raw boxes covered 26%
of the labelled objects, and the exported boxes reached 71% precision but
24% recall. Two detectors trained for this find far more:

* YOLO11x (COCO) for cars, people, vehicle traffic lights and motorcycles;
* Mask2Former panoptic (Mapillary Vistas) for traffic signs, trucks,
  buses, billboards, riders, crosswalks and manholes, which COCO lacks.

Taking these groups from the detectors (classes and thresholds below)
raised F1 on 160 held-out Vistas images from 0.36 to 0.54 (precision 67%,
recall 46%; 0.49 -> 0.65 for objects at least 16 px across). The Vistas
model was trained on Vistas, so on CROWD frames it was also compared with
the reviewed boxes of the fifth global run: it found 68% of the boxes
reviewers marked right and 31% of those marked wrong, and about 70% of the
boxes it added were real objects the pipeline had missed.

Each proposal carries an ontology class (a general one, such as ``car`` or
``unreadable_traffic_sign``); the conflict check later asks the scene VLM
for the class, light state or sign type, as for any other box.

The Vistas model's hub config resizes every image to 384 x 384, which
loses most signs, lights and people in a street scene; images are run at
up to ``panoptic_max_side`` pixels on the long side instead.

Outputs (under ``outputs/detector_proposals``)
----------------------------------------------
<stem>.json   the proposals of one image (reused when the stage runs again)
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


# COCO class name -> (ontology class, minimum confidence). Thresholds are
# the per-group best F1 on 40 Vistas dev images.
DEFAULT_YOLO_CLASSES: dict[str, tuple[str, float]] = {
    "car": ("car", 0.35),
    "person": ("pedestrian", 0.15),
    "traffic light": ("traffic_light_3_phase", 0.15),
    "motorcycle": ("motorcycle", 0.25),
}

# Vistas v1.2 class name -> (ontology class, minimum score). Street lights
# are left out: Vistas labels the lamp head only, the pipeline boxes the
# whole lamp post.
DEFAULT_PANOPTIC_CLASSES: dict[str, tuple[str, float]] = {
    "Traffic Sign (Front)": ("unreadable_traffic_sign", 0.7),
    "Truck": ("truck", 0.7),
    "Bus": ("city_bus", 0.7),
    "Billboard": ("billboard", 0.5),
    "Banner": ("billboard", 0.5),
    "Bicyclist": ("cyclist", 0.5),
    "Motorcyclist": ("motorcyclist", 0.5),
    "Crosswalk - Plain": ("zebra_crossing", 0.5),
    "Lane Marking - Crosswalk": ("zebra_crossing", 0.5),
    "Manhole": ("manhole_cover", 0.9),
}


def proposal_output_dir(outputs_dir: Path) -> Path:

    return Path(outputs_dir) / "detector_proposals"


def _class_table(raw: Any, default: dict[str, tuple[str, float]]) -> dict[str, tuple[str, float]]:
    """Config table {detector class: [ontology class, threshold]}; empty = default."""

    if not raw:
        return dict(default)
    return {str(k): (str(v[0]), float(v[1])) for k, v in dict(raw).items()}


class DetectorProposalEngine:
    """YOLO11x and Vistas panoptic Mask2Former boxes for one run."""

    def __init__(
        self,
        output_dir: Path,
        yolo_weights: str = "yolo11x.pt",
        yolo_image_size: int = 1920,
        yolo_classes: dict | None = None,
        panoptic_model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
        panoptic_max_side: int = 1920,
        panoptic_classes: dict | None = None,
        device: str | None = None,
    ) -> None:

        self.output_dir = Path(output_dir)
        self.yolo_weights = yolo_weights
        self.yolo_image_size = int(yolo_image_size)
        self.yolo_classes = _class_table(yolo_classes, DEFAULT_YOLO_CLASSES)
        self.panoptic_model_id = panoptic_model_id
        self.panoptic_max_side = int(panoptic_max_side)
        self.panoptic_classes = _class_table(panoptic_classes, DEFAULT_PANOPTIC_CLASSES)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Detectors (each loaded, run over all images and released in turn)
    # ------------------------------------------------------------------

    def _run_yolo(self, image_paths: list[Path]) -> dict[str, list[dict]]:

        from ultralytics import YOLO

        model = YOLO(self.yolo_weights)
        names = model.names
        wanted = {i for i, n in names.items() if n in self.yolo_classes}
        lowest = min((t for _, t in self.yolo_classes.values()), default=0.25)
        results: dict[str, list[dict]] = {}

        try:
            for path in image_paths:
                prediction = model.predict(
                    str(path),
                    imgsz=self.yolo_image_size,
                    conf=lowest,
                    classes=sorted(wanted),
                    device=self.device,
                    verbose=False,
                )[0]
                boxes = []
                for xyxy, cls, conf in zip(
                    prediction.boxes.xyxy.tolist(),
                    prediction.boxes.cls.tolist(),
                    prediction.boxes.conf.tolist(),
                ):
                    label = names[int(cls)]
                    class_name, threshold = self.yolo_classes[label]
                    if conf < threshold:
                        continue
                    boxes.append({
                        "bbox": [round(float(v), 1) for v in xyxy],
                        "class_name": class_name,
                        "detector": "yolo11x",
                        "detector_label": label,
                        "detector_confidence": round(float(conf), 3),
                    })
                results[path.name] = boxes
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _panoptic_size(self, width: int, height: int) -> dict[str, int]:
        """Long side at most panoptic_max_side, multiples of 32."""

        scale = min(1.0, self.panoptic_max_side / max(width, height))
        return {
            "height": max(32, int(round(height * scale / 32)) * 32),
            "width": max(32, int(round(width * scale / 32)) * 32),
        }

    def _run_panoptic(self, image_paths: list[Path]) -> dict[str, list[dict]]:

        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
        )

        processor = AutoImageProcessor.from_pretrained(self.panoptic_model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.panoptic_model_id,
        ).to(self.device).eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        lowest = min((t for _, t in self.panoptic_classes.values()), default=0.5)
        results: dict[str, list[dict]] = {}

        try:
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                width, height = image.size
                inputs = processor(
                    images=image,
                    size=self._panoptic_size(width, height),
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                panoptic = processor.post_process_panoptic_segmentation(
                    outputs,
                    threshold=lowest,
                    label_ids_to_fuse=set(),
                    target_sizes=[(height, width)],
                )[0]
                segmentation = panoptic["segmentation"]
                boxes = []
                for segment in panoptic["segments_info"]:
                    label = id2label.get(int(segment["label_id"]), "")
                    if label not in self.panoptic_classes:
                        continue
                    class_name, threshold = self.panoptic_classes[label]
                    score = float(segment["score"])
                    if score < threshold:
                        continue
                    ys, xs = torch.nonzero(segmentation == segment["id"], as_tuple=True)
                    if len(xs) == 0:
                        continue
                    boxes.append({
                        "bbox": [
                            float(xs.min()), float(ys.min()),
                            float(xs.max() + 1), float(ys.max() + 1),
                        ],
                        "class_name": class_name,
                        "detector": "mask2former_vistas_panoptic",
                        "detector_label": label,
                        "detector_confidence": round(score, 3),
                    })
                results[path.name] = boxes
        finally:
            del model, processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def process_images(self, image_paths: list[Path]) -> dict[str, list[dict]]:
        """Return {image name: proposals}; cached per image under output_dir."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        proposals: dict[str, list[dict]] = {}
        todo = []

        for path in image_paths:
            cached = self.output_dir / f"{Path(path).stem}.json"
            if cached.exists():
                try:
                    proposals[path.name] = json.loads(cached.read_text())["proposals"]
                    continue
                except (OSError, ValueError, KeyError):
                    pass
            todo.append(Path(path))

        if todo:
            start = time.perf_counter()
            yolo = self._run_yolo(todo) if self.yolo_classes else {}
            panoptic = self._run_panoptic(todo) if self.panoptic_classes else {}
            for path in todo:
                boxes = yolo.get(path.name, []) + panoptic.get(path.name, [])
                proposals[path.name] = boxes
                (self.output_dir / f"{path.stem}.json").write_text(json.dumps({
                    "image": path.name,
                    "proposals": boxes,
                }, indent=1))
            logger.info(
                "Detector proposals for {} image(s) in {:.0f} s ({} cached).",
                len(todo),
                time.perf_counter() - start,
                len(image_paths) - len(todo),
            )

        return proposals

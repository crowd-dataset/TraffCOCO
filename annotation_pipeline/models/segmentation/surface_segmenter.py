"""
surface_segmenter.py

Semantic segmentation of road surfaces and roadside structure.

The VLM -> LocateAnything -> SAM 2 chain finds objects (cars, signs,
lights) but not the large regions they stand on: road, sidewalk, curb,
crosswalk, bike lane, terrain. A box around "road" spans the whole frame
and gives SAM 2 nothing to separate. This stage labels those regions per
pixel with a Mask2Former model trained on Mapillary Vistas (street-level
images from around the world, many from dashcams).

Outputs (under ``outputs/surface_segmentation``)
------------------------------------------------
masks/<stem>_surface.png   uint8 label map: 0 = not a surface class,
                           i = classes[i - 1]
classes.json               the class list that the mask values index
<stem>_surface.png         colour overlay for a quick look

The FINAL image draws these regions under the object detections, and
``COCO_FORMAT/annotations/surfaces_final.json`` stores them as COCO-Stuff
style annotations (one mask per class per image) next to the object file.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


# Mapillary Vistas classes that describe the road and its surroundings and
# are not already produced as objects by the grounding + SAM 2 chain.
# Sky, buildings, mountains, the ego vehicle and the object classes
# (vehicles, people, lights, signs, poles) are left out.
DEFAULT_SURFACE_CLASSES: tuple[str, ...] = (
    "Road",
    "Sidewalk",
    "Curb",
    "Curb Cut",
    "Bike Lane",
    "Service Lane",
    "Parking",
    "Pedestrian Area",
    "Crosswalk - Plain",
    "Lane Marking - Crosswalk",
    "Lane Marking - General",
    "Rail Track",
    "Manhole",
    "Catch Basin",
    "Pothole",
    "Guard Rail",
    "Barrier",
    "Fence",
    "Wall",
    "Bridge",
    "Tunnel",
    "Terrain",
    "Vegetation",
    "Sand",
    "Snow",
    "Water",
)

# Ontology road-marking classes that are lines or painted areas with no
# clear object boundary. Grounding boxes for them were mostly fragments
# of a line, one box over a whole hatched area, or a railing (pilot check,
# 2026-11-18: 13 of 16 marking boxes wrong), while the surface masks label
# the same paint correctly ("Lane Marking - General"). When surface
# segmentation runs, these boxes are dropped and the paint is taken from
# the surface masks. Compact painted symbols (arrows, crossings, stop and
# give-way lines, bicycle/bus/diamond symbols) stay object annotations.
DEFAULT_SURFACE_MARKING_CLASSES: tuple[str, ...] = (
    "lane_divider_broken",
    "lane_divider_solid",
    "no_passing_marking",
    "chevron_marking",
    "hatched_marking",
    "diagonal_stripes",
    "parking_bay_marking",
    "no_parking_zone_marking",
    "rumble_strip_marking",
)


def _class_key(name: Any) -> str:

    return str(name or "").strip().lower().replace(" ", "_")


def remove_surface_covered_markings(
    detections_by_image: dict[str, list[dict[str, Any]]],
    classes: tuple[str, ...] | list[str] = (),
) -> int:
    """Drop line/area marking boxes that the surface masks cover.

    Changes ``detections_by_image`` in place and returns how many boxes
    were removed. ``classes`` defaults to DEFAULT_SURFACE_MARKING_CLASSES.
    """

    drop = {_class_key(c) for c in (classes or DEFAULT_SURFACE_MARKING_CLASSES)}
    total = 0

    for image_name, detections in list(detections_by_image.items()):
        if not isinstance(detections, list):
            continue

        kept = [
            detection
            for detection in detections
            if not (
                isinstance(detection, dict)
                and _class_key(
                    detection.get("class_name") or detection.get("final_label")
                ) in drop
            )
        ]
        removed = len(detections) - len(kept)

        if removed:
            logger.info(
                "Removed {} line/area marking box(es) from '{}'; these "
                "markings come from the surface masks.",
                removed,
                image_name,
            )

        detections_by_image[image_name] = kept
        total += removed

    return total


# Ontology classes for barriers that stand up from the ground. Grounding
# sometimes puts a "guardrail" box on a curb or sidewalk edge (pilot check,
# 2026-11-18: 3 of 5 guardrail boxes in a Japanese frame lay on the curb,
# with 71-99% of their SAM mask on Curb or Sidewalk pixels, while the real
# rails were 82-99% Guard Rail or Fence). road_median is left out because
# a median can be painted flat on the road.
DEFAULT_UPRIGHT_BARRIER_CLASSES: tuple[str, ...] = (
    "guardrail",
    "guard_rail",
    "barricade",
    "construction_barrier",
    "water_filled_barrier",
    "crash_cushion",
)

# Surface classes that are flat ground: an upright barrier's mask should
# not lie mostly on these.
FLAT_GROUND_SURFACES: tuple[str, ...] = (
    "Road",
    "Sidewalk",
    "Curb",
    "Curb Cut",
    "Bike Lane",
    "Service Lane",
    "Parking",
    "Pedestrian Area",
    "Crosswalk - Plain",
    "Lane Marking - Crosswalk",
    "Lane Marking - General",
    "Rail Track",
    "Manhole",
    "Catch Basin",
    "Pothole",
)

FLAT_GROUND_MAX_FRACTION = 0.6


def remove_barriers_on_flat_ground(
    detections_by_image: dict[str, list[dict[str, Any]]],
    outputs_dir: Path,
    classes: tuple[str, ...] | list[str] = (),
    max_fraction: float = FLAT_GROUND_MAX_FRACTION,
) -> list[str]:
    """Drop upright-barrier boxes whose SAM mask lies on flat ground.

    A box is dropped when at least ``max_fraction`` of its SAM 2 mask
    falls on road, sidewalk, curb or similar surface pixels. Boxes without
    a SAM mask or surface map are kept. Changes ``detections_by_image`` in
    place and returns the names of the images that lost a box.
    """

    barrier_classes = {
        _class_key(c) for c in (classes or DEFAULT_UPRIGHT_BARRIER_CLASSES)
    }
    surface_classes = load_surface_classes(outputs_dir)
    flat_values = np.array(
        [
            index
            for index, name in enumerate(surface_classes, start=1)
            if name in FLAT_GROUND_SURFACES
        ],
        dtype=np.uint8,
    )
    changed: list[str] = []

    if flat_values.size == 0:
        return changed

    for image_name, detections in list(detections_by_image.items()):
        if not isinstance(detections, list):
            continue

        label_path = surface_mask_path(outputs_dir, Path(image_name).stem)
        label_map = None
        kept = []

        for detection in detections:
            class_name = _class_key(
                detection.get("class_name") or detection.get("final_label")
            ) if isinstance(detection, dict) else ""
            mask_path = (
                (detection.get("segmentation") or {}).get("mask_path")
                if class_name in barrier_classes
                else None
            )

            if not mask_path or not Path(mask_path).exists():
                kept.append(detection)
                continue

            if label_map is None:
                if not label_path.exists():
                    kept.append(detection)
                    continue
                label_map = np.asarray(Image.open(label_path))

            with Image.open(mask_path) as mask_image:
                if mask_image.size != (label_map.shape[1], label_map.shape[0]):
                    mask_image = mask_image.resize(
                        (label_map.shape[1], label_map.shape[0]),
                        Image.NEAREST,
                    )
                mask = np.asarray(mask_image.convert("L")) > 127

            area = int(mask.sum())
            if area == 0:
                kept.append(detection)
                continue

            fraction = float(np.isin(label_map[mask], flat_values).mean())
            detection["flat_ground_fraction"] = round(fraction, 3)

            if fraction >= max_fraction:
                logger.info(
                    "Removed {} box {} from '{}': {:.0%} of its mask is "
                    "road, sidewalk or curb.",
                    class_name,
                    detection.get("bbox"),
                    image_name,
                    fraction,
                )
                continue

            kept.append(detection)

        if len(kept) != len(detections):
            changed.append(image_name)
        detections_by_image[image_name] = kept

    return changed


# Display colours (RGB). Unlisted classes get a colour from their name.
_SURFACE_COLOURS: dict[str, tuple[int, int, int]] = {
    "Road": (128, 64, 128),
    "Sidewalk": (244, 35, 232),
    "Curb": (196, 196, 196),
    "Curb Cut": (170, 170, 170),
    "Bike Lane": (128, 64, 255),
    "Service Lane": (110, 110, 110),
    "Parking": (250, 170, 160),
    "Pedestrian Area": (96, 96, 96),
    "Crosswalk - Plain": (140, 140, 200),
    "Lane Marking - Crosswalk": (255, 255, 255),
    "Lane Marking - General": (255, 255, 0),
    "Rail Track": (230, 150, 140),
    "Manhole": (100, 128, 160),
    "Catch Basin": (220, 128, 128),
    "Pothole": (70, 100, 150),
    "Guard Rail": (180, 165, 180),
    "Barrier": (90, 120, 150),
    "Fence": (190, 153, 153),
    "Wall": (102, 102, 156),
    "Bridge": (150, 100, 100),
    "Tunnel": (150, 120, 90),
    "Terrain": (152, 251, 152),
    "Vegetation": (107, 142, 35),
    "Sand": (210, 170, 100),
    "Snow": (255, 255, 255),
    "Water": (0, 170, 30),
}


def surface_colour(name: str) -> tuple[int, int, int]:
    """Return the display colour of a surface class."""

    if name in _SURFACE_COLOURS:
        return _SURFACE_COLOURS[name]

    seed = sum(ord(c) * (i + 1) for i, c in enumerate(name))
    rng = np.random.default_rng(seed)

    return tuple(int(v) for v in rng.integers(60, 230, size=3))


def surface_output_dir(outputs_dir: Path) -> Path:
    """Directory that holds the surface segmentation outputs."""

    return Path(outputs_dir) / "surface_segmentation"


def surface_mask_path(outputs_dir: Path, stem: str) -> Path:
    """Path of the uint8 label map for one image."""

    return surface_output_dir(outputs_dir) / "masks" / f"{stem}_surface.png"


def load_surface_classes(outputs_dir: Path) -> list[str]:
    """Return the class list written by the last surface run, or []."""

    path = surface_output_dir(outputs_dir) / "classes.json"

    if not path.exists():
        return []

    return list(json.loads(path.read_text(encoding="utf-8")))


class SurfaceSegmenter:
    """Mask2Former (Mapillary Vistas) surface segmentation."""

    def __init__(
        self,
        model_id: str,
        classes: tuple[str, ...] | list[str] = (),
        max_side: int = 1280,
        device: str | None = None,
    ) -> None:

        self.model_id = model_id
        self.requested_classes = tuple(classes) or DEFAULT_SURFACE_CLASSES
        self.max_side = int(max_side)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.processor = None

        self.classes: list[str] = []
        self._lookup: np.ndarray | None = None

        self.last_inference_time = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model and build the model-id -> output-index lookup."""

        from transformers import (
            AutoImageProcessor,
            Mask2FormerForUniversalSegmentation,
        )

        logger.info(
            "Loading surface segmentation model '{}' on {}.",
            self.model_id,
            self.device,
        )

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_id,
        ).to(self.device).eval()

        id2label = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        by_name = {v.lower(): k for k, v in id2label.items()}

        missing = [
            c for c in self.requested_classes if c.lower() not in by_name
        ]

        if missing:
            logger.warning(
                "Surface classes not in model '{}', ignored: {}",
                self.model_id,
                ", ".join(missing),
            )

        self.classes = [
            id2label[by_name[c.lower()]]
            for c in self.requested_classes
            if c.lower() in by_name
        ]

        if len(self.classes) > 254:
            raise ValueError("At most 254 surface classes fit in a uint8 mask.")

        lookup = np.zeros(max(id2label) + 1, dtype=np.uint8)

        for index, name in enumerate(self.classes, start=1):
            lookup[by_name[name.lower()]] = index

        self._lookup = lookup

        logger.info(
            "Surface segmentation keeps {} class(es): {}",
            len(self.classes),
            ", ".join(self.classes),
        )

    def unload(self) -> None:
        """Release the model and GPU memory."""

        self.model = None
        self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _input_size(self, width: int, height: int) -> dict[str, int]:
        """Resize so the long side is at most max_side, multiples of 32."""

        scale = min(1.0, self.max_side / max(width, height))

        return {
            "height": max(32, int(round(height * scale / 32)) * 32),
            "width": max(32, int(round(width * scale / 32)) * 32),
        }

    def segment(self, image_path: Path) -> np.ndarray:
        """Return the uint8 surface label map of one image (H x W)."""

        if self.model is None or self._lookup is None:
            raise RuntimeError("SurfaceSegmenter.load() must be called first.")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        inputs = self.processor(
            images=image,
            size=self._input_size(width, height),
            return_tensors="pt",
        ).to(self.device)

        start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(**inputs)

        semantic = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(height, width)],
        )[0]

        self.last_inference_time = time.perf_counter() - start

        model_ids = semantic.to("cpu").numpy().astype(np.int64)

        return self._lookup[model_ids]

    def save(
        self,
        label_map: np.ndarray,
        image_path: Path,
        outputs_dir: Path,
        save_overlay: bool = True,
    ) -> Path:
        """Write the label map, the class list and an optional overlay."""

        out_dir = surface_output_dir(outputs_dir)
        mask_path = surface_mask_path(outputs_dir, image_path.stem)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        (out_dir / "classes.json").write_text(
            json.dumps(self.classes, indent=2),
            encoding="utf-8",
        )

        Image.fromarray(label_map, mode="L").save(mask_path)

        if save_overlay:
            base = Image.open(image_path).convert("RGBA")
            overlay = overlay_surfaces(base, mask_path, self.classes)
            overlay.convert("RGB").save(
                out_dir / f"{image_path.stem}_surface.png"
            )

        return mask_path


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def _label_points(
    binary: np.ndarray,
    min_fraction: float,
) -> list[tuple[int, int]]:
    """Points deep inside each large enough region, for text labels."""

    total = binary.size
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )

    points = []

    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < min_fraction * total:
            continue

        region = (components == component).astype(np.uint8)
        distance = cv2.distanceTransform(region, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        points.append((int(x), int(y)))

    return points


def overlay_surfaces(
    base: Image.Image,
    mask_path: Path,
    classes: list[str],
    alpha: int = 110,
    label_min_fraction: float = 0.004,
) -> Image.Image:
    """Tint surface regions on ``base`` and name each large region."""

    if not mask_path.exists() or not classes:
        return base

    labels = np.array(Image.open(mask_path))
    base = base.convert("RGBA")

    if labels.shape != (base.height, base.width):
        labels = np.array(
            Image.fromarray(labels).resize(base.size, Image.NEAREST)
        )

    colour = np.zeros((*labels.shape, 4), dtype=np.uint8)

    for index, name in enumerate(classes, start=1):
        colour[labels == index] = (*surface_colour(name), alpha)

    result = Image.alpha_composite(base, Image.fromarray(colour, "RGBA"))

    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()

    for index, name in enumerate(classes, start=1):
        for x, y in _label_points(labels == index, label_min_fraction):
            text = name.lower()
            left, top, right, bottom = draw.textbbox((x, y), text, font=font)
            draw.rectangle(
                (left - 2, top - 1, right + 2, bottom + 1),
                fill=(*surface_colour(name), 230),
            )
            draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)

    return result


# ----------------------------------------------------------------------
# COCO-Stuff export
# ----------------------------------------------------------------------


def _rle_counts(binary: np.ndarray) -> list[int]:
    """COCO uncompressed RLE counts (column-major, starting with zeros)."""

    flat = np.asarray(binary, dtype=np.uint8).T.reshape(-1)
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(bounds).tolist()

    if flat.size and flat[0] == 1:
        counts = [0] + counts

    return [int(c) for c in counts]


def export_surface_coco(
    outputs_dir: Path,
    coco_images: list[dict[str, Any]],
    out_path: Path,
    min_pixels: int = 64,
) -> dict[str, int]:
    """Write one COCO-Stuff style file for the surface masks.

    ``coco_images`` are the COCO ``images`` entries of the object export,
    so both files share image ids and file names. Each (image, class)
    pair with at least ``min_pixels`` pixels becomes one annotation.
    """

    classes = load_surface_classes(outputs_dir)

    categories = [
        {
            "id": index,
            "name": name.lower(),
            "supercategory": "surface",
            "isthing": 0,
            "color": list(surface_colour(name)),
        }
        for index, name in enumerate(classes, start=1)
    ]

    annotations = []
    images = []

    for image in coco_images:
        stem = Path(image["file_name"]).stem
        mask_path = surface_mask_path(outputs_dir, stem)

        if not mask_path.exists():
            continue

        labels = np.array(Image.open(mask_path))
        height, width = labels.shape

        if (height, width) != (image.get("height"), image.get("width")):
            labels = np.array(
                Image.fromarray(labels).resize(
                    (image["width"], image["height"]),
                    Image.NEAREST,
                )
            )
            height, width = labels.shape

        images.append(image)

        for index in range(1, len(classes) + 1):
            binary = labels == index
            area = int(binary.sum())

            if area < min_pixels:
                continue

            ys, xs = np.nonzero(binary)
            x0, y0 = int(xs.min()), int(ys.min())

            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image["id"],
                "category_id": index,
                "segmentation": {
                    "size": [int(height), int(width)],
                    "counts": _rle_counts(binary),
                },
                "area": area,
                "bbox": [
                    x0,
                    y0,
                    int(xs.max()) - x0 + 1,
                    int(ys.max()) - y0 + 1,
                ],
                "iscrowd": 0,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "info": {
                    "description": (
                        "TraffCOCO surface masks (Mask2Former, Mapillary "
                        "Vistas classes)"
                    ),
                },
                "images": images,
                "categories": categories,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )

    return {"images": len(images), "annotations": len(annotations)}

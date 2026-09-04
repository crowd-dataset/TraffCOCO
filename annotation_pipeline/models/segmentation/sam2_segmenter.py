"""
Final SAM 2.1 Hiera Tiny segmentation stage.

This module:
- bootstraps the official Meta SAM 2 repository under models/segmentation/sam2_repo;
- uses the project's .venv through uv without resolving its Python symlink;
- automatically downloads sam2.1_hiera_tiny.pt when missing;
- uses the final annotation bbox as an authoritative SAM box prompt;
- writes one binary mask per successful detection and a combined visualization;
- fails loudly when the pipeline gives SAM no detections, instead of silently
  producing an unchanged image.
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image, ImageDraw

from custom_logger import CustomLogger

logger = CustomLogger(__name__)

SAM2_REPO_URL = "https://github.com/facebookresearch/sam2.git"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
    "sam2.1_hiera_tiny.pt"
)


class SAM2SegmentationEngine:
    """Run SAM 2.1 Hiera Tiny with final pipeline bboxes as box prompts."""

    def __init__(
        self,
        output_dir: Path,
        checkpoint: str | Path | None = None,
        model_config: str | Path | None = None,
        device: str | None = None,
        overlay_alpha: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.masks_dir = self.output_dir / "masks"

        # sam2_segmenter.py lives at:
        # <project>/annotation_pipeline/models/segmentation/sam2_segmenter.py
        self.project_root = Path(__file__).resolve().parents[3]
        self.model_dir = self.project_root /  "annotation_pipeline" / "models" / "segmentation"
        self.sam2_repo = self.model_dir / "sam2_repo"

        self.model_dir.mkdir(parents=True, exist_ok=True)

        env_checkpoint = os.getenv("SAM2_CHECKPOINT", "").strip()
        self.checkpoint = Path(
            checkpoint
            or env_checkpoint
            or self.model_dir / "sam2.1_hiera_tiny.pt"
        ).expanduser()

        env_config = os.getenv("SAM2_CONFIG", "").strip()
        # build_sam2 expects a Hydra config name, not an arbitrary absolute
        # filesystem path. Keep the default as the official repo-relative name.
        self.config_name = (
            model_config
            or env_config
            or "configs/sam2.1/sam2.1_hiera_t.yaml"
        )
        self.config_name = str(self.config_name)

        requested_device = (
            device
            or os.getenv("SAM2_DEVICE", "").strip()
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "SAM2_DEVICE='{}' requested but CUDA is unavailable. Falling back to CPU.",
                requested_device,
            )
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.overlay_alpha = max(
            0,
            min(
                255,
                int(
                    overlay_alpha
                    if overlay_alpha is not None
                    else os.getenv("SAM2_OVERLAY_ALPHA", "110")
                ),
            ),
        )

        self.model = None
        self.predictor = None

        self._palette = [
            (255, 80, 80),
            (80, 180, 255),
            (90, 220, 120),
            (255, 190, 70),
            (190, 110, 255),
            (70, 220, 210),
            (255, 120, 190),
            (170, 220, 70),
        ]

    # ------------------------------------------------------------------
    # uv / installation
    # ------------------------------------------------------------------

    def _project_python(self) -> Path:
        """Return the project's literal .venv interpreter path.

        Do NOT call resolve() here. uv treats a symlinked .venv interpreter as
        part of the project environment, while the resolved target under
        ~/.local/share/uv/python is an externally managed interpreter.
        """
        candidates = (
            self.project_root / ".venv" / "bin" / "python3",
            self.project_root / ".venv" / "bin" / "python",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "TraffCOCO .venv was not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )

    def _uv_executable(self) -> str:
        uv = shutil.which("uv")
        if uv:
            return uv
        local_uv = Path.home() / ".local" / "bin" / "uv"
        if local_uv.exists():
            return str(local_uv)
        raise RuntimeError(
            "uv was not found. Install uv or make it available on PATH."
        )

    def _install_sam2(self) -> None:
        python = self._project_python()
        uv = self._uv_executable()

        if not self.sam2_repo.exists():
            logger.info(
                "SAM 2 repository not found. Cloning official Meta SAM 2 into '{}'",
                self.sam2_repo,
            )
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    SAM2_REPO_URL,
                    str(self.sam2_repo),
                ],
                check=True,
            )

        logger.info(
            "Installing SAM 2 into TraffCOCO project venv: {}",
            python,
        )

        # uv must be pointed at .venv/bin/python3 itself, not its resolved
        # ~/.local/share/uv/python/... target.
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "-e",
                str(self.sam2_repo),
            ],
            check=True,
            cwd=str(self.project_root),
        )

        # Make the repo importable in this already-running process too.
        repo_string = str(self.sam2_repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        importlib.invalidate_caches()

        try:
            import sam2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SAM 2 was installed by uv but cannot be imported in the "
                "current process. Repository: " + str(self.sam2_repo)
            ) from exc

        logger.info("SAM 2 installation verified.")

    def _ensure_sam2(self) -> None:
        repo_string = str(self.sam2_repo)
        if repo_string not in sys.path:
            sys.path.insert(0, repo_string)
        importlib.invalidate_caches()

        try:
            import sam2  # noqa: F401
            logger.info("SAM 2 package is already available.")
        except ImportError:
            self._install_sam2()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _download_checkpoint(self) -> None:
        if self.checkpoint.exists() and self.checkpoint.stat().st_size > 0:
            logger.info("SAM 2.1 checkpoint already exists: {}", self.checkpoint)
            return

        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint.with_suffix(self.checkpoint.suffix + ".part")

        logger.info(
            "Downloading SAM 2.1 Hiera Tiny checkpoint to {}",
            self.checkpoint,
        )

        request = Request(
            SAM2_CHECKPOINT_URL,
            headers={"User-Agent": "TraffCOCO-SAM2-bootstrap/1.0"},
        )

        try:
            with urlopen(request, timeout=60) as response, tmp.open("wb") as out:
                total = response.headers.get("Content-Length")
                total_bytes = int(total) if total else None
                downloaded = 0

                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = downloaded * 100 / total_bytes
                        logger.info(
                            "SAM 2.1 checkpoint download: {:.1f}% ({:.1f}/{:.1f} MB)",
                            pct,
                            downloaded / 1024**2,
                            total_bytes / 1024**2,
                        )

            tmp.replace(self.checkpoint)

        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

        if self.checkpoint.stat().st_size == 0:
            raise RuntimeError("SAM 2 checkpoint download produced an empty file.")

        logger.info("SAM 2.1 checkpoint downloaded successfully.")

    def _ensure_config(self) -> None:
        config_path = self.sam2_repo / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                "Official SAM 2.1 Hiera Tiny config was not found at "
                f"{config_path}. The cloned SAM 2 repository appears incomplete."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self.predictor is not None:
            return

        self._ensure_sam2()
        self._ensure_config()
        self._download_checkpoint()

        # The official SAM2 build API expects the Hydra config name
        # configs/sam2.1/sam2.1_hiera_t.yaml.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logger.info(
            "Loading SAM 2.1 Hiera Tiny: checkpoint='{}', config='{}', device='{}'",
            self.checkpoint,
            self.config_name,
            self.device,
        )

        self.model = build_sam2(
            self.config_name,
            str(self.checkpoint),
            device=str(self.device),
        )
        self.predictor = SAM2ImagePredictor(self.model)

        logger.info("SAM 2.1 Hiera Tiny loaded successfully.")

    def unload(self) -> None:
        self.predictor = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label(detection: dict[str, Any]) -> str:
        for key in (
            "class_name",
            "final_label",
            "object_name",
            "observed_object",
            "grounding_prompt",
        ):
            value = detection.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "unknown"

    @staticmethod
    def _valid_bbox(
        bbox: Any,
        width: int,
        height: int,
    ) -> tuple[list[float], str] | tuple[None, str]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None, "bbox_missing_or_wrong_length"

        try:
            values = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None, "bbox_non_numeric"

        if not all(np.isfinite(values)):
            return None, "bbox_non_finite"

        x1, y1, x2, y2 = values

        # Normalized coordinates are accepted defensively.
        if all(0.0 <= value <= 1.0 for value in values):
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        x1 = max(0.0, min(x1, float(width - 1)))
        y1 = max(0.0, min(y1, float(height - 1)))
        x2 = max(1.0, min(x2, float(width)))
        y2 = max(1.0, min(y2, float(height)))

        if x2 <= x1 or y2 <= y1:
            return None, "bbox_zero_or_negative_area"

        return [x1, y1, x2, y2], "ok"

    @staticmethod
    def _object_key(detection: dict[str, Any], index: int) -> str:
        object_id = detection.get("object_id")
        raw = str(object_id) if object_id is not None else str(index)
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in raw)
        return f"object_{safe}"

    # ------------------------------------------------------------------
    # SAM inference
    # ------------------------------------------------------------------

    def _predict_mask(
        self,
        bbox: list[float],
    ) -> tuple[np.ndarray, float]:
        if self.predictor is None:
            raise RuntimeError("SAM 2 predictor is not loaded.")

        box = np.asarray(bbox, dtype=np.float32)

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    masks, scores, _ = self.predictor.predict(
                        box=box,
                        multimask_output=False,
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False,
                )

        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM 2 returned no masks.")

        mask = np.asarray(masks[0]).astype(bool)
        area = int(mask.sum())

        if area <= 0:
            raise RuntimeError("SAM 2 returned an empty mask.")

        if scores is None or len(scores) == 0:
            score = 0.0
        else:
            score = float(np.asarray(scores).reshape(-1)[0])

        return mask, score

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_overlay(
        self,
        image: Image.Image,
        masks: list[np.ndarray],
        detections: list[dict[str, Any]],
    ) -> Image.Image:
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

        for index, (mask, detection) in enumerate(zip(masks, detections)):
            if mask is None:
                continue

            if mask.shape != (base.height, base.width):
                raise RuntimeError(
                    f"SAM mask shape {mask.shape} does not match image "
                    f"shape {(base.height, base.width)}."
                )

            color = self._palette[index % len(self._palette)]
            mask_image = Image.fromarray(
                (mask.astype(np.uint8) * 255),
                mode="L",
            )
            solid = Image.new(
                "RGBA",
                base.size,
                (*color, self.overlay_alpha),
            )
            overlay = Image.composite(solid, overlay, mask_image)

        # Draw authoritative bboxes after compositing.
        box_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(box_layer)

        for index, detection in enumerate(detections):
            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, bbox)
            except (TypeError, ValueError):
                continue

            color = self._palette[index % len(self._palette)]
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=(*color, 255),
                width=3,
            )

            label = self._label(detection)
            draw.text(
                (x1 + 3, max(0, y1 - 14)),
                label,
                fill=(*color, 255),
            )

        overlay = Image.alpha_composite(overlay, box_layer)
        return Image.alpha_composite(base, overlay)

    @staticmethod
    def _save_binary_mask(mask: np.ndarray, path: Path) -> None:
        Image.fromarray(
            (mask.astype(np.uint8) * 255),
            mode="L",
        ).save(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_image(
        self,
        image_path: Path,
        detections: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
        if self.predictor is None:
            raise RuntimeError("SAM 2 predictor is not loaded.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        width, height = image.size

        logger.info(
            "SAM 2 INPUT | image='{}' | final detections={}",
            image_path.name,
            len(detections),
        )

        if not detections:
            image.close()
            raise RuntimeError(
                f"SAM 2 received 0 final detections for '{image_path.name}'. "
                "The annotation stage did not provide anything to segment."
            )

        valid_detections: list[dict[str, Any]] = []
        valid_boxes: list[list[float]] = []
        records: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        for index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                skipped.append({"index": index, "reason": "not_a_dict"})
                continue

            bbox, reason = self._valid_bbox(
                detection.get("bbox"),
                width,
                height,
            )

            logger.info(
                "SAM 2 DETECTION {} | id={} | label='{}' | bbox={} | bbox_status={}",
                index + 1,
                detection.get("object_id", index),
                self._label(detection),
                detection.get("bbox"),
                reason,
            )

            if bbox is None:
                skipped.append(
                    {
                        "index": index,
                        "object_id": detection.get("object_id"),
                        "label": self._label(detection),
                        "reason": reason,
                    }
                )
                continue

            valid_detections.append(detection)
            valid_boxes.append(bbox)

        if not valid_boxes:
            image.close()
            raise RuntimeError(
                f"SAM 2 received {len(detections)} detections for "
                f"'{image_path.name}', but none had valid bboxes."
            )

        self.predictor.set_image(np.asarray(image))

        masks: list[np.ndarray] = []
        successful_detections: list[dict[str, Any]] = []

        for index, (detection, bbox) in enumerate(
            zip(valid_detections, valid_boxes)
        ):
            object_key = self._object_key(detection, index)

            try:
                mask, score = self._predict_mask(bbox)
                mask_path = self.masks_dir / f"{image_path.stem}_{object_key}.png"
                self._save_binary_mask(mask, mask_path)

                area = int(mask.sum())
                detection["segmentation"] = {
                    "mask_path": str(mask_path),
                    "score": score,
                    "mask_area_pixels": area,
                    "status": "segmented",
                    "bbox_prompt": bbox,
                }

                masks.append(mask)
                successful_detections.append(detection)
                records.append(
                    {
                        "object_id": detection.get("object_id"),
                        "class_name": self._label(detection),
                        "bbox": bbox,
                        "mask": str(mask_path),
                        "score": score,
                        "mask_area_pixels": area,
                        "status": "segmented",
                    }
                )

                logger.info(
                    "SAM 2 MASK {} | id={} | area={} px | score={:.4f}",
                    index + 1,
                    detection.get("object_id", index),
                    area,
                    score,
                )

            except Exception as exc:
                records.append(
                    {
                        "object_id": detection.get("object_id"),
                        "class_name": self._label(detection),
                        "bbox": bbox,
                        "mask": None,
                        "score": None,
                        "mask_area_pixels": 0,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                logger.error(
                    "SAM 2 FAILED | image='{}' | id={} | label='{}' | bbox={} | error={}",
                    image_path.name,
                    detection.get("object_id", index),
                    self._label(detection),
                    bbox,
                    exc,
                )

        if not successful_detections:
            image.close()
            raise RuntimeError(
                f"SAM 2 produced 0 successful masks for '{image_path.name}'. "
                "See per-detection SAM 2 FAILED messages above."
            )

        overlay_path = self.output_dir / f"{image_path.stem}_segmented.png"
        rendered = self._render_overlay(
            image=image,
            masks=masks,
            detections=successful_detections,
        )
        rendered.convert("RGB").save(overlay_path)

        json_path = self.output_dir / f"{image_path.stem}.json"
        payload = {
            "image": str(image_path),
            "image_name": image_path.name,
            "image_size": {"width": width, "height": height},
            "segmentation_model": {
                "name": "SAM 2.1 Hiera Tiny",
                "checkpoint": str(self.checkpoint),
                "config": self.config_name,
                "device": str(self.device),
            },
            "detections_input": len(detections),
            "detections_valid_bbox": len(valid_detections),
            "detections_segmented": len(successful_detections),
            "detections_failed": sum(r["status"] == "failed" for r in records),
            "skipped": skipped,
            "detections": records,
            "visualization_path": str(overlay_path),
        }
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        image.close()

        logger.info(
            "SAM 2 segmentation completed for '{}': {}/{} detection(s).",
            image_path.name,
            len(successful_detections),
            len(detections),
        )
        logger.info("SAM 2 visualization: {}", overlay_path)

        return detections, overlay_path, payload

    def process_images(
        self,
        image_paths: list[Path],
        detections_by_image: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        self.load()

        # Never silently manufacture an empty success. Log exactly what the
        # caller supplied for every requested image.
        requested = {}
        for image_path in image_paths:
            detections = detections_by_image.get(image_path.name)
            requested[image_path.name] = len(detections or [])
            logger.info(
                "SAM 2 FINAL INPUT | {} -> {} detection(s)",
                image_path.name,
                len(detections or []),
            )

        total = sum(requested.values())
        if total == 0:
            raise RuntimeError(
                "SAM 2 received zero final detections across all requested "
                f"images. Input keys={list(detections_by_image.keys())}; "
                f"requested images={[p.name for p in image_paths]}. "
                "This is a pipeline data-flow problem, not a SAM model problem."
            )

        processed = dict(detections_by_image)

        try:
            for image_path in image_paths:
                detections = processed.get(image_path.name, [])
                if not detections:
                    logger.warning(
                        "SAM 2: skipping '{}' because it has 0 final detections.",
                        image_path.name,
                    )
                    continue

                updated, _, _ = self.segment_image(
                    image_path=image_path,
                    detections=detections,
                )
                processed[image_path.name] = updated

        finally:
            self.unload()

        return processed

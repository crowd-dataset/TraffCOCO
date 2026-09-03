"""
semantic_verification.py

Standalone semantic verification / recovery stage.

Purpose
-------
Re-examine only detections whose current class is
``unreadable_traffic_sign``.

The existing Locate Anything bounding box is authoritative and is NEVER
recomputed here.

Workflow
--------
Saved annotation JSON
        |
        v
Find unreadable_traffic_sign detections
        |
        v
Create crop from existing bbox
        |
        v
Gemma receives:
    1. original image
    2. sign crop
        |
        v
Recovered scene object JSON
        |
        v
OntologyEngine resolves canonical class
        |
        +--> unresolved -> keep unreadable_traffic_sign
        |
        +--> resolved -> update semantic fields ONLY
                              |
                              v
                       preserve original bbox
                              |
                              v
                       rerender visualization

This process is intentionally independent from Locate Anything.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from annotation_pipeline.configs.settings import load_config
from annotation_pipeline.models.vlm.model_loader import create_scene_model
from annotation_pipeline.pipeline.annotation import AnnotationEngine
from annotation_pipeline.pipeline.ontology import OntologyEngine
from annotation_pipeline.prompts.load_prompt import load_prompt
from custom_logger import CustomLogger
from logmod import logs

logger = CustomLogger(__name__)


UNREADABLE_CLASS = "unreadable_traffic_sign"


def _normalize(value: Any) -> str:
    return (
        str(value or "")
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def _is_unreadable_sign(detection: dict[str, Any]) -> bool:
    return _normalize(
        detection.get(
            "class_name",
            detection.get("final_label", ""),
        )
    ) == _normalize(UNREADABLE_CLASS)


def _create_semantic_verification_crop(
    image_path: Path,
    bbox: list | tuple,
    output_dir: Path,
    object_id: Any = None,
) -> Path | None:
    """
    Create a crop from the EXISTING Locate Anything bbox.

    No localization is performed here.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.warning(
            "Could not open '{}' for semantic verification: {}",
            image_path.name,
            exc,
        )
        return None

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        logger.warning(
            "Invalid bbox for semantic verification in '{}': %r",
            image_path.name,
            bbox,
        )
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        logger.warning(
            "Non-numeric bbox for semantic verification in '{}': %r",
            image_path.name,
            bbox,
        )
        return None

    width, height = image.size

    # Preserve compatibility with either normalized or pixel bboxes.
    if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(1, min(x2, width)))
    y2 = int(max(1, min(y2, height)))

    if x2 <= x1 or y2 <= y1:
        logger.warning(
            "Degenerate bbox for semantic verification in '{}': %r",
            image_path.name,
            bbox,
        )
        return None

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = max(10, int(box_width * 0.15))
    margin_y = max(10, int(box_height * 0.15))

    crop = image.crop(
        (
            max(0, x1 - margin_x),
            max(0, y1 - margin_y),
            min(width, x2 + margin_x),
            min(height, y2 + margin_y),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = (
        f"_object_{object_id}"
        if object_id is not None
        else "_unreadable"
    )

    crop_path = (
        output_dir
        / f"{image_path.stem}{suffix}_semantic_verification.png"
    )

    crop.save(crop_path)
    return crop_path


def _parse_recovery_json(raw_response: str) -> dict[str, Any] | None:
    """
    Parse Gemma JSON, including common Markdown fenced JSON output.
    """
    text = str(raw_response or "").strip()

    if not text:
        return None

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: recover the first JSON object if Gemma added prose.
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            return None

        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

    return parsed if isinstance(parsed, dict) else None


def _recover_one_sign(
    image_path: Path,
    detection: dict[str, Any],
    recovery_model: Any,
    recovery_ontology: OntologyEngine,
    config: Any,
    crop_dir: Path,
    index: int,
    total: int,
) -> None:
    """
    Perform one semantic verification attempt.

    Localization is immutable.
    """
    bbox = detection.get("bbox")

    if bbox is None:
        detection["semantic_verification"] = {
            "attempted": True,
            "recovered": False,
            "reason": "missing_bbox",
        }
        logger.warning(
            "Unreadable traffic sign in '{}' has no bbox.",
            image_path.name,
        )
        return

    original_bbox = list(bbox)

    object_id = detection.get("object_id")

    crop_path = _create_semantic_verification_crop(
        image_path=image_path,
        bbox=bbox,
        output_dir=crop_dir,
        object_id=object_id,
    )

    if crop_path is None:
        detection["semantic_verification"] = {
            "attempted": True,
            "recovered": False,
            "reason": "invalid_bbox_or_crop",
        }
        return

    logger.info(
        "Semantic verification %d/%d for '{}'.",
        index,
        total,
        image_path.name,
    )

    try:
        response = recovery_model.infer(
            image_paths=[
                image_path,
                crop_path,
            ],
            prompt=load_prompt("semantic_recovery.txt"),
        )
    except Exception as exc:
        detection["semantic_verification"] = {
            "attempted": True,
            "recovered": False,
            "reason": f"inference_failed: {exc}",
            "crop": str(crop_path),
        }
        logger.warning(
            "Semantic verification inference failed for '{}': {}",
            image_path.name,
            exc,
        )
        return

    responses = (
        response.get("responses", [])
        if isinstance(response, dict)
        else []
    )

    if not responses:
        detection["semantic_verification"] = {
            "attempted": True,
            "recovered": False,
            "reason": "empty_model_response",
            "crop": str(crop_path),
        }
        logger.warning(
            "Semantic verification returned no response for '{}'.",
            image_path.name,
        )
        return

    recovered_scene_object = _parse_recovery_json(
        str(responses[0])
    )

    if recovered_scene_object is None:
        detection["semantic_verification"] = {
            "attempted": True,
            "recovered": False,
            "reason": "invalid_json",
            "crop": str(crop_path),
            "raw_response": str(responses[0]),
        }
        logger.warning(
            "Semantic verification returned invalid JSON for '{}'.",
            image_path.name,
        )
        return

    # The recovered object belongs to the same localized object.
    recovered_scene_object["object_id"] = object_id
    recovered_scene_object["object_group"] = "traffic_sign"

    detection["semantic_verification"] = {
        "attempted": True,
        "recovered": False,
        "crop": str(crop_path),
        "result": recovered_scene_object,
    }

    recovered_name = _normalize(
        recovered_scene_object.get("observed_object")
    )

    if not recovered_name or recovered_name == _normalize(UNREADABLE_CLASS):
        logger.info(
            "Sign %d in '{}' remains unreadable.",
            index,
            image_path.name,
        )
        return

    try:
        ontology_result = recovery_ontology._reason_object(
            recovered_scene_object
        )
    except Exception as exc:
        detection["semantic_verification"]["reason"] = (
            f"ontology_failed: {exc}"
        )
        logger.warning(
            "Ontology verification failed for '{}': {}",
            image_path.name,
            exc,
        )
        return

    detection["semantic_verification"]["ontology"] = ontology_result

    prediction = (
        ontology_result.get("prediction", {})
        if isinstance(ontology_result, dict)
        else {}
    )

    if not isinstance(prediction, dict):
        prediction = {}

    recovered_class = str(
        prediction.get("class_name") or ""
    ).strip()

    if (
        not recovered_class
        or _normalize(recovered_class) == _normalize(UNREADABLE_CLASS)
    ):
        detection["semantic_verification"]["reason"] = (
            "ontology_unresolved"
        )
        logger.info(
            "Ontology could not resolve sign %d in '{}'. "
            "Keeping unreadable_traffic_sign.",
            index,
            image_path.name,
        )
        return

    # --------------------------------------------------------------
    # SEMANTIC UPDATE ONLY.
    #
    # The original Locate Anything bbox is restored explicitly so
    # this function can never accidentally replace localization.
    # --------------------------------------------------------------
    detection["bbox"] = original_bbox
    detection["class_id"] = prediction.get("class_id")
    detection["class_name"] = recovered_class
    detection["final_label"] = recovered_class
    detection["grounding_prompt"] = prediction.get("grounding_prompt")
    detection["score"] = prediction.get("score", 0.0)

    detection["semantic_verification"]["recovered"] = True

    logger.info(
        "Recovered sign %d in '{}' as '{}'.",
        index,
        image_path.name,
        recovered_class,
    )


def rerender_annotation(
    annotation_engine: AnnotationEngine,
    image_path: Path,
    detections: list[dict[str, Any]],
) -> Path:
    """
    Render the verified detections using the existing annotation visualizer.
    """
    output_path = (
        annotation_engine.output_dir
        / f"{image_path.stem}_annotated.png"
    )

    annotation_engine.visualizer.visualize(
        image_path=image_path,
        detections=detections,
        output_path=output_path,
    )

    return output_path


def _find_input_files(
    input_dir: Path,
    image_filter: set[str] | None,
) -> list[Path]:
    if not input_dir.exists():
        return []

    files = sorted(input_dir.glob("*.json"))

    if image_filter is None:
        return files

    return [
        path
        for path in files
        if path.stem in image_filter
        or path.name in image_filter
    ]


def _load_input(path: Path) -> tuple[Path, list[dict[str, Any]]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "Could not read semantic verification input '{}': {}",
            path,
            exc,
        )
        return None

    if not isinstance(payload, dict):
        logger.warning("Invalid verification input '{}'.", path)
        return None

    image_value = payload.get("image")
    detections = payload.get("detections")

    if not image_value or not isinstance(detections, list):
        logger.warning(
            "Verification input '{}' must contain image and detections.",
            path,
        )
        return None

    image_path = Path(image_value)

    if not image_path.exists():
        logger.warning(
            "Image '{}' referenced by '{}' does not exist.",
            image_path,
            path,
        )
        return None

    normalized_detections = [
        dict(item)
        for item in detections
        if isinstance(item, dict)
    ]

    return image_path, normalized_detections


def _save_result(
    output_dir: Path,
    image_path: Path,
    detections: list[dict[str, Any]],
    visualization_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "image": str(image_path),
        "detections": detections,
        "visualization_path": str(visualization_path),
    }

    output_path = output_dir / f"{image_path.stem}.json"

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def cleanup_stage_resources(*objects: Any) -> None:
    for obj in objects:
        if obj is None:
            continue

        try:
            unload = getattr(obj, "unload", None)
            if callable(unload):
                unload()
        except Exception as exc:
            logger.warning(
                "Semantic verification resource unload failed: {}",
                exc,
            )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def main() -> None:
    config = load_config()
    config.paths.ensure_output_dirs()

    logs(
        show_level=config.logging.level.lower(),
        save_level=config.logging.level.lower(),
        program_name="semantic_verification",
        path=config.paths.logs,
    )

    parser = argparse.ArgumentParser(
        description="Standalone semantic verification for unreadable traffic signs."
    )

    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Optional image names/stems to verify. Omit to verify all saved inputs.",
    )

    args = parser.parse_args()

    input_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "inputs"
    )

    output_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "results"
    )

    crop_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "crops"
    )

    input_files = _find_input_files(
        input_dir=input_dir,
        image_filter=set(args.images) if args.images else None,
    )

    if not input_files:
        logger.error(
            "No semantic verification inputs found in '{}'.",
            input_dir,
        )
        sys.exit(1)

    loaded = []

    for input_file in input_files:
        item = _load_input(input_file)
        if item is not None:
            loaded.append(item)

    if not loaded:
        logger.error("No valid semantic verification inputs found.")
        sys.exit(1)

    # --------------------------------------------------------------
    # Only load Gemma if at least one image actually needs verification.
    # --------------------------------------------------------------
    has_targets = any(
        any(_is_unreadable_sign(detection) for detection in detections)
        for _, detections in loaded
    )

    if not has_targets:
        logger.info(
            "No unreadable traffic signs found in the selected inputs."
        )
        return

    recovery_model = None
    recovery_ontology = None

    annotation_engine = AnnotationEngine(
        pipeline_cache_dir=config.paths.pipeline_cache,
        output_dir=config.paths.annotations,
    )

    try:
        logger.info("Loading semantic verification Gemma model.")

        recovery_model = create_scene_model(config)
        recovery_model.load()

        logger.info("Loading semantic verification Ontology engine.")

        recovery_ontology = OntologyEngine(config=config)

        for image_path, detections in loaded:
            targets = [
                detection
                for detection in detections
                if _is_unreadable_sign(detection)
            ]

            if not targets:
                logger.info(
                    "No unreadable traffic signs in '{}'.",
                    image_path.name,
                )
                continue

            logger.info(
                "Verifying {} unreadable traffic sign(s) in '{}'.",
                len(targets),
                image_path.name,
            )

            for index, detection in enumerate(
                targets,
                start=1,
            ):
                _recover_one_sign(
                    image_path=image_path,
                    detection=detection,
                    recovery_model=recovery_model,
                    recovery_ontology=recovery_ontology,
                    config=config,
                    crop_dir=crop_dir,
                    index=index,
                    total=len(targets),
                )

            visualization_path = rerender_annotation(
                annotation_engine=annotation_engine,
                image_path=image_path,
                detections=detections,
            )

            _save_result(
                output_dir=output_dir,
                image_path=image_path,
                detections=detections,
                visualization_path=visualization_path,
            )

            recovered_count = sum(
                1
                for detection in targets
                if detection.get(
                    "semantic_verification",
                    {},
                ).get("recovered", False)
            )

            logger.info(
                "Semantic verification completed for '{}': "
                "{}/{}/ sign(s) recovered.",
                image_path.name,
                recovered_count,
                len(targets),
            )

    finally:
        cleanup_stage_resources(
            recovery_ontology,
            recovery_model,
        )

        logger.info("Semantic verification resources unloaded.")


if __name__ == "__main__":
    main()
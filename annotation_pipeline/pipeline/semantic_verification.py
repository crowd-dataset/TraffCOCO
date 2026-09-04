"""
semantic_verification.py

Standalone semantic verification / recovery stage.

Purpose
-------
Re-examine detections whose current class is ``unreadable_traffic_sign`` or whose current label is ``None``.

The existing Locate Anything bounding box is authoritative and is NEVER
recomputed here.

Workflow
--------
Saved annotation JSON
        |
        v
Find unreadable_traffic_sign / None detections
        |
        v
Normalize existing bbox to [0, 1] coordinates
        |
        v
Gemma receives:
    1. original image
    2. normalized bbox coordinates in the prompt
        |
        v
Recovered scene object JSON
        |
        v
OntologyEngine resolves canonical class
        |
        +--> unresolved unreadable sign -> keep unreadable_traffic_sign
        +--> unresolved None -> drop detection
        |
        +--> resolved -> update semantic fields ONLY
                              |
                              v
                       preserve original bbox
                              |
                              v
                       full annotated visualization in outputs/semantic_verified/

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
NONE_LABELS = {
    "",
    "none",
    "null",
    "unknown",
}

SEMANTIC_RECOVERY_PROMPT = 'You are an expert traffic scene understanding model performing SEMANTIC RECOVERY for ONE ALREADY-LOCALIZED candidate object.\n\nThe candidate has already been localized by Locate Anything. Its bounding box is authoritative.\nDO NOT perform object detection.\nDO NOT change, refine, expand, shrink, or replace the bounding box.\nYour only task is to determine what traffic-related object, if any, is actually present inside the provided candidate region.\n\nYou receive:\n1. the ORIGINAL FULL IMAGE, for scene context and visual confirmation;\n2. the CROPPED REGION corresponding to the existing bounding box.\n\nInspect both images before deciding.\n\n==================================================\nPRIMARY TASK\n==================================================\n\nDetermine whether the localized candidate is a real, detectable traffic-related object.\n\nIf it is a real traffic-related object and its identity can be determined from DIRECT VISUAL EVIDENCE:\n- return the most specific reliable traffic-object identity;\n- assign exactly one broad object_group from the allowed list;\n- provide a concise visual description.\n\nIf the candidate is present but its exact semantic identity cannot be determined:\n- use the most specific GENERIC traffic-object description that is actually supported by the image;\n- do NOT guess a fine-grained class from context.\n\nIf the localized region does NOT contain a detectable traffic-related object, or the visible content is not a valid object that should be represented by this traffic annotation pipeline:\n- set "discard": true;\n- set "observed_object": "";\n- do NOT invent a replacement class.\n\nA candidate MUST be discarded when the evidence shows that it is:\n- background, texture, shadow, glare, reflection, compression artifact, or noise;\n- an ordinary non-traffic object;\n- road, road surface, asphalt, pavement, sidewalk, building, vegetation, sky,\n  terrain, or another excluded scene element;\n- an accidental/invalid localization that does not correspond to a distinct\n  traffic-related object;\n- too visually ambiguous to establish that a traffic-related object is actually present.\n\nDo NOT keep a false positive merely because a bounding box exists.\n\n==================================================\nTRAFFIC OBJECT SEARCH\n==================================================\n\nInspect the candidate region and the corresponding area in the full image for:\n\nTRAFFIC CONTROL\n- traffic signs and supplementary plates\n- regulatory, warning, guide, information, parking, mandatory,\n  prohibition, priority and temporary signs\n- overhead guide signs\n- traffic, pedestrian, bicycle and bus signals\n- variable message signs\n\nROAD MARKINGS\n- lane boundaries\n- dashed and solid lane dividers\n- centre and edge lines\n- stop lines\n- crosswalks / zebra crossings\n- straight, turn and merge arrows\n- bicycle and bus symbols\n- painted road text and speed markings\n- chevrons and hatch markings\n\nTRAFFIC INFRASTRUCTURE\n- guardrails, barriers and medians\n- bollards and delineator posts\n- traffic cones and temporary barriers\n- street lights\n- traffic cameras\n- gantries\n- bridges and tunnels\n\nTRAFFIC PARTICIPANTS\n- passenger cars, taxis, vans, buses and trucks\n- motorcycles and bicycles\n- pedestrians, cyclists and riders\n- animals\n\nDo not return ordinary utility poles or other non-traffic structures unless\nthe visible object itself is a traffic-related object.\n\n==================================================\nTRAFFIC SIGN INSPECTION\n==================================================\n\nIf the candidate is a traffic sign, inspect the visible board itself.\n\nConsider:\n- circular, triangular, rectangular and octagonal geometry\n- coloured roadside boards and reflective panels\n- border and background\n- visible pictograms and arrows\n- readable text\n- supplementary plates\n- mounting arrangement\n\nA sign is still a valid traffic object when its text or pictogram cannot be\nidentified.\n\nUse "unreadable" for text ONLY when visible text exists but cannot be\ndeciphered.\n\nUse "unreadable" for symbol ONLY when a visible pictogram exists but cannot\nbe identified.\n\nUse "none" when no text or symbol is visibly present.\n\nNever invent text, symbols, languages, countries, traffic rules, or unseen\nproperties.\n\nUnreadable text does NOT automatically mean that the whole sign is\nunreadable.\n\n==================================================\nTRAFFIC SIGNAL INSPECTION\n==================================================\n\nIf the candidate is a traffic signal:\n- inspect the individual signal head;\n- determine the ACTIVE signal state whenever an illuminated lens is visible;\n- preserve the directly visible active colour and lens position.\n\nIf a red, yellow, or green lens is visibly illuminated, that colour is\nDIRECT visual evidence and MUST be included in the description and,\nwhen useful, distinguishing_features.\n\nExamples:\n- "bottom green lens illuminated"\n- "red lens lit"\n\nDo NOT infer hidden lenses, hidden signal heads, or states that are not\ndirectly visible.\n\nA distant coloured light alone is NOT sufficient evidence for a traffic\nsignal.\n\n==================================================\nSEMANTIC INTERPRETATION\n==================================================\n\nUse semantic interpretation ONLY when the identity is visually supported by\nCLEAR, DIRECT evidence.\n\nEvidence priority:\n1. directly visible text, symbols and pictograms\n2. visible geometry and layout\n3. clearly recognizable semantic identity\n4. generic physical description\n\nDo NOT infer semantic meaning from:\n- shape or colour alone\n- text layout alone\n- mounting position\n- road location\n- surrounding objects\n- common traffic-sign conventions\n- similarity to an ontology class\n\nFor example:\n- a clearly recognizable STOP sign may be identified as a stop sign;\n- a rectangular board with unreadable text should remain a generic\n  rectangular traffic sign with unreadable text unless stronger evidence\n  identifies it.\n\nIf multiple semantic classes remain plausible, DO NOT choose one.\nUse a generic physical traffic-object description instead.\n\nDo NOT assign ontology class IDs.\n\n==================================================\nOBJECT GROUP\n==================================================\n\nEvery kept object must have exactly one object_group.\n\nUse ONLY:\nvehicle\nroad_user\nanimal\ntraffic_sign\ntraffic_signal\nroad_marking\nroad_infrastructure\ninfrastructure\ntemporary_object\ncountry_specific\n\nobject_group is the BROAD PHYSICAL CATEGORY, not the fine-grained ontology\nclass.\n\nExamples:\n- bus/car/truck -> vehicle\n- pedestrian/cyclist/rider -> road_user\n- traffic sign -> traffic_sign\n- traffic light -> traffic_signal\n- crosswalk/lane divider/road arrow -> road_marking\n- guardrail/bollard/delineator/median -> road_infrastructure\n\nDo NOT use fine-grained ontology class names as object_group.\nDo NOT use "person" as object_group.\n\n==================================================\nDESCRIPTION\n==================================================\n\nFor a kept object, write one concise, information-rich description of about\n30–70 words.\n\nDescribe ONLY observable characteristics.\n\nWhere applicable include:\n- colour\n- shape\n- approximate size\n- border and background\n- pictograms\n- arrows\n- active signal state and lens position\n- readable text\n- reflective appearance\n- material only if visually obvious\n- mounting\n- distinctive markings\n- damage\n- partial occlusion\n\nFor signs, describe visible geometry, border, background, pictogram, arrow,\ntext, supplementary plates and mounting.\n\nFor road markings, describe solid/dashed pattern, single/double lines,\ndirection, symbols, continuity, wear and orientation.\n\nFor traffic signals, explicitly describe the illuminated lens when clearly\nvisible.\n\nDo not replace direct visual evidence with "unknown" when something is\nclearly visible.\n\n==================================================\nEXCLUDED OBJECTS\n==================================================\n\nThe following are NOT traffic-object detections and MUST be discarded:\n\n- road\n- road surface\n- asphalt\n- pavement\n- sidewalk\n- buildings\n- houses\n- offices\n- shops\n- vegetation\n- trees\n- sky\n- clouds\n- terrain\n\nAlso discard an invalid localization when the bbox does not actually\ncorrespond to a distinct traffic-related object.\n\n==================================================\nDISCARD DECISION\n==================================================\n\nThe "discard" field is authoritative.\n\nSet:\n"discard": false\nwhen the candidate is a valid traffic-related object that should remain\nrepresented.\n\nSet:\n"discard": true\nwhen the candidate should NOT be represented as a traffic-object\nannotation.\n\nWhen discard is true:\n- "observed_object" MUST be "";\n- "object_group" may be "";\n- "description" may be "";\n- do NOT provide a guessed replacement object.\n\nFor a candidate whose current annotation is "none", this is especially\nimportant: inspect what is actually inside the existing bbox and determine\nwhether it is a valid traffic-related object. If it is, identify it using\nthe same evidence-based rules as the main scene-understanding prompt. If it\nis not, explicitly discard it.\n\n==================================================\nOUTPUT FORMAT\n==================================================\n\nReturn EXACTLY ONE valid JSON object.\n\nReturn ONLY the JSON object.\nNo Markdown.\nNo explanations.\n\nFor a valid object:\n\n{\n    "discard": false,\n    "observed_object": "...",\n    "object_group": "...",\n    "description": "..."\n}\n\nFor an object that must be discarded:\n\n{\n    "discard": true,\n    "observed_object": "",\n    "object_group": "",\n    "description": ""\n}\n\n==================================================\nFINAL VERIFICATION\n==================================================\n\nBefore returning the JSON:\n1. Confirm the candidate region contains a real traffic-related object.\n2. Confirm the proposed identity is supported by direct visual evidence.\n3. Confirm object_group is one of the allowed broad categories.\n4. Confirm no excluded scene element is being returned.\n5. If the candidate is not a valid detectable traffic object, set\n   "discard": true.\n6. Never guess merely because Locate Anything supplied a bbox.\n'


def _normalize(value: Any) -> str:
    return (
        str(value or "")
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def _detection_label(detection: dict[str, Any]) -> str:
    """Return the current annotation label."""
    value = detection.get("class_name")

    if value is None:
        value = detection.get("final_label")

    return _normalize(value)


def _is_recovery_target(detection: dict[str, Any]) -> bool:
    """Return whether this detection needs semantic class recovery."""
    label = _detection_label(detection)

    return (
        label == _normalize(UNREADABLE_CLASS)
        or label in NONE_LABELS
    )


def _is_none_label(detection: dict[str, Any]) -> bool:
    """Return whether this detection has no usable current class."""
    return _detection_label(detection) in NONE_LABELS




def _semantic_discard_requested(
    semantic_result: dict[str, Any],
) -> bool:
    """Return True when Gemma explicitly instructs the pipeline to discard.

    Accept the canonical boolean field and a few defensive textual forms so
    minor JSON formatting variations cannot accidentally keep a false
    positive. An explicit discard instruction always takes precedence over
    semantic recovery.
    """
    if not isinstance(semantic_result, dict):
        return False

    value = semantic_result.get("discard")

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "yes",
            "discard",
            "drop",
            "remove",
        }

    decision = semantic_result.get("decision")
    if isinstance(decision, str):
        return decision.strip().lower() in {
            "discard",
            "drop",
            "remove",
        }

    action = semantic_result.get("action")
    if isinstance(action, str):
        return action.strip().lower() in {
            "discard",
            "drop",
            "remove",
        }

    return False


def _normalize_bbox_for_model(
    image_path: Path,
    bbox: list | tuple,
) -> list[float] | None:
    """
    Convert the authoritative existing bbox to normalized [0, 1]
    coordinates for Gemma.

    The returned order is [x1, y1, x2, y2].
    No localization or bbox modification is performed. Pixel bboxes are
    converted using the original image dimensions; already-normalized
    bboxes are preserved after validation.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        logger.warning(
            "Invalid bbox for semantic verification in '{}': %r",
            image_path.name,
            bbox,
        )
        return None

    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        logger.warning(
            "Non-numeric bbox for semantic verification in '{}': %r",
            image_path.name,
            bbox,
        )
        return None

    if not all(0.0 <= value <= 1.0 for value in values):
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            logger.warning(
                "Could not open '{}' to normalize bbox: {}",
                image_path.name,
                exc,
            )
            return None

        if width <= 0 or height <= 0:
            logger.warning(
                "Invalid image dimensions for '{}': {}x{}",
                image_path.name,
                width,
                height,
            )
            return None

        x1, y1, x2, y2 = values
        values = [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        ]

    x1, y1, x2, y2 = values

    # Clamp only the coordinate representation sent to Gemma. The original
    # detection bbox is never modified.
    normalized = [
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
        max(0.0, min(1.0, x2)),
        max(0.0, min(1.0, y2)),
    ]

    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        logger.warning(
            "Degenerate bbox for semantic verification in '{}': %r",
            image_path.name,
            bbox,
        )
        return None

    return normalized



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


def _recover_one_detection(
    image_path: Path,
    detection: dict[str, Any],
    recovery_model: Any,
    recovery_ontology: OntologyEngine,
    config: Any,
    index: int,
    total: int,
) -> None:
    """
    Perform one semantic recovery attempt.

    Localization is immutable. The existing bbox is authoritative.

    For ``unreadable_traffic_sign``:
        - resolve when a valid class can be determined;
        - otherwise keep the unreadable class.

    For ``None``:
        - resolve when a valid class can be determined;
        - if Gemma explicitly returns ``discard=true``, drop it immediately;
        - otherwise drop it when no valid class can be determined.

    For either target:
        - an explicit Gemma ``discard=true`` is authoritative and the
          detection is not counted in the final annotation.

    The model is NOT required to invent a description. Only a usable
    ``observed_object`` is needed to attempt ontology resolution.

    Gemma receives only the original full image plus the authoritative bbox
    as normalized coordinates in the prompt. No crop is created or passed.
    """
    is_none_target = _is_none_label(detection)

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
        return not is_none_target

    original_bbox = list(bbox)

    object_id = detection.get("object_id")

    normalized_bbox = _normalize_bbox_for_model(
        image_path=image_path,
        bbox=bbox,
    )

    if normalized_bbox is None:
        detection["semantic_verification"] = {
            "attempted": True,
            "recovered": False,
            "reason": "invalid_bbox",
        }
        return not is_none_target

    logger.info(
        "Semantic verification %d/%d for '{}'.",
        index,
        total,
        image_path.name,
    )

    try:
        recovery_prompt = SEMANTIC_RECOVERY_PROMPT

        bbox_instruction = (
            "\n\nAUTHORITATIVE CANDIDATE BBOX (NORMALIZED): "
            f"[{normalized_bbox[0]:.6f}, {normalized_bbox[1]:.6f}, "
            f"{normalized_bbox[2]:.6f}, {normalized_bbox[3]:.6f}]"
            "\nCoordinates are [x1, y1, x2, y2], normalized to [0, 1]. "
            "Inspect exactly this region in the ORIGINAL FULL IMAGE."
        )

        response = recovery_model.infer(
            image_paths=[image_path],
            prompt=recovery_prompt + bbox_instruction,
        )
    except Exception as exc:
        detection["semantic_verification"] = {
            "attempted": True,
            "recovered": False,
            "reason": f"inference_failed: {exc}",
            "bbox_normalized": normalized_bbox,
        }
        logger.warning(
            "Semantic verification inference failed for '{}': {}",
            image_path.name,
            exc,
        )
        return not is_none_target

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
            "bbox_normalized": normalized_bbox,
        }
        logger.warning(
            "Semantic verification returned no response for '{}'.",
            image_path.name,
        )
        return not is_none_target

    raw_response = str(responses[0])

    recovered_scene_object = _parse_recovery_json(
        raw_response
    )

    if recovered_scene_object is None:
        detection["semantic_verification"] = {
            "attempted": True,
            "recovered": False,
            "reason": "invalid_json",
            "bbox_normalized": normalized_bbox,
            "raw_response": str(responses[0]),
        }
        logger.warning(
            "Semantic verification returned invalid JSON for '{}'.",
            image_path.name,
        )
        return not is_none_target

    detection["semantic_verification"] = {
        "attempted": True,
        "recovered": False,
        "bbox_normalized": normalized_bbox,
        "raw_response": raw_response,
        "discard_requested": _semantic_discard_requested(
            recovered_scene_object
        ),
        "accounted_for": True,
    }

    if _semantic_discard_requested(recovered_scene_object):
        detection["semantic_verification"]["reason"] = "gemma_discard"
        detection["semantic_verification"]["accounted_for"] = False

        logger.warning(
            "Gemma instructed semantic verification to discard detection "
            "{} in '{}'. It will not be counted as a final detection.",
            detection.get("object_id", "unknown"),
            image_path.name,
        )
        return False

    # The recovered object belongs to the same localized object.
    recovered_scene_object["object_id"] = object_id

    # Preserve an existing semantic group when one is available. Only the
    # unreadable-traffic-sign path is known to be a traffic sign. A None
    # detection can be any ontology class, so do not force it into a group.
    if not is_none_target:
        recovered_scene_object["object_group"] = "traffic_sign"
    else:
        existing_group = (
            detection.get("object_group")
            or detection.get("superclass")
            or detection.get("group")
        )

        if isinstance(existing_group, str) and existing_group.strip():
            recovered_scene_object["object_group"] = existing_group

    detection["semantic_verification"].update({
        "result": recovered_scene_object,
    })

    recovered_name = _normalize(
        recovered_scene_object.get("observed_object")
    )

    if not recovered_name:
        detection["semantic_verification"]["reason"] = (
            "no_class_recovered"
        )

        if is_none_target:
            logger.warning(
                "None-labeled detection %d in '{}' could not be assigned "
                "a class and will be dropped.",
                index,
                image_path.name,
            )
            return False

        logger.info(
            "Unreadable traffic sign %d in '{}' remains unreadable.",
            index,
            image_path.name,
        )
        return True

    if recovered_name == _normalize(UNREADABLE_CLASS):
        detection["semantic_verification"]["reason"] = (
            "still_unreadable"
        )

        if is_none_target:
            logger.warning(
                "None-labeled detection %d in '{}' could not be assigned "
                "a usable class and will be dropped.",
                index,
                image_path.name,
            )
            return False

        logger.info(
            "Unreadable traffic sign %d in '{}' remains unreadable.",
            index,
            image_path.name,
        )
        return True

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
        return not is_none_target

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

        if is_none_target:
            logger.warning(
                "None-labeled detection %d in '{}' could not be resolved "
                "by Ontology and will be dropped.",
                index,
                image_path.name,
            )
            return False

        logger.info(
            "Ontology could not resolve unreadable traffic sign %d in '{}'. "
            "Keeping unreadable_traffic_sign.",
            index,
            image_path.name,
        )
        return True

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
    detection["semantic_verification"]["accounted_for"] = True

    logger.info(
        "Recovered detection %d in '{}' as '{}'.",
        index,
        image_path.name,
        recovered_class,
    )

    return True


def rerender_annotation(
    annotation_engine: AnnotationEngine,
    image_path: Path,
    detections: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """
    Render the COMPLETE verified annotation set into the dedicated
    ``semantic_verified`` directory.

    This includes both:
      - classes recovered by semantic verification, and
      - classes that were already determined before verification.

    The original bboxes are preserved in ``detections``; this function only
    reuses the existing annotation visualizer to draw them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_annotated.png"

    annotation_engine.visualizer.visualize(
        image_path=image_path,
        detections=detections,
        output_path=output_path,
    )

    return output_path


def _load_pipeline_cache_snapshot(
    config: Any,
    image_path: Path,
) -> dict[str, Any] | None:
    """
    Load the complete PipelineCache snapshot for this image.

    Semantic verification is a separate process, so its input JSON only
    contains the final detections by default. For debugging, preserve the
    original pipeline cache as a read-only snapshot alongside the semantic
    verification results. This makes it possible to trace an object through
    Scene Understanding -> Ontology -> Grounding -> Annotation without
    changing the semantic verification logic.
    """
    cache_path = (
        config.paths.pipeline_cache
        / f"{image_path.stem}.json"
    )

    if not cache_path.exists():
        logger.warning(
            "Pipeline cache for '{}' was not found at '{}'.",
            image_path.name,
            cache_path,
        )
        return None

    try:
        payload = json.loads(
            cache_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        logger.warning(
            "Could not read pipeline cache for '{}': {}",
            image_path.name,
            exc,
        )
        return None

    if not isinstance(payload, dict):
        logger.warning(
            "Pipeline cache for '{}' is not a JSON object.",
            image_path.name,
        )
        return None

    return payload


def _save_diagnostic_result(
    output_dir: Path,
    image_path: Path,
    detections: list[dict[str, Any]],
    visualization_path: Path,
    pipeline_cache: dict[str, Any] | None,
) -> Path:
    """
    Save a diagnostic JSON containing both the complete original
    PipelineCache and the final semantic-verification detections.

    ``pipeline_cache`` is intentionally kept under its own key rather than
    flattened into detections. That preserves the exact cache hierarchy and
    makes it clear which data existed before semantic verification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "image": str(image_path),
        "image_name": image_path.name,
        "pipeline_cache": pipeline_cache,
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

    semantic_verified_dir = (
        config.paths.outputs
        / "semantic_verified"
    )

    results_dir = (
        config.paths.outputs
        / "semantic_verification"
        / "results"
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
        any(_is_recovery_target(detection) for detection in detections)
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
            # Preserve the complete pre-verification pipeline state for
            # diagnostics. The semantic stage never mutates this snapshot.
            pipeline_cache_snapshot = _load_pipeline_cache_snapshot(
                config=config,
                image_path=image_path,
            )

            targets = [
                detection
                for detection in detections
                if _is_recovery_target(detection)
            ]

            if not targets:
                logger.info(
                    "No semantic recovery targets in '{}'.",
                    image_path.name,
                )
                continue

            logger.info(
                "Verifying {} semantic recovery target(s) in '{}'.",
                len(targets),
                image_path.name,
            )

            dropped_none_ids = set()

            for index, detection in enumerate(
                targets,
                start=1,
            ):
                keep_detection = _recover_one_detection(
                    image_path=image_path,
                    detection=detection,
                    recovery_model=recovery_model,
                    recovery_ontology=recovery_ontology,
                    config=config,
                    index=index,
                    total=len(targets),
                )

                if (
                    not keep_detection
                    and _is_none_label(detection)
                ):
                    dropped_none_ids.add(id(detection))

            if dropped_none_ids:
                detections[:] = [
                    detection
                    for detection in detections
                    if id(detection) not in dropped_none_ids
                ]

            visualization_path = rerender_annotation(
                annotation_engine=annotation_engine,
                image_path=image_path,
                detections=detections,
                output_dir=semantic_verified_dir,
            )

            diagnostic_path = _save_diagnostic_result(
                output_dir=results_dir,
                image_path=image_path,
                detections=detections,
                visualization_path=visualization_path,
                pipeline_cache=pipeline_cache_snapshot,
            )

            logger.info(
                "Semantic verification diagnostic JSON saved for '{}': {}",
                image_path.name,
                diagnostic_path,
            )

            recovered_count = sum(
                1
                for detection in targets
                if detection.get(
                    "semantic_verification",
                    {},
                ).get("recovered", False)
            )

            dropped_none_count = len(dropped_none_ids)

            logger.info(
                "Semantic verification completed for '{}': "
                "{} recovered, {} None-labeled detection(s) dropped.",
                image_path.name,
                recovered_count,
                dropped_none_count,
            )

    finally:
        cleanup_stage_resources(
            recovery_ontology,
            recovery_model,
        )

        logger.info("Semantic verification resources unloaded.")


if __name__ == "__main__":
    main()
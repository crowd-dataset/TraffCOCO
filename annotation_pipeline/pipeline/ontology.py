"""
ontology.py

Ontology Reasoning Engine.

This module implements the second stage of the VLM-First annotation
pipeline.

Pipeline
--------
Pipeline Cache
        │
        ▼
Retrieval Query Builder
        │
        ▼
Ontology Reasoner
        │
        ▼
Pipeline Cache
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import json

from custom_logger import CustomLogger

from annotation_pipeline.cache.pipeline_cache import PipelineCache

from annotation_pipeline.configs.settings import (
    PipelineConfig,
)

from annotation_pipeline.pipeline.sign_text import (
    GUIDE_SIGN_CLASS,
    extract_sign_text,
    needs_guide_sign_class,
)
from annotation_pipeline.pipeline.retrieval_query_builder import (
    RetrievalQueryBuilder,
)

from annotation_pipeline.pipeline.ontology_reasoner import (
    OntologyReasoner,
)

logger = CustomLogger(__name__)


class OntologyEngine:
    """
    Ontology Reasoning stage of the annotation pipeline.

    Responsibilities
    ----------------

    • Read Scene Understanding objects from PipelineCache
    • Build ontology retrieval queries
    • Execute ontology reasoning
    • Store ontology predictions in PipelineCache
    """

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:

        self.config = config

        self.query_builder = RetrievalQueryBuilder()

        self.reasoner = OntologyReasoner(
            config=config,
        )

        # class_name -> ontology entry, for sign-text checks.
        self._classes_by_name = self._load_class_entries()

        logger.info(
            "Initialized OntologyEngine."
        )

        # ----------------------------------------------------------
        # Benchmark Statistics
        # ----------------------------------------------------------

        self.last_reasoning_time = 0.0

        self.last_objects_processed = 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def process_images(
        self,
        image_name: str,
        cache: PipelineCache,
    ) -> list[dict[str, Any]]:

        logger.info(
            "Running Ontology Reasoning for '{}'.",
            image_name,
        )

        # ----------------------------------------------------------
        # Load parsed Scene Understanding JSON
        # ----------------------------------------------------------

        scene_file = (
            self.config.paths.scene_cache
            / "parsed"
            / f"{Path(image_name).stem}.json"
        )

        if not scene_file.exists():

            raise FileNotFoundError(
                f"Parsed Scene Understanding file not found: {scene_file}"
            )

        with scene_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            scene_objects = json.load(f)

        if not scene_objects:

            logger.warning(
                "No Scene Understanding objects found for '{}'.",
                image_name,
            )

            return []

        # A run resumed with --start-stage ontology_reasoning starts with
        # an empty in-memory cache; fill it from the parsed file.
        if image_name not in getattr(cache, "_cache", {}):

            cache.add_scene_objects(
                image_name=image_name,
                objects=scene_objects,
            )

        logger.info(
            "Loaded {} Scene Understanding object(s).",
            len(scene_objects),
        )

        # ----------------------------------------------------------
        # Ontology Reasoning
        # ----------------------------------------------------------

        start_time = time.perf_counter()

        ontology_results = []

        for index, scene_object in enumerate(
            scene_objects,
            start=1,
        ):

            logger.info(
                "Processing object {}/{} : {}",
                index,
                len(scene_objects),
                scene_object.get(
                    "observed_object",
                    "Unknown",
                ),
            )

            result = self._reason_object(
                scene_object,
            )

            ontology_results.append(result)

            # ------------------------------------------------------
            # Update existing PipelineCache object
            # ------------------------------------------------------

            cache.add_ontology_result(
                image_name=image_name,
                object_id=scene_object["object_id"],
                ontology_result=result,
            )

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        self.last_reasoning_time = (
            time.perf_counter() - start_time
        )

        self.last_objects_processed = len(scene_objects)

        logger.info(
            "Ontology Reasoning completed for '{}'.",
            image_name,
        )

        logger.info(
            "Objects Processed : {}",
            self.last_objects_processed,
        )

        logger.info(
            "Reasoning Time : {:.2f} s",
            self.last_reasoning_time,
        )

        if self.last_objects_processed:

            logger.info(
                "Average/Object : {:.4f} s",
                self.last_reasoning_time
                / self.last_objects_processed,
            )

        # ----------------------------------------------------------
        # Save updated Pipeline Cache
        # ----------------------------------------------------------

        if self.config.pipeline.save_intermediate_cache:

            cache.save_image_cache(
                image_name=image_name,
                directory=self.config.paths.pipeline_cache,
                stage="ontology_reasoning",
            )

        return ontology_results

    # ------------------------------------------------------------------
    # Object Reasoning
    # ------------------------------------------------------------------

    def _reason_object(
        self,
        scene_object: dict[str, Any],
    ) -> dict[str, Any]:

        retrieval_query = self.query_builder.build(
            scene_object,
        )

        result = self.reasoner.reason(
            retrieval_query,
        )

        if result.predicted_class is None:
            logger.warning(
                "No ontology prediction available for '{}'.",
                retrieval_query.observed_object,
            )
            return {
                "retrieval_query": retrieval_query.embedding_text,
                "prediction": {
                    "class_id": None,
                    "class_name": None,
                    "ontology_score": 0.0,
                    "grounding_prompt": None,
                },
                "top_candidates": [],
            }

        predicted_entry = result.predicted_class.entry

        ontology_result = {
            "retrieval_query": retrieval_query.embedding_text,
            "scene_confidence": retrieval_query.scene_confidence,
            "prediction": {
                "class_id": result.predicted_class.entry.class_id,
                "class_name": result.predicted_class.entry.class_name,
                "ontology_score": result.predicted_class.final_score,
                "grounding_prompt": (
                    result.predicted_class.entry.grounding_prompt
                ),
            },
            "top_candidates": [
                {
                    "class_id": candidate.entry.class_id,
                    "class_name": candidate.entry.class_name,
                    "score": candidate.final_score,
                    "embedding_score": candidate.embedding_score,
                    "attribute_score": candidate.attribute_score,
                    "final_score": candidate.final_score,
                }
                for candidate in result.candidates
            ],
        }

        logger.debug(
            "Ontology Prediction:\n{}",
            ontology_result["prediction"],
        )

        return self._apply_sign_text(scene_object, ontology_result)

    # ------------------------------------------------------------------
    # Sign text
    # ------------------------------------------------------------------

    def _load_class_entries(self) -> dict[str, dict[str, Any]]:

        try:
            with open(
                self.config.paths.ontology_file,
                "r",
                encoding="utf-8",
            ) as f:
                ontology = json.load(f)
        except Exception as exc:
            logger.warning(
                "Could not read ontology classes for sign-text checks: {}",
                exc,
            )
            return {}

        classes = ontology.get("classes", [])

        if isinstance(classes, dict):
            classes = list(classes.values())

        return {
            entry["class_name"]: entry
            for entry in classes
            if isinstance(entry, dict) and entry.get("class_name")
        }

    def _apply_sign_text(
        self,
        scene_object: dict[str, Any],
        ontology_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Store a sign's text/language/meaning and relabel signs whose
        words the predicted class cannot represent (see sign_text.py)."""

        group = str(scene_object.get("object_group") or "").lower()
        prediction = ontology_result.get("prediction") or {}
        predicted = prediction.get("class_name")
        entry = self._classes_by_name.get(predicted or "", {})

        if group != "traffic_sign" and entry.get("super_category") != "traffic_sign":
            return ontology_result

        sign_text = extract_sign_text(scene_object)

        if sign_text is None:
            return ontology_result

        ontology_result["sign_text"] = sign_text

        guide = self._classes_by_name.get(GUIDE_SIGN_CLASS)

        if guide and needs_guide_sign_class(
            sign_text=sign_text,
            predicted_class=predicted,
            class_entry=entry,
        ):
            guide_score = next(
                (
                    candidate.get("final_score")
                    for candidate in ontology_result.get("top_candidates", [])
                    if candidate.get("class_name") == GUIDE_SIGN_CLASS
                ),
                None,
            )

            logger.info(
                "Sign '{}' shows text '{}' that class '{}' cannot represent; "
                "relabelled as '{}'.",
                scene_object.get("observed_object"),
                sign_text.get("text"),
                predicted,
                GUIDE_SIGN_CLASS,
            )

            ontology_result["prediction"] = {
                "class_id": guide.get("class_id"),
                "class_name": GUIDE_SIGN_CLASS,
                "ontology_score": (
                    guide_score
                    if guide_score is not None
                    else prediction.get("ontology_score")
                ),
                # Ground this specific sign, not a generic text sign.
                "grounding_prompt": (
                    scene_object.get("observed_object")
                    or guide.get("grounding_prompt")
                ),
                "replaced_class": predicted,
            }

        return ontology_result

    def reason_recovered_scene_object(
        self,
        scene_object: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Re-run ontology reasoning for a scene object produced by the
        semantic recovery pass.

        This method performs ontology reasoning only.

        Locate Anything is intentionally not involved.
        """

        retrieval_query = self.query_builder.build(
            scene_object,
        )

        result = self.reasoner.reason(
            retrieval_query,
        )

        if result.predicted_class is None:

            return {
                "retrieval_query": retrieval_query.embedding_text,
                "prediction": {
                    "class_id": None,
                    "class_name": None,
                    "ontology_score": 0.0,
                    "grounding_prompt": None,
                },
                "top_candidates": [],
            }

        predicted_entry = (
            result.predicted_class.entry
        )

        return self._apply_sign_text(scene_object, {
            "retrieval_query": (
                retrieval_query.embedding_text
            ),
            "scene_confidence": retrieval_query.scene_confidence,

            "prediction": {
                            "class_id": result.predicted_class.entry.class_id,
                            "class_name": result.predicted_class.entry.class_name,
                            "ontology_score": result.predicted_class.final_score,
                            "grounding_prompt": (
                                result.predicted_class.entry.grounding_prompt
                            ),
                        },

            "top_candidates": [
                {
                    "class_id": candidate.entry.class_id,
                    "class_name": candidate.entry.class_name,
                    "score": candidate.final_score,
                    "embedding_score": candidate.embedding_score,
                    "attribute_score": candidate.attribute_score,
                    "final_score": candidate.final_score,
                }
                for candidate in result.candidates
            ],
        })

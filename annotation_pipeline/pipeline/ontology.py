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

        ontology_result = {

            "retrieval_query":
                retrieval_query.embedding_text,

            "prediction": {

                "class_id":
                    result.predicted_class.entry.class_id,

                "class_name":
                    result.predicted_class.entry.class_name,

                "score":
                    result.predicted_class.final_score,

            },

            "top_candidates": [

                {

                    "class_id":
                        candidate.entry.class_id,

                    "class_name":
                        candidate.entry.class_name,

                    "embedding_score":
                        candidate.embedding_score,

                    "attribute_score":
                        candidate.attribute_score,

                    "final_score":
                        candidate.final_score,

                }

                for candidate in result.candidates

            ],

        }

        logger.debug(
            "Ontology Prediction:\n{}",
            ontology_result["prediction"],
        )

        return ontology_result

    
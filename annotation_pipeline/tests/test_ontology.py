"""
test_ontology.py

Standalone ontology retrieval test.

This script evaluates ontology retrieval independently of the
Scene Understanding stage by loading a parsed Scene Understanding
JSON file and retrieving the most likely ontology classes.

Pipeline

Parsed Scene Understanding JSON
        │
        ▼
RetrievalQueryBuilder
        │
        ▼
OntologyReasoner
        │
        ▼
Top-K Ontology Candidates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from custom_logger import CustomLogger
from logmod import logs

from annotation_pipeline.configs.settings import (
    load_config,
)

from annotation_pipeline.pipeline.retrieval_query_builder import (
    RetrievalQueryBuilder,
)

from annotation_pipeline.pipeline.ontology_reasoner import (
    OntologyReasoner,
)

logger = CustomLogger(__name__)

def main() -> None:
    """
    Run ontology retrieval on a parsed Scene Understanding JSON file.
    """

    parser = argparse.ArgumentParser(
        description="Standalone Ontology Retrieval Test",
    )

    parser.add_argument(
        "scene_json",
        type=Path,
        help="Path to parsed Scene Understanding JSON.",
    )

    args = parser.parse_args()

    logs()

    logger.info(
        "Loading pipeline configuration.",
    )

    config = load_config()

    logger.info(
        "Initializing ontology components.",
    )

    query_builder = RetrievalQueryBuilder()

    ontology_reasoner = OntologyReasoner(
        config=config,
    )

    logger.info(
        "Loading parsed Scene Understanding JSON: %s",
        args.scene_json,
    )

    with args.scene_json.open(
        "r",
        encoding="utf-8",
    ) as file:

        scene_objects = json.load(
            file,
        )

    if not isinstance(
        scene_objects,
        list,
    ):

        raise ValueError(
            "Parsed Scene Understanding JSON must contain a list of objects."
        )

    logger.info(
        "Loaded %d scene objects.",
        len(scene_objects),
    )

    ontology_results: list[dict] = []

    logger.info(
        "Starting ontology retrieval.",
    )

    for index, scene_object in enumerate(
        scene_objects,
        start=1,
    ):

        logger.info(
            "Processing object %d: %s",
            index,
            scene_object.get(
                "object_name",
                "Unknown",
            ),
        )

        retrieval_query = query_builder.build(
            scene_object,
        )

        result = ontology_reasoner.reason(
            retrieval_query,
        )

        ontology_results.append(
            {
                "object_id": scene_object.get("object_id"),
                "object_name": scene_object.get("object_name"),
                "retrieval_query": retrieval_query.embedding_text,

                "prediction": {
                    "class_id": result.predicted_class.entry.class_id,
                    "class_name": result.predicted_class.entry.class_name,
                    "score": result.predicted_class.final_score,
                },

                "top_candidates": [
                    {
                        "class_id": candidate.entry.class_id,
                        "class_name": candidate.entry.class_name,
                        "score": candidate.final_score,
                    }
                    for candidate in result.candidates
                ],
            }
        )

        print("\n" + "=" * 80)

        print(
            f"Object {index}: "
            f"{scene_object.get('object_name', 'Unknown')}"
        )

        print("-" * 80)

        print(
            retrieval_query.embedding_text,
        )

        print("-" * 80)

        print(
            f"Prediction : {result.predicted_class.entry.class_name}"
        )

        print(
            f"Score      : {result.predicted_class.final_score:.3f}"
        )

        print("\nTop Candidates:")

        for candidate in result.candidates[:5]:
            print(
                f"{candidate.entry.class_name:30}"
                f"Embedding: {candidate.embedding_score:.3f}  "
                f"Attributes: {candidate.attribute_score:.3f}  "
                f"Final: {candidate.final_score:.3f}"
            )

        output_path = (
        args.scene_json.parent
        / f"{args.scene_json.stem}_ontology_results.json"
    )

    logger.info(
        "Saving ontology results to %s",
        output_path,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            ontology_results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Ontology retrieval completed successfully.",
    )


if __name__ == "__main__":
    main()
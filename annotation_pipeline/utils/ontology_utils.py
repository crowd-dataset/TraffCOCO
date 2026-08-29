"""
ontology_utils.py

Utility functions for converting the ontology JSON into OntologyEntry
objects used by the ontology index.
"""

from __future__ import annotations

from typing import Any

from annotation_pipeline.models.ontology.ontology_models import OntologyEntry


def build_ontology_entries(
    ontology: dict[str, Any],
) -> list[OntologyEntry]:
    """
    Convert the ontology JSON into OntologyEntry objects.

    The production ontology JSON will eventually include a
    ``grounding_prompt`` field. The code is made compatible with that
    future schema while still accepting older entries that do not define
    it yet.
    """

    classes = ontology.get("classes", [])

    entries: list[OntologyEntry] = []

    for item in classes:

        class_id = item.get("class_id")
        class_name = item.get("class_name")
        embedding_text = item.get("embedding_text", "")
        grounding_prompt = item.get("grounding_prompt", class_name)

        if not isinstance(grounding_prompt, str):
            grounding_prompt = str(class_name or "")

        grounding_prompt = grounding_prompt.strip()

        if not grounding_prompt and class_name is not None:
            grounding_prompt = str(class_name).strip()

        entries.append(
            OntologyEntry(
                class_id=class_id,
                class_name=class_name,
                embedding_text=embedding_text,
                grounding_prompt=grounding_prompt,
                data=item,
            )
        )

    return entries
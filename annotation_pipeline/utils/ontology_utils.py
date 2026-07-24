"""
ontology_utils.py

Utility functions for converting the ontology JSON into OntologyEntry
objects used by the ontology index.
"""

from __future__ import annotations

from typing import Any

from annotation_pipeline.ontology.ontology_models import OntologyEntry


def build_ontology_entries(
    ontology: dict[str, Any],
) -> list[OntologyEntry]:
    """
    Convert the ontology JSON into OntologyEntry objects.

    Parameters
    ----------
    ontology
        Parsed ontology JSON.

    Returns
    -------
    list[OntologyEntry]
    """

    classes = ontology.get("classes", [])

    entries: list[OntologyEntry] = []

    for item in classes:

        entries.append(
            OntologyEntry(
                class_id=item["class_id"],
                class_name=item["class_name"],
                embedding_text=item["embedding_text"],
                data=item,
            )
        )

    return entries
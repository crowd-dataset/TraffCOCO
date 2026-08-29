"""
ontology_models.py

Shared data models for the Ontology Reasoning stage.

These models define the data exchanged between:

Scene Understanding
        │
        ▼
Retrieval Query
        │
        ▼
Ontology Index
        │
        ▼
Candidate Ranker
        │
        ▼
Ontology Prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# Ontology Entry
# ============================================================================


@dataclass(slots=True)
class OntologyEntry:
    """
    Represents a single ontology class.
    """

    class_id: str

    class_name: str

    embedding_text: str

    grounding_prompt: str

    data: dict[str, Any]


# ============================================================================
# Retrieval Query
# ============================================================================


@dataclass(slots=True)
class RetrievalQuery:
    """
    Semantic query produced from Scene Understanding.
    """

    observed_object: str

    object_group: str

    description: str

    shape: str | None = None

    primary_color: str | None = None

    secondary_color: str | None = None

    material: str | None = None

    text: str | None = None

    symbol: str | None = None

    distinguishing_features: list[str] = field(
        default_factory=list
    )

    attached_to: str | None = None

    nearby_objects: list[str] = field(
        default_factory=list
    )

    road_side: str | None = None

    possible_function: str | None = None

    classification_hint: str | None = None

    confidence: float | None = None

    embedding_text: str = ""


# ============================================================================
# Candidate Match
# ============================================================================


@dataclass(slots=True)
class CandidateMatch:
    """
    Candidate returned by semantic retrieval.
    """

    entry: OntologyEntry

    embedding_score: float

    attribute_score: float = 0.0

    final_score: float = 0.0

    matched_attributes: dict[str, Any] = field(
        default_factory=dict
    )

    penalties: list[str] = field(
        default_factory=list
    )


# ============================================================================
# Search Result
# ============================================================================


@dataclass(slots=True)
class OntologySearchResult:
    """
    Output of the ontology reasoning stage.
    """

    predicted_class: CandidateMatch

    candidates: list[CandidateMatch]

    retrieval_query: RetrievalQuery

    
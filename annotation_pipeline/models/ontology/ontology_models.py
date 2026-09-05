"""
ontology_models.py

Shared data models for the Ontology Reasoning stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OntologyEntry:
    class_id: str
    class_name: str
    embedding_text: str
    grounding_prompt: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalQuery:
    observed_object: str
    object_group: str
    description: str
    shape: str | None = None
    primary_color: str | None = None
    secondary_color: str | None = None
    material: str | None = None
    text: str | None = None
    symbol: str | None = None
    distinguishing_features: list[str] = field(default_factory=list)
    attached_to: str | None = None
    nearby_objects: list[str] = field(default_factory=list)
    road_side: str | None = None
    possible_function: str | None = None
    classification_hint: str | None = None
    scene_confidence: float | None = None
    embedding_text: str = ""


@dataclass
class CandidateMatch:
    entry: OntologyEntry
    embedding_score: float
    attribute_score: float = 0.0
    final_score: float = 0.0
    matched_attributes: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)


@dataclass
class OntologySearchResult:
    predicted_class: OntologyEntry
    candidates: list[CandidateMatch]
    retrieval_query: RetrievalQuery

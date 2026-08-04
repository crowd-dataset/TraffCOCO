"""
ontology_reasoner.py

Ontology reasoning module.

This module performs semantic ontology retrieval and knowledge-based
reranking for traffic object classification.

Pipeline

RetrievalQuery
        │
        ▼
Ontology Embedding Search
        │
        ▼
Top-K Candidate Retrieval
        │
        ▼
Knowledge-based Candidate Ranking
        │
        ▼
OntologySearchResult
"""

from __future__ import annotations

from custom_logger import CustomLogger

from annotation_pipeline.configs.settings import PipelineConfig

from annotation_pipeline.models.ontology.ontology_index import (
    OntologyIndex,
)

from annotation_pipeline.models.ontology.candidate_ranker import (
    CandidateRanker,
)

from annotation_pipeline.models.ontology.ontology_models import (
    RetrievalQuery,
    OntologySearchResult,
)

logger = CustomLogger(__name__)


class OntologyReasoner:
    """
    Performs ontology retrieval and knowledge-based reranking.
    """

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        """
        Initialize ontology reasoning module.
        """

        self.config = config

        logger.info(
            "Initializing ontology reasoner."
        )

        self.index = OntologyIndex(
            config,
        )

        self.index.initialize()

        self.ranker = CandidateRanker(
            config,
        )

        logger.info(
            "Ontology reasoner initialized."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def reason(
        self,
        query: RetrievalQuery,
    ) -> OntologySearchResult:
        """
        Perform ontology reasoning for an observed object.
        """
        if not query.embedding_text:
        
                raise ValueError(
                    "RetrievalQuery.embedding_text is empty."
                )

        logger.info(
            "Running ontology reasoning for '{}'.",
            query.observed_object,
        )

        candidates = self.index.search(
            query,
        )

        if not candidates:

            logger.warning(
                "No ontology candidates were retrieved."
            )

            return OntologySearchResult(
                predicted_class=None,
                candidates=[],
                retrieval_query=query,
            )

        ranked_candidates = self.ranker.rank_candidates(
            query=query,
            candidates=candidates,
        )

        result = OntologySearchResult(
            predicted_class=ranked_candidates[0],
            candidates=ranked_candidates,
            retrieval_query=query,
        )

        logger.info(
            "Predicted ontology class: {} (score={:.3f})",
            result.predicted_class.entry.class_name,
            result.predicted_class.final_score,
        )

        return result

        # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def ready(
        self,
    ) -> bool:
        """
        Return whether the ontology reasoner is ready.
        """

        return self.index.ready

    def __len__(
        self,
    ) -> int:
        """
        Return the number of ontology entries.
        """

        return len(self.index)

    def reload(
        self,
    ) -> None:
        """
        Reload the ontology index.
        """

        logger.info(
            "Reloading ontology index."
        )

        self.index.initialize()

        logger.info(
            "Ontology index reloaded."
        )
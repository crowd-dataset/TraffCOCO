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

        # ----------------------------------------------------------
        # Unreadable traffic sign override
        # ----------------------------------------------------------

        if self._is_unreadable_traffic_sign(query):

            unreadable_candidate = next(
                (
                    candidate
                    for candidate in ranked_candidates
                    if self._normalize(
                        candidate.entry.class_name
                    ) == "unreadable traffic sign"
                ),
                None,
            )

            if unreadable_candidate is not None:

                logger.info(
                    "Traffic sign pictogram/text is unreadable. "
                    "Overriding specific ontology prediction '{}' "
                    "with 'unreadable_traffic_sign'.",
                    ranked_candidates[0].entry.class_name,
                )

                predicted_class = unreadable_candidate

            else:

                logger.warning(
                    "Traffic sign has unreadable pictogram, "
                    "but 'unreadable_traffic_sign' is not present "
                    "in the ontology."
                )

                predicted_class = ranked_candidates[0]

        else:

            predicted_class = ranked_candidates[0]


        result = OntologySearchResult(
            predicted_class=predicted_class,
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

    def _normalize(
        self,
        value: str | None,
    ) -> str:
        """
        Normalize text for ontology class comparison.
        """

        if not value:
            return ""

        return (
            value
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .strip()
        )


    def _is_unreadable_traffic_sign(
        self,
        query: RetrievalQuery,
    ) -> bool:
        """
        Return True only when the object is a traffic sign and the
        identifying pictogram/symbol/icon is explicitly present but
        visually unreadable or cannot be identified.

        Generic uncertainty such as "unknown", "unclear", or
        "unidentified" must not by itself trigger the override.
        """

        if self._normalize(
            query.observed_object
        ) != "traffic sign":
            return False

        description = self._normalize(
            query.description
        )

        symbol = self._normalize(
            query.symbol
        )

        text = self._normalize(
            query.text
        )

        # ----------------------------------------------------------
        # Explicit pictogram/symbol/icon unreadability
        # ----------------------------------------------------------

        pictogram_terms = (
            "pictogram",
            "symbol",
            "icon",
            "graphic",
        )

        unreadable_terms = (
            "unreadable",
            "cannot be determined",
            "cannot be identified",
            "not identifiable",
            "unable to identify",
            "too small to identify",
            "too small to read",
            "not readable",
        )

        # Description must explicitly connect a visual symbol/pictogram
        # to the fact that it cannot be identified/read.
        for pictogram_term in pictogram_terms:
            for unreadable_term in unreadable_terms:
                if (
                    pictogram_term in description
                    and unreadable_term in description
                ):
                    return True

        # The structured symbol field is stronger evidence.
        # Only accept explicit unreadability here.
        if symbol:
            if any(
                term in symbol
                for term in unreadable_terms
            ):
                return True

        return False

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
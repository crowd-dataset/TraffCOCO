"""
ontology_index.py

FAISS-based semantic ontology index used by the KGFLM Traffic Annotation
Pipeline.

This module is responsible for loading the traffic ontology, generating
dense embeddings for every ontology class, constructing a FAISS index,
and performing semantic retrieval during the Ontology Reasoning stage.

The index is built once and reused throughout the pipeline.

Pipeline

Traffic Ontology JSON
        │
        ▼
Load Ontology Entries
        │
        ▼
Generate Embeddings
        │
        ▼
Build FAISS Index
        │
        ▼
Ontology Retrieval
        │
        ▼
Candidate Classes

Responsibilities

• Load ontology entries.
• Generate ontology embeddings.
• Build and maintain a FAISS index.
• Save and load the index.
• Retrieve the most semantically similar ontology classes.
"""

from __future__ import annotations

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

import pickle
import json
from pathlib import Path

import faiss
import numpy as np

from custom_logger import CustomLogger

from annotation_pipeline.configs.settings import PipelineConfig
from annotation_pipeline.models.ontology.embedding_model import (
    EmbeddingModel,
)
from annotation_pipeline.models.ontology.ontology_models import (
    CandidateMatch,
    OntologyEntry,
    RetrievalQuery,
)
from annotation_pipeline.utils.ontology_utils import (
    build_ontology_entries,
)

logger = CustomLogger(__name__)


# -------------------------------------------------------------------------
# Ontology Index
# -------------------------------------------------------------------------


class OntologyIndex:
    """
    FAISS-backed semantic ontology index.

    The ontology index maintains a dense vector representation of every
    ontology class and provides efficient nearest-neighbour retrieval
    during the Ontology Reasoning stage.

    The index is constructed from the ontology JSON using the configured
    sentence embedding model.

    Parameters
    ----------
    config
        Global pipeline configuration.

    Attributes
    ----------
    config
        Pipeline configuration.

    embedding_model
        Sentence embedding model used for ontology encoding.

    entries
        Ordered list of ontology entries.

    embeddings
        Matrix containing one embedding vector per ontology entry.

    index
        FAISS index used for similarity search.
    """

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        """
        Initialize the ontology index.

        Parameters
        ----------
        config
            Global pipeline configuration.
        """

        self.config = config

        self.embedding_model = EmbeddingModel(

            model_name=config.models.embedding_model_id,

            device=config.models.embedding_device,

            normalize_embeddings=True,
        )

        self.entries: list[OntologyEntry] = []

        self.embeddings: np.ndarray | None = None

        self.index: faiss.Index | None = None

        logger.info(
            "Initialized ontology index."
        )

    # ---------------------------------------------------------------------
    # Index Construction
    # ---------------------------------------------------------------------

    def build(self) -> None:
        """
        Build the ontology search index.

        The build process consists of:

        1. Loading ontology entries.
        2. Generating embeddings.
        3. Constructing the FAISS index.
        """

        logger.info(
            "Building ontology index."
        )

        self._load_ontology()

        self._generate_embeddings()

        self._build_faiss_index()

        if self.config.models.save_index:

            self.save()

        logger.info(
            "Ontology index built successfully."
        )

    # ---------------------------------------------------------------------
    # Helper Function
    # ---------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize the ontology index.

        Loads an existing FAISS index when available.
        Otherwise builds a new one.
        """

        if (

            not self.config.models.rebuild_index

            and

            self.config.paths.ontology_index.exists()

            and

            self.config.paths.ontology_metadata.exists()

        ):

            logger.info(

                "Loading existing ontology index."

            )

            self.load()

        else:

            logger.info(

                "Building ontology index."

            )

            self.build()

    # ---------------------------------------------------------------------
    # Ontology Loading
    # ---------------------------------------------------------------------

    def _load_ontology(self) -> None:
        ontology_path = self.config.paths.ontology_file

        logger.info(
            "Loading ontology from '{}'.",
            ontology_path,
        )

        with open(ontology_path, "r", encoding="utf-8") as f:
            ontology = json.load(f)

        self.entries = build_ontology_entries(
            ontology,
        )

        if not self.entries:
            raise ValueError(
                "Ontology contains no entries."
            )

        logger.info(
            "Loaded {} ontology entries.",
            len(self.entries),
        )

    # ---------------------------------------------------------------------
    # Embedding Generation
    # ---------------------------------------------------------------------

    def _generate_embeddings(self) -> None:
        """
        Generate embeddings for every ontology entry.

        The embedding text stored in each ontology class is encoded into
        dense vectors using the configured sentence embedding model.

        Raises
        ------
        RuntimeError
            If ontology entries have not been loaded.
        """

        if not self.entries:

            raise RuntimeError(
                "Ontology entries have not been loaded."
            )

        logger.info(
            "Generating ontology embeddings."
        )

        texts = [

            entry.embedding_text

            for entry in self.entries

        ]

        self.embeddings = self.embedding_model.encode_batch(
            texts,
        )

        logger.info(
            "Generated {} embedding vectors.",
            len(self.embeddings),
        )

    # ---------------------------------------------------------------------
    # FAISS Index Construction
    # ---------------------------------------------------------------------

    def _build_faiss_index(self) -> None:
        """
        Construct the FAISS similarity index.

        The ontology embeddings are inserted into an IndexFlatIP index.
        Since embeddings are normalized by the embedding model, inner
        product corresponds to cosine similarity.

        Raises
        ------
        RuntimeError
            If embeddings have not been generated.
        """

        if self.embeddings is None:

            raise RuntimeError(
                "Ontology embeddings have not been generated."
            )

        logger.info(
            "Building FAISS index."
        )

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension,
        )

        self.index.add(
            self.embeddings.astype(np.float32),
        )

        logger.info(
            "Indexed {} ontology classes.",
            self.index.ntotal,
        )

    # ---------------------------------------------------------------------
    # Index Persistence
    # ---------------------------------------------------------------------

    def save(self) -> None:
        """
        Save FAISS index and ontology metadata.
        """

        if self.index is None:

            raise RuntimeError(
                "Ontology index has not been built."
            )

        self.config.paths.ensure_output_dirs()

        logger.info(
            "Saving ontology index."
        )

        faiss.write_index(

            self.index,

            str(self.config.paths.ontology_index),

        )

        with open(

            self.config.paths.ontology_metadata,

            "wb",

        ) as file:

            logger.info(
                "Saving ontology metadata."
            )

            pickle.dump(
                {
                    "entries": self.entries,
                    "embedding_model": self.embedding_model.model_name,
                    "dimension": self.embedding_model.embedding_dimension,
                },
                file,
            )

        logger.info(
            "Ontology index saved successfully."
        )

    # ---------------------------------------------------------------------
    # Index Loading
    # ---------------------------------------------------------------------

    def load(self) -> None:
        """
        Load FAISS ontology index.
        """

        if not self.config.paths.ontology_index.exists():

            raise FileNotFoundError(

                self.config.paths.ontology_index

            )

        if not self.config.paths.ontology_metadata.exists():

            raise FileNotFoundError(

                self.config.paths.ontology_metadata

            )

        self.index = faiss.read_index(

            str(

                self.config.paths.ontology_index

            )

        )

        logger.info(
            "Loaded FAISS index."
        )

        with open(

            self.config.paths.ontology_metadata,

            "rb",

        ) as file:

            metadata = pickle.load(file)

            self.entries = metadata["entries"]

        logger.info(

            "Loaded {} ontology entries.",

            len(self.entries),

        )
    # ---------------------------------------------------------------------
    # Ontology Search
    # ---------------------------------------------------------------------

    def search(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
    ) -> list[CandidateMatch]:
        """
        Retrieve the most semantically similar ontology classes.

        Parameters
        ----------
        query
            Retrieval query constructed from the Scene Understanding
            output.

        top_k
            Number of ontology candidates to retrieve.

            If omitted, the configured default value is used.

        Returns
        -------
        list[CandidateMatch]
            Candidate ontology classes ranked by embedding similarity.

        Raises
        ------
        RuntimeError
            If the ontology index has not been loaded or built.
        """

        if self.index is None:

            raise RuntimeError(
                "Ontology index has not been initialized."
            )

        if top_k is None:

            top_k = self.config.models.top_k

        logger.info(
            "Searching ontology (top_k={}).",
            top_k,
        )

        query_embedding = self.embedding_model.encode(
            query.embedding_text,
        ).astype(np.float32)

        if query_embedding.ndim != 1:

            raise RuntimeError(
                "Embedding model returned an invalid embedding."
            )

        scores, indices = self.index.search(
            query_embedding.reshape(1, -1),
            top_k,
        )

        candidates: list[CandidateMatch] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:

                continue

            entry = self.entries[index]

            candidates.append(

                CandidateMatch(

                    entry=entry,

                    embedding_score=float(score),

                    attribute_score=0.0,

                    final_score=float(score),

                )

            )

        logger.info(
            "Retrieved {} ontology candidate(s).",
            len(candidates),
        )

        return candidates
    

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """
        Whether the ontology index is ready.
        """

        return (
            self.index is not None
            and
            len(self.entries) > 0
        )
       

    def __len__(
        self,
    ) -> int:
        """
        Return the number of ontology entries.

        Returns
        -------
        int
            Number of indexed ontology classes.
        """

        return len(self.entries)
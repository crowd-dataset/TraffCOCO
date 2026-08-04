"""
embedding_model.py

Sentence embedding wrapper used by the Ontology Reasoning stage.

This module provides a lightweight wrapper around a SentenceTransformer
model used to generate dense vector embeddings for ontology entries and
Scene Understanding queries.

Pipeline

Ontology Entry / Retrieval Query
            │
            ▼
SentenceTransformer
            │
            ▼
Normalized Embedding

Responsibilities

• Load the configured embedding model.
• Encode text into dense embeddings.
• Normalize embeddings for cosine similarity search.
"""

from __future__ import annotations

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from custom_logger import CustomLogger

logger = CustomLogger(__name__)


# -------------------------------------------------------------------------
# Embedding Model
# -------------------------------------------------------------------------


class EmbeddingModel:
    """
    SentenceTransformer embedding wrapper.

    Parameters
    ----------
    model_name
        Hugging Face SentenceTransformer model identifier.

    device
        Device used for embedding generation.

    normalize_embeddings
        Whether embeddings should be L2-normalized.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        normalize_embeddings: bool = True,
    ) -> None:

        self.model_name = model_name

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but unavailable. Falling back to CPU."
            )
            device = "cpu"

        self.device = device
        self.normalize_embeddings = normalize_embeddings

        logger.info(
            "Loading embedding model '{}' on {}.",
            self.model_name,
            self.device,
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate an embedding for a single text.

        Parameters
        ----------
        text
            Input text.

        Returns
        -------
        np.ndarray
            Embedding vector.
        """

        logger.debug(
            "Encoding text ({} characters).",
            len(text),
        )

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    # Batch Encoding
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Parameters
        ----------
        texts
            Collection of input texts.

        Returns
        -------
        np.ndarray
            Array of embeddings.
        """

        texts = list(texts)

        logger.info(
            "Encoding {} ontology entries.",
            len(texts),
        )

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        logger.info(
            "Embedding generation completed."
        )

        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def embedding_dimension(self) -> int:
        """
        Return the embedding dimension.

        Returns
        -------
        int
            Embedding vector dimension.
        """

        return self.model.get_sentence_embedding_dimension()

    def __repr__(self) -> str:
        """
        Return a concise representation of the embedding model.
        """

        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"dimension={self.embedding_dimension})"
        )
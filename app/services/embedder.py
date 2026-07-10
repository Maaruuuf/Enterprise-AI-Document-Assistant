"""
Embedding service.

Wraps the BGE sentence-transformer model. BGE models use an asymmetric
convention: passages/documents are embedded as-is, but queries must be
prefixed with an instruction string to get correct retrieval performance.
Forgetting this prefix doesn't error out — it just silently degrades
retrieval quality, so it's centralized here to avoid that mistake creeping
in elsewhere in the codebase.

The model is loaded once (module-level singleton via get_embedding_model)
since loading it repeatedly is expensive (~seconds) and unnecessary.
"""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


_model_instance: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load (once) and return the shared embedding model instance.

    Returns:
        Loaded SentenceTransformer model.

    Raises:
        EmbeddingError: If the model fails to load (e.g. network issue on
            first download, corrupted cache, out of memory).
    """
    global _model_instance
    if _model_instance is None:
        try:
            logger.info(f"Loading embedding model '{settings.EMBEDDING_MODEL_NAME}'...")
            _model_instance = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            actual_dim = _model_instance.get_sentence_embedding_dimension()
            if actual_dim != settings.EMBEDDING_DIM:
                raise EmbeddingError(
                    f"Model dimension mismatch: expected {settings.EMBEDDING_DIM}, got {actual_dim}. "
                    f"Update EMBEDDING_DIM in config to match the model."
                )
            logger.info(f"Embedding model loaded (dim={actual_dim})")
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    return _model_instance


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of passages/documents (no instruction prefix).

    Args:
        texts: List of raw text strings to embed.
        batch_size: Batch size for encoding.

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized.

    Raises:
        EmbeddingError: If texts is empty or encoding fails.
    """
    if not texts:
        raise EmbeddingError("Cannot embed an empty list of texts.")

    model = get_embedding_model()
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings
    except Exception as e:
        raise EmbeddingError(f"Failed to embed {len(texts)} text(s): {e}") from e


def embed_query(query: str) -> np.ndarray:
    """Embed a single user query, with BGE's required instruction prefix.

    Args:
        query: Raw user question text.

    Returns:
        1D numpy array (embedding_dim,), L2-normalized.

    Raises:
        EmbeddingError: If query is empty/whitespace-only or encoding fails.
    """
    cleaned = query.strip()
    if not cleaned:
        raise EmbeddingError("Cannot embed an empty query.")

    model = get_embedding_model()
    try:
        prefixed = settings.BGE_QUERY_INSTRUCTION + cleaned
        embedding = model.encode(prefixed, normalize_embeddings=True)
        return embedding
    except Exception as e:
        raise EmbeddingError(f"Failed to embed query: {e}") from e
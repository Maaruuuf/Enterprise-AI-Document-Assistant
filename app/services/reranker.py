"""
Cross-encoder reranking service.

Pinecone's bi-encoder search is fast but approximate — it embeds query and
documents independently, so it's good at narrowing a large corpus down to a
plausible candidate set, but not always precise about which candidate is
truly the best match. A cross-encoder scores (query, passage) pairs jointly,
which is far more accurate but too slow to run over an entire corpus — so it
only reranks the small candidate set Pinecone already returned.

Raw cross-encoder output is an unbounded logit, not a 0-1 probability. It's
passed through a sigmoid here so scores are interpretable as a confidence
value and can be compared against a fixed threshold.
"""

import logging
import math
from typing import List, Optional

from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.models.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class RerankingError(Exception):
    """Raised when reranking fails."""


_reranker_instance: Optional[CrossEncoder] = None


def get_reranker_model() -> CrossEncoder:
    """Load (once) and return the shared cross-encoder reranker instance.

    Returns:
        Loaded CrossEncoder model.

    Raises:
        RerankingError: If the model fails to load.
    """
    global _reranker_instance
    if _reranker_instance is None:
        try:
            logger.info(f"Loading reranker model '{settings.RERANKER_MODEL_NAME}'...")
            _reranker_instance = CrossEncoder(settings.RERANKER_MODEL_NAME)
            logger.info("Reranker model loaded.")
        except Exception as e:
            raise RerankingError(f"Failed to load reranker model: {e}") from e
    return _reranker_instance


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid, mapping any real number to (0, 1).

    Args:
        x: Raw score (e.g. a cross-encoder logit).

    Returns:
        Value in (0, 1).
    """
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def rerank(
    query: str,
    candidates: List[RetrievedChunk],
    top_k: int,
) -> List[RetrievedChunk]:
    """Rerank candidate chunks using a cross-encoder, attaching a normalized
    0-1 confidence score to each.

    Args:
        query: The user's raw question.
        candidates: Candidate chunks from vector search (any order).
        top_k: Number of top chunks to keep after reranking.

    Returns:
        Top-K RetrievedChunk list, sorted by cross-encoder confidence
        descending. Returns an empty list if candidates is empty (not an error
        — an empty retrieval result is a valid, expected case handled by the
        caller's confidence gate).

    Raises:
        RerankingError: If the reranker model itself fails during scoring.
    """
    if not candidates:
        return []

    model = get_reranker_model()

    try:
        pairs = [(query, c.text) for c in candidates]
        raw_scores = model.predict(pairs)
    except Exception as e:
        raise RerankingError(f"Cross-encoder scoring failed: {e}") from e

    scored = list(zip(candidates, raw_scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked: List[RetrievedChunk] = []
    for chunk, raw_score in scored[:top_k]:
        chunk.score = float(raw_score)
        chunk.confidence = _sigmoid(float(raw_score))
        reranked.append(chunk)

    return reranked


def compute_overall_confidence(reranked_chunks: List[RetrievedChunk]) -> float:
    """Compute a single overall confidence score from reranked chunks, used
    to gate whether the LLM should even be called (hallucination prevention).

    Weights the top chunk's confidence heavily (it drives the actual answer)
    while factoring in the average across all kept chunks (rewards consistent
    agreement rather than one lucky high-scoring outlier).

    Args:
        reranked_chunks: Reranked, confidence-attached chunks (descending order).

    Returns:
        Float in [0, 1]. Returns 0.0 if reranked_chunks is empty.
    """
    if not reranked_chunks:
        return 0.0

    top_confidence = reranked_chunks[0].confidence
    avg_confidence = sum(c.confidence for c in reranked_chunks) / len(reranked_chunks)
    return round(0.7 * top_confidence + 0.3 * avg_confidence, 4)
"""
Retrieval orchestration service.

Ties together query embedding, vector search, and reranking into a single
retrieve() call. This is the main entry point the API layer uses to go from
a raw question to a ranked, confidence-scored set of context chunks.
"""

import logging
from typing import List

from app.core.config import settings
from app.models.schemas import RetrievedChunk
from app.services import embedder, reranker, vector_store

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when the retrieval pipeline fails end-to-end."""


def retrieve(question: str) -> List[RetrievedChunk]:
    """Run the full retrieve -> rerank pipeline for a user question.

    Args:
        question: Raw user question text.

    Returns:
        Top reranked RetrievedChunk list (length <= settings.TOP_K_RERANK),
        each with a 'confidence' score attached. Returns an empty list if
        nothing is indexed yet or nothing relevant is found — this is a
        valid outcome, not an error, and should be handled by the caller's
        confidence-threshold gate.

    Raises:
        RetrievalError: If embedding, vector search, or reranking fail
            due to an infrastructure problem (not due to lack of results).
    """
    try:
        query_embedding = embedder.embed_query(question)
    except embedder.EmbeddingError as e:
        raise RetrievalError(f"Failed to embed query: {e}") from e

    try:
        candidates = vector_store.search(query_embedding, top_k=settings.TOP_K_RETRIEVE)
    except vector_store.VectorStoreError as e:
        raise RetrievalError(f"Vector search failed: {e}") from e

    if not candidates:
        logger.info(f"No candidates found for query: '{question[:80]}'")
        return []

    try:
        reranked = reranker.rerank(question, candidates, top_k=settings.TOP_K_RERANK)
    except reranker.RerankingError as e:
        raise RetrievalError(f"Reranking failed: {e}") from e

    return reranked
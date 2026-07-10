"""
Retrieval orchestration service.

Simplified retrieval: no cross-encoder reranking step. Relies directly on
Pinecone's cosine similarity score (query and passage embeddings are both
L2-normalized BGE vectors, so this score is a true cosine similarity in
roughly [0, 1] for relevant matches).

Trade-off vs. the reranked version: a bi-encoder similarity score is a
coarser relevance signal than a cross-encoder's joint (query, passage)
score. This keeps the service lightweight (one embedding model instead of
two), which matters on memory-constrained deployments — but expect lower
precision, and re-tune CONFIDENCE_THRESHOLD against real observed scores.
"""

import logging
from typing import List

from app.core.config import settings
from app.models.schemas import RetrievedChunk
from app.services import embedder, vector_store

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when the retrieval pipeline fails end-to-end."""


def retrieve(question: str) -> List[RetrievedChunk]:
    """Embed the question, search Pinecone, and return the top chunks by
    raw cosine similarity — no reranking step.

    Args:
        question: Raw user question text.

    Returns:
        Top RetrievedChunk list (length <= settings.TOP_K_CONTEXT), sorted
        by score descending, each with `confidence` set to its raw cosine
        similarity. Returns an empty list if nothing is indexed yet or
        nothing relevant is found — a valid outcome, not an error.

    Raises:
        RetrievalError: If embedding or vector search fail due to an
            infrastructure problem (not due to lack of results).
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

    # Pinecone already returns matches sorted by score descending; sort
    # explicitly anyway so this doesn't silently depend on that guarantee.
    candidates.sort(key=lambda c: c.score, reverse=True)

    top_chunks = candidates[: settings.TOP_K_CONTEXT]
    for chunk in top_chunks:
        chunk.confidence = chunk.score

    return top_chunks


def compute_overall_confidence(top_chunks: List[RetrievedChunk]) -> float:
    """Gate confidence purely on the single most confident (top-ranked)
    chunk's raw similarity score — simple threshold, no weighted blending.

    Args:
        top_chunks: Retrieved chunks, sorted descending by score.

    Returns:
        The top chunk's confidence, or 0.0 if the list is empty.
    """
    if not top_chunks:
        return 0.0
    return round(top_chunks[0].confidence, 4)
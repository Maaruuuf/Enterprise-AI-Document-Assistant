"""
Vector store service (Pinecone).

Handles index lifecycle (create-if-missing), upserting chunk embeddings with
their citation metadata, and raw similarity search. This is the only module
that talks to Pinecone directly — everything else goes through these
functions, so a future vector-DB swap only touches this file.
"""

import logging
import time
from typing import List, Optional

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.models.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when a Pinecone operation fails (connection, index, query, upsert)."""


_pinecone_client: Optional[Pinecone] = None
_index_instance = None


def get_pinecone_client() -> Pinecone:
    """Return a shared Pinecone client instance, creating it on first use.

    Returns:
        Initialized Pinecone client.

    Raises:
        VectorStoreError: If the client cannot be initialized (e.g. bad API key format).
    """
    global _pinecone_client
    if _pinecone_client is None:
        try:
            _pinecone_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone client: {e}") from e
    return _pinecone_client


def ensure_index_exists() -> None:
    """Create the configured Pinecone index if it doesn't already exist.

    Safe to call multiple times — no-ops if the index is already present.

    Raises:
        VectorStoreError: If index creation or the readiness check fails.
    """
    pc = get_pinecone_client()
    try:
        existing = [idx["name"] for idx in pc.list_indexes()]
        if settings.PINECONE_INDEX_NAME in existing:
            logger.info(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' already exists.")
            return

        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.PINECONE_CLOUD, region=settings.PINECONE_REGION),
        )

        max_wait_seconds = 60
        waited = 0
        while waited < max_wait_seconds:
            if pc.describe_index(settings.PINECONE_INDEX_NAME).status["ready"]:
                break
            time.sleep(1)
            waited += 1
        else:
            raise VectorStoreError(
                f"Index '{settings.PINECONE_INDEX_NAME}' did not become ready within {max_wait_seconds}s."
            )

        logger.info(f"Pinecone index '{settings.PINECONE_INDEX_NAME}' created and ready.")
    except VectorStoreError:
        raise
    except Exception as e:
        raise VectorStoreError(f"Failed to ensure index exists: {e}") from e


def get_index():
    """Return a shared handle to the configured Pinecone index.

    Returns:
        Pinecone Index object.

    Raises:
        VectorStoreError: If the index cannot be reached.
    """
    global _index_instance
    if _index_instance is None:
        pc = get_pinecone_client()
        try:
            _index_instance = pc.Index(settings.PINECONE_INDEX_NAME)
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Pinecone index: {e}") from e
    return _index_instance


def upsert_chunks(chunks: List[Chunk], embeddings: np.ndarray, batch_size: int = 100) -> int:
    """Upsert chunk embeddings + metadata into Pinecone, in batches.

    Args:
        chunks: List of Chunk objects (order must match embeddings rows).
        embeddings: numpy array of shape (len(chunks), embedding_dim).
        batch_size: Vectors per upsert request (Pinecone recommends <=100).

    Returns:
        Total number of vectors upserted.

    Raises:
        VectorStoreError: If chunks/embeddings are mismatched, or if upsert fails.
    """
    if len(chunks) != len(embeddings):
        raise VectorStoreError(
            f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})."
        )
    if not chunks:
        raise VectorStoreError("No chunks provided to upsert.")

    index = get_index()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk.chunk_id,
            "values": embedding.tolist(),
            "metadata": {
                "document": chunk.document_name,
                "page_start": int(chunk.page_start),
                "page_end": int(chunk.page_end),
                "text": chunk.text,
            },
        })

    total = len(vectors)
    upserted = 0
    try:
        for i in range(0, total, batch_size):
            batch = vectors[i: i + batch_size]
            index.upsert(vectors=batch)
            upserted += len(batch)
            logger.info(f"Upserted {upserted}/{total} vectors")
    except Exception as e:
        raise VectorStoreError(
            f"Upsert failed after {upserted}/{total} vectors: {e}"
        ) from e

    time.sleep(1)  # brief pause for eventual consistency before any immediate query
    return upserted


def search(query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
    """Run a similarity search against the Pinecone index.

    Args:
        query_embedding: 1D embedding vector for the user's query.
        top_k: Number of candidate matches to retrieve.

    Returns:
        List of RetrievedChunk, ordered by similarity score descending.
        Returns an empty list if the index has no vectors yet (not an error).

    Raises:
        VectorStoreError: If the query itself fails (connection, timeout, etc.).
    """
    index = get_index()
    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
        )
    except Exception as e:
        raise VectorStoreError(f"Pinecone query failed: {e}") from e

    candidates: List[RetrievedChunk] = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        try:
            candidates.append(
                RetrievedChunk(
                    chunk_id=match["id"],
                    document=meta["document"],
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    text=meta["text"],
                    score=float(match["score"]),
                )
            )
        except (KeyError, ValueError, TypeError) as e:
            # A malformed metadata entry shouldn't crash the whole search
            logger.warning(f"Skipping malformed match '{match.get('id', 'unknown')}': {e}")
            continue

    return candidates


def get_index_stats() -> dict:
    """Get current index statistics (used by /health and /documents endpoints).

    Returns:
        Dict with at least 'total_vector_count'. Returns a zeroed dict on failure
        rather than raising, since this is used for health checks that shouldn't
        themselves crash the app.
    """
    try:
        index = get_index()
        stats = index.describe_index_stats()
        return {
            "total_vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", settings.EMBEDDING_DIM),
        }
    except Exception as e:
        logger.error(f"Failed to fetch index stats: {e}")
        return {"total_vector_count": 0, "dimension": settings.EMBEDDING_DIM}
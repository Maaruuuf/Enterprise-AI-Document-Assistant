"""
API route definitions.

Each route is a thin wrapper: validate via Pydantic (handled automatically
by FastAPI), delegate to the service layer, and translate service-layer
exceptions into appropriate HTTP responses. No business logic lives here.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import DocumentsResponse, HealthResponse, QueryRequest, QueryResponse
from app.services import rag_pipeline, vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse, tags=["Assistant"])
def query(request: QueryRequest) -> QueryResponse:
    """Ask a natural language question against the indexed documents.

    Returns a grounded answer with source citations and a confidence score.
    If retrieval confidence is below the configured threshold, the LLM is
    not called and a clear "insufficient information" response is returned
    instead of a fabricated answer.
    """
    try:
        return rag_pipeline.answer_question(
            question=request.question,
            session_id=request.session_id,
        )
    except rag_pipeline.RAGPipelineError as e:
        logger.error(f"Pipeline error for question '{request.question[:80]}': {e}")
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unable to process this request. Please try again shortly.",
        ) from e
    except Exception as e:
        # Catch-all: never let an unexpected error leak a raw stack trace to the client.
        logger.exception(f"Unexpected error handling query: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") from e


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Health check endpoint. Reports whether the vector store is reachable
    and how many chunks are currently indexed.

    Always returns HTTP 200 — the 'status' field itself communicates
    degraded state, so monitoring tools can distinguish "app is up but
    dependency is down" from "app is completely unreachable".
    """
    try:
        stats = vector_store.get_index_stats()
        pinecone_connected = stats["total_vector_count"] >= 0
        return HealthResponse(
            status="healthy" if pinecone_connected else "degraded",
            pinecone_connected=pinecone_connected,
            documents_indexed=stats["total_vector_count"],
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(status="degraded", pinecone_connected=False, documents_indexed=0)


@router.get("/documents", response_model=DocumentsResponse, tags=["System"])
def documents() -> DocumentsResponse:
    """List the total number of indexed chunks currently in the vector store.

    Note: Pinecone doesn't expose distinct source filenames via stats alone
    without a metadata scan, so this endpoint reports chunk count as a proxy
    for "is the knowledge base populated". Document-level listing would
    require an additional metadata index if needed later.
    """
    try:
        stats = vector_store.get_index_stats()
        return DocumentsResponse(documents=[], total_chunks=stats["total_vector_count"])
    except Exception as e:
        logger.error(f"Failed to fetch document stats: {e}")
        raise HTTPException(status_code=503, detail="Unable to retrieve index information.") from e
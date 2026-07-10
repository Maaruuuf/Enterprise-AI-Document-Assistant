"""
End-to-end RAG orchestration.

This is the single entry point the API layer calls to go from a raw question
(+ optional session_id) to a fully formed answer. It wires together:
retrieval -> confidence gate (hallucination prevention) -> conversation
memory -> LLM generation -> source deduplication.

Kept separate from api/routes.py so the core business logic is testable
independently of FastAPI/HTTP concerns.
"""

import logging
from typing import List

from app.core.config import settings
from app.models.schemas import QueryResponse, SourceReference
from app.services import llm_service, memory, retriever

logger = logging.getLogger(__name__)

NO_INFO_MESSAGE = "I couldn't find sufficient information in the provided documents to answer this question."


class RAGPipelineError(Exception):
    """Raised when the pipeline fails due to an infrastructure problem
    (not due to lack of retrieved information, which is a valid outcome)."""


def _dedupe_sources(reranked_chunks) -> List[SourceReference]:
    """Merge chunks from the same document into one source entry covering
    all referenced pages, for cleaner display to the end user.

    Args:
        reranked_chunks: Reranked RetrievedChunk list.

    Returns:
        List of SourceReference, one per unique document.
    """
    merged: dict = {}
    for chunk in reranked_chunks:
        pages = set(range(chunk.page_start, chunk.page_end + 1))
        merged.setdefault(chunk.document, set()).update(pages)

    return [
        SourceReference(document=doc, pages=sorted(pages))
        for doc, pages in merged.items()
    ]


def answer_question(question: str, session_id: str | None) -> QueryResponse:
    """Run the full RAG pipeline for a single user question.

    Args:
        question: Raw user question text (already validated non-empty by
            the Pydantic QueryRequest schema at the API layer).
        session_id: Optional client-provided session ID for conversation
            memory continuity. If None/unknown/expired, a new session starts.

    Returns:
        QueryResponse with answer, confidence, sources, session_id, and title.

    Raises:
        RAGPipelineError: If retrieval or generation fail due to an
            infrastructure issue. The API layer is responsible for catching
            this and returning an appropriate HTTP error response — this
            function never returns a fabricated answer on failure.
    """
    session = memory.get_or_create_session(session_id)

    try:
        reranked_chunks = retriever.retrieve(question)
    except retriever.RetrievalError as e:
        raise RAGPipelineError(f"Retrieval failed: {e}") from e

    overall_confidence = _compute_confidence(reranked_chunks)

    if overall_confidence < settings.CONFIDENCE_THRESHOLD:
        logger.info(
            f"Confidence {overall_confidence:.4f} below threshold "
            f"{settings.CONFIDENCE_THRESHOLD} — returning no-info response without calling LLM."
        )
        memory.add_turn(session, question, NO_INFO_MESSAGE)
        return QueryResponse(
            answer=NO_INFO_MESSAGE,
            confidence=overall_confidence,
            sources=[],
            session_id=session.session_id,
            session_title=session.title,
            llm_model_used=None,
        )

    history = memory.get_recent_history(session, max_turns=2)

    try:
        answer_text, model_used = llm_service.generate_answer(question, reranked_chunks, history)
    except llm_service.LLMGenerationError as e:
        raise RAGPipelineError(f"Answer generation failed: {e}") from e

    memory.add_turn(session, question, answer_text)

    # Suppress sources if the LLM itself concluded it had no grounding,
    # even though retrieval confidence passed the gate — otherwise the UI
    # shows source pages next to an answer that isn't actually based on them.
    sources = [] if answer_text.strip() == NO_INFO_MESSAGE else _dedupe_sources(reranked_chunks)


    return QueryResponse(
        answer=answer_text,
        confidence=overall_confidence,
        sources=sources,
        session_id=session.session_id,
        session_title=session.title,
        llm_model_used=model_used,
    )


def _compute_confidence(retrieved_chunks) -> float:
    """Delegate to retriever's confidence computation (top-chunk score)."""
    return retriever.compute_overall_confidence(retrieved_chunks)
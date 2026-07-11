"""
Data models used across the application.

- Internal dataclasses (PageContent, Chunk, RetrievedChunk) represent data
  flowing through the ingestion/retrieval pipeline.
- Pydantic models (QueryRequest, QueryResponse, etc.) define the public API
  contract and get automatic validation + OpenAPI docs from FastAPI.
"""

from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# ---------- Internal pipeline dataclasses ----------

@dataclass
class PageContent:
    """Raw text extracted from a single PDF page.

    extraction_method distinguishes text pulled from the PDF's native text
    layer ("native") from text recovered via OCR on a rasterized page
    ("ocr") — the latter happens for scanned/image-only pages (common in
    older or scanned policy documents like labour act handbooks).
    """
    document_name: str
    page_number: int
    text: str
    start_char: int
    end_char: int
    extraction_method: str = "native"        # "native" or "ocr"
    ocr_confidence: Optional[float] = None    # None for native pages, 0-100 for OCR'd pages


@dataclass
class Chunk:
    """A retrieval unit: a slice of text with provenance metadata."""
    chunk_id: str
    document_name: str
    text: str
    page_start: int
    page_end: int
    start_char: int
    end_char: int


@dataclass
class RetrievedChunk:
    """A chunk returned from vector search, carrying its relevance score."""
    chunk_id: str
    document: str
    page_start: int
    page_end: int
    text: str
    score: float
    confidence: Optional[float] = None


@dataclass
class ConversationTurn:
    """A single question-answer exchange within a session, used to give the
    LLM short-term memory of the conversation for follow-up questions."""
    question: str
    answer: str
    timestamp: datetime


@dataclass
class Session:
    """In-memory conversation session state."""
    session_id: str
    title: Optional[str] = None                    # auto-generated after first turn
    turns: List[ConversationTurn] = None
    created_at: datetime = None
    last_active_at: datetime = None

    def __post_init__(self):
        if self.turns is None:
            self.turns = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_active_at is None:
            self.last_active_at = datetime.utcnow()


# ---------- Public API schemas ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(
        default=None,
        description="Omit to start a new session; the response will include the created session_id."
    )

class SourceReference(BaseModel):
    document: str
    pages: List[int]


class QueryResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: List[SourceReference]
    session_id: str
    session_title: Optional[str] = None
    llm_model_used: Optional[str] = None   # NEW: transparency into which fallback model answered

class HealthResponse(BaseModel):
    status: str
    pinecone_connected: bool
    documents_indexed: int


class DocumentsResponse(BaseModel):
    documents: List[str]
    total_chunks: int
"""
FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload

Run in production (e.g. inside Docker) with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=(
        "Enterprise AI Document Assistant — a RAG-based API for answering "
        "natural language questions over company policy documents, with "
        "source citations and confidence-gated hallucination prevention."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    """Log startup confirmation. Does NOT eagerly load the embedding/reranker
    models here — those load lazily on first use (see embedder.py/reranker.py
    singletons) so the API becomes available quickly even if model download
    is slow, rather than blocking startup."""
    logger.info(f"{settings.API_TITLE} v{settings.API_VERSION} starting up...")


@app.get("/", tags=["System"])
def root() -> dict:
    """Root endpoint — simple landing response confirming the API is running."""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
    }
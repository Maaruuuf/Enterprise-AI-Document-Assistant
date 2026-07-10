"""
One-time (or re-runnable) ingestion script.

Loads all PDFs from the documents/ directory, chunks them, embeds the
chunks, and upserts them into Pinecone. Run this once before starting the
API, and re-run whenever documents/ changes.

Usage:
    python -m scripts.ingest
"""

import logging
import sys
from pathlib import Path

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.services import chunker, embedder, pdf_processor, vector_store

configure_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    docs_dir = Path(settings.DOCUMENTS_DIR)

    logger.info(f"Starting ingestion from '{docs_dir}'...")

    try:
        all_pages = pdf_processor.load_all_documents(docs_dir)
    except pdf_processor.PDFProcessingError as e:
        logger.error(f"Ingestion aborted — PDF processing failed: {e}")
        sys.exit(1)

    try:
        all_chunks = chunker.chunk_all_documents(
            all_pages,
            chunk_size_words=settings.CHUNK_SIZE_WORDS,
            overlap_words=settings.CHUNK_OVERLAP_WORDS,
        )
    except ValueError as e:
        logger.error(f"Ingestion aborted — chunking failed: {e}")
        sys.exit(1)

    try:
        texts = [c.text for c in all_chunks]
        embeddings = embedder.embed_texts(texts)
    except embedder.EmbeddingError as e:
        logger.error(f"Ingestion aborted — embedding failed: {e}")
        sys.exit(1)

    try:
        vector_store.ensure_index_exists()
        upserted_count = vector_store.upsert_chunks(all_chunks, embeddings)
    except vector_store.VectorStoreError as e:
        logger.error(f"Ingestion aborted — vector store operation failed: {e}")
        sys.exit(1)

    logger.info(f"✅ Ingestion complete. {upserted_count} chunks indexed into Pinecone.")


if __name__ == "__main__":
    main()
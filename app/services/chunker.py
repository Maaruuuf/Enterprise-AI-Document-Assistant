"""
Chunking service.

Splits per-page document text into overlapping word-window chunks while
preserving page attribution, so every chunk can be traced back to the exact
page(s) it came from (required for citations).

Chunking is performed strictly per-document — chunks never span across
different source files, only across page boundaries within the same file.
"""

import logging
import re
from itertools import groupby
from typing import List, Tuple

from app.models.schemas import Chunk, PageContent

logger = logging.getLogger(__name__)


def _slugify(filename: str) -> str:
    """Turn a PDF filename into a short, filesystem/ID-safe slug.

    Args:
        filename: Original PDF filename, e.g. "Employee Handbook.pdf".

    Returns:
        Lowercase, underscore-joined slug truncated to 40 chars.
    """
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return name[:40] if name else "document"


def chunk_document(
    pages: List[PageContent],
    chunk_size_words: int,
    overlap_words: int,
) -> List[Chunk]:
    """Chunk all pages of a single document into overlapping word-window chunks.

    Args:
        pages: Ordered PageContent list, all belonging to the same document.
        chunk_size_words: Target number of words per chunk. Must be > 0.
        overlap_words: Overlap words between consecutive chunks. Must be >= 0
            and less than chunk_size_words (otherwise it's clamped safely).

    Returns:
        List of Chunk objects with correct page_start/page_end attribution.
        Returns an empty list if `pages` is empty.
    """
    if not pages:
        return []

    if chunk_size_words <= 0:
        logger.warning(f"Invalid chunk_size_words={chunk_size_words}, defaulting to 400")
        chunk_size_words = 400

    if overlap_words < 0 or overlap_words >= chunk_size_words:
        logger.warning(
            f"Invalid overlap_words={overlap_words} for chunk_size_words={chunk_size_words}, "
            f"clamping to {chunk_size_words // 4}"
        )
        overlap_words = chunk_size_words // 4

    document_name = pages[0].document_name

    word_page_pairs: List[Tuple[str, int]] = []
    for page in pages:
        words = page.text.split()
        word_page_pairs.extend((w, page.page_number) for w in words)

    if not word_page_pairs:
        return []

    chunks: List[Chunk] = []
    step = max(chunk_size_words - overlap_words, 1)
    chunk_index = 0
    doc_slug = _slugify(document_name)

    for start_idx in range(0, len(word_page_pairs), step):
        window = word_page_pairs[start_idx: start_idx + chunk_size_words]
        if not window:
            break

        words = [w for w, _ in window]
        page_numbers = [p for _, p in window]
        chunk_text = " ".join(words)

        chunks.append(
            Chunk(
                chunk_id=f"{doc_slug}_chunk_{chunk_index}",
                document_name=document_name,
                text=chunk_text,
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                start_char=0,
                end_char=len(chunk_text),
            )
        )
        chunk_index += 1

        if start_idx + chunk_size_words >= len(word_page_pairs):
            break

    return chunks


def chunk_all_documents(
    all_pages: List[PageContent],
    chunk_size_words: int,
    overlap_words: int,
) -> List[Chunk]:
    """Group pages by document and chunk each document independently.

    Args:
        all_pages: Flat list of PageContent across all documents. Must be
            grouped by document (i.e. same document's pages are contiguous) —
            this holds true when produced by pdf_processor.load_all_documents.
        chunk_size_words: Target words per chunk.
        overlap_words: Overlap words between consecutive chunks.

    Returns:
        Flat list of Chunk objects across all documents.

    Raises:
        ValueError: If all_pages is empty.
    """
    if not all_pages:
        raise ValueError("Cannot chunk an empty page list — no documents were loaded.")

    all_chunks: List[Chunk] = []

    for doc_name, doc_pages_iter in groupby(all_pages, key=lambda p: p.document_name):
        doc_pages = list(doc_pages_iter)
        try:
            doc_chunks = chunk_document(doc_pages, chunk_size_words, overlap_words)
            all_chunks.extend(doc_chunks)
            logger.info(f"'{doc_name}': {len(doc_chunks)} chunks created")
        except Exception as e:
            # A chunking failure on one document shouldn't abort the whole batch
            logger.error(f"Failed to chunk '{doc_name}': {e}")

    if not all_chunks:
        raise ValueError("Chunking produced zero chunks across all documents.")

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
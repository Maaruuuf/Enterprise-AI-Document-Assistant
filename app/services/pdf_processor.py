"""
PDF text extraction service.

Extracts per-page text from PDF documents using PyMuPDF, preserving page
boundaries so downstream chunking can attribute any piece of text back to
its exact source page for citation purposes.

Designed to be fault-tolerant: a single corrupt/unreadable/scanned PDF
should never crash the whole ingestion run — it gets logged and skipped.
"""

import logging
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from app.models.schemas import PageContent

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed at all (unreadable, corrupt, no text)."""


def _clean_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph structure.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text with collapsed whitespace and stripped null bytes.
    """
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages_from_pdf(pdf_path: Path) -> List[PageContent]:
    """Extract per-page text from a single PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageContent objects, one per non-empty page.

    Raises:
        PDFProcessingError: If the file doesn't exist, can't be opened,
            or has no extractable text (e.g. scanned/image-only PDF).
    """
    if not pdf_path.exists():
        raise PDFProcessingError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        # Covers corrupt files, password-protected PDFs, unsupported formats, etc.
        raise PDFProcessingError(f"Failed to open PDF '{pdf_path.name}': {e}") from e

    pages: List[PageContent] = []
    running_offset = 0

    try:
        for page_index in range(len(doc)):
            try:
                page = doc[page_index]
                raw_text = page.get_text("text")
            except Exception as e:
                # A single bad page shouldn't kill the whole document
                logger.warning(f"Failed to extract page {page_index + 1} of '{pdf_path.name}': {e}")
                continue

            cleaned_text = _clean_text(raw_text)
            if not cleaned_text:
                continue  # blank/separator page — skip silently, not an error

            start = running_offset
            end = start + len(cleaned_text)

            pages.append(
                PageContent(
                    document_name=pdf_path.name,
                    page_number=page_index + 1,
                    text=cleaned_text,
                    start_char=start,
                    end_char=end,
                )
            )
            running_offset = end + 1
    finally:
        doc.close()

    if not pages:
        raise PDFProcessingError(
            f"No extractable text found in '{pdf_path.name}' — likely a scanned/image-only PDF."
        )

    logger.info(f"Extracted {len(pages)} pages from '{pdf_path.name}'")
    return pages


def load_all_documents(docs_dir: Path) -> List[PageContent]:
    """Extract pages from every PDF in a directory.

    Individual PDF failures are logged and skipped rather than raised, so
    that ingestion of a batch of documents is resilient to one bad file.

    Args:
        docs_dir: Directory containing .pdf files.

    Returns:
        Flat list of PageContent across all successfully processed documents.

    Raises:
        PDFProcessingError: If the directory doesn't exist, or if NO documents
            could be processed at all (i.e. every single PDF failed).
    """
    if not docs_dir.exists():
        raise PDFProcessingError(f"Documents directory not found: {docs_dir}")

    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise PDFProcessingError(f"No PDF files found in '{docs_dir}'")

    all_pages: List[PageContent] = []
    failed_files: List[str] = []

    for pdf_path in pdf_files:
        try:
            pages = extract_pages_from_pdf(pdf_path)
            all_pages.extend(pages)
        except PDFProcessingError as e:
            logger.error(f"Skipping '{pdf_path.name}': {e}")
            failed_files.append(pdf_path.name)

    if not all_pages:
        raise PDFProcessingError(
            f"All {len(pdf_files)} PDF(s) failed to process. Failed files: {failed_files}"
        )

    if failed_files:
        logger.warning(f"Ingestion completed with {len(failed_files)} failed file(s): {failed_files}")

    logger.info(f"Total pages extracted across all documents: {len(all_pages)}")
    return all_pages
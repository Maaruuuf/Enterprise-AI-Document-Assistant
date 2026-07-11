"""
PDF text extraction service.

Extracts per-page text from PDF documents using PyMuPDF, preserving page
boundaries so downstream chunking can attribute any piece of text back to
its exact source page for citation purposes.

Pages with no usable native text layer (scanned/image-only pages — common
in older or scanned policy documents, e.g. government labour act handbooks)
fall back to OCR via pytesseract on a rasterized render of that page. The
average OCR confidence per page is recorded so low-confidence pages can be
flagged for manual review rather than silently trusted.

Designed to be fault-tolerant: a single corrupt/unreadable PDF, or a single
bad page, should never crash the whole ingestion run — it gets logged and
skipped.
"""

import io
import logging
import re
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from app.core.config import settings
from app.models.schemas import PageContent

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed at all (unreadable, corrupt, no text)."""


def _clean_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph structure.

    Args:
        text: Raw text extracted from a PDF page (native or OCR).

    Returns:
        Cleaned text with collapsed whitespace and stripped null bytes.
    """
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_page(page: "fitz.Page", dpi: int) -> Tuple[str, float]:
    """Rasterize a PDF page and run OCR on it, returning both the extracted
    text and the average per-word OCR confidence for that page.

    Args:
        page: A PyMuPDF page object.
        dpi: Render resolution — higher improves OCR accuracy at the cost
            of speed.

    Returns:
        Tuple of (ocr_text, average_confidence_0_to_100). Confidence is 0.0
        if no words were detected at all.

    Raises:
        PDFProcessingError: If rasterization or OCR itself fails (e.g.
            tesseract binary not installed/found on the host).
    """
    try:
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        text = pytesseract.image_to_string(img)

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data["conf"] if c not in ("-1", -1)]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return text, avg_confidence
    except pytesseract.TesseractNotFoundError as e:
        raise PDFProcessingError(
            "Tesseract OCR binary not found on this system. Install it "
            "(e.g. `apt-get install tesseract-ocr` on Debian/Ubuntu) or "
            "ensure it's on PATH."
        ) from e
    except Exception as e:
        raise PDFProcessingError(f"OCR failed on page: {e}") from e


def extract_pages_from_pdf(pdf_path: Path) -> List[PageContent]:
    """Extract per-page text from a single PDF file, falling back to OCR
    for pages with no usable native text layer.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageContent objects, one per non-empty page, each tagged
        with its extraction_method ("native"/"ocr") and ocr_confidence.

    Raises:
        PDFProcessingError: If the file doesn't exist, can't be opened, or
            has no extractable text even after OCR.
    """
    if not pdf_path.exists():
        raise PDFProcessingError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFProcessingError(f"Failed to open PDF '{pdf_path.name}': {e}") from e

    pages: List[PageContent] = []
    running_offset = 0
    native_count = 0
    ocr_count = 0
    blank_count = 0
    low_confidence_pages: List[Tuple[int, float]] = []

    try:
        for page_index in range(len(doc)):
            try:
                page = doc[page_index]
                raw_text = page.get_text("text")
                method = "native"
                confidence = None

                if len(raw_text.strip()) < settings.OCR_MIN_CHARS:
                    raw_text, confidence = _ocr_page(page, dpi=settings.OCR_RENDER_DPI)
                    method = "ocr"
                    ocr_count += 1
                    if confidence < settings.OCR_LOW_CONFIDENCE_THRESHOLD:
                        low_confidence_pages.append((page_index + 1, confidence))
                else:
                    native_count += 1

            except PDFProcessingError as e:
                # A single bad page (e.g. OCR failure) shouldn't kill the whole document
                logger.warning(f"Failed to extract page {page_index + 1} of '{pdf_path.name}': {e}")
                continue

            cleaned_text = _clean_text(raw_text)
            if not cleaned_text:
                blank_count += 1
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
                    extraction_method=method,
                    ocr_confidence=confidence,
                )
            )
            running_offset = end + 1
    finally:
        doc.close()

    if not pages:
        raise PDFProcessingError(
            f"No extractable text found in '{pdf_path.name}' — even OCR failed."
        )

    logger.info(
        f"'{pdf_path.name}': {len(pages)} pages extracted "
        f"({native_count} native, {ocr_count} OCR'd, {blank_count} blank/skipped)"
    )
    for page_num, conf in low_confidence_pages:
        logger.warning(
            f"'{pdf_path.name}' page {page_num}: low OCR confidence ({conf:.1f}%) — "
            f"consider manual review of this page's extracted text."
        )

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
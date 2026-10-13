"""
VerityAI - Document Ingestion Module
Phase 2: Local Document Ingestion

This module provides reliable, local, in-memory document ingestion for plain-text (.txt)
and PDF (.pdf) files. It validates input files, enforces size and format constraints,
extracts text cleanly, and structures the output for citation grounding in later phases.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


# ==============================================================================
# Exceptions
# ==============================================================================


class IngestionError(Exception):
    """Base exception for document ingestion failures."""

    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file format is not supported."""

    pass


class FileSizeLimitExceededError(IngestionError):
    """Raised when an uploaded file exceeds the configured size ceiling."""

    pass


class EmptyDocumentError(IngestionError):
    """Raised when an uploaded document has 0 bytes or contains only whitespace."""

    pass


class InvalidEncodingError(IngestionError):
    """Raised when a text document cannot be decoded as valid UTF-8."""

    pass


class CorruptedFileError(IngestionError):
    """Raised when a document is corrupted or malformed."""

    pass


class EncryptedPDFError(IngestionError):
    """Raised when a PDF requires a password to open."""

    pass


class ScannedPDFError(IngestionError):
    """Raised when a PDF contains no extractable text (e.g., scanned images)."""

    pass


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass(frozen=True)
class PageContent:
    """Represents a single document page with 1-indexed numbering for citation provenance.

    Attributes:
        page_number: 1-indexed page number (Page 1, Page 2, ...).
        text: Extracted plain text content for this page.
        char_count: Total character count of this page's text.
    """

    page_number: int
    text: str
    char_count: int


@dataclass(frozen=True)
class DocumentResult:
    """Immutable result structure containing extracted document text and metadata.

    Attributes:
        filename: Sanitized original filename.
        doc_type: Detected document format ("txt" or "pdf").
        total_pages: Total number of pages (1 for plain text, >= 1 for PDF).
        extracted_text: Full concatenated text across all pages.
        pages: List of per-page PageContent instances with 1-based indexing.
        char_count: Total character count across the full document.
        status: Ingestion status ("success" or "error").
        error_message: User-facing error message if status is "error".
    """

    filename: str
    doc_type: str
    total_pages: int
    extracted_text: str
    pages: List[PageContent]
    char_count: int
    status: str = "success"
    error_message: Optional[str] = None


# ==============================================================================
# Helper Utilities
# ==============================================================================


def sanitize_filename(filename: str) -> str:
    """Sanitizes an untrusted filename, stripping directory traversal and null bytes.

    Args:
        filename: Untrusted raw filename from an uploaded file.

    Returns:
        Safe base filename string.
    """
    if not isinstance(filename, str):
        return "unnamed_document"
    clean_name = Path(filename).name.strip()
    return clean_name or "unnamed_document"


# ==============================================================================
# TXT Extraction
# ==============================================================================


def extract_txt(
    file_content: Union[bytes, str],
    filename: str = "document.txt",
    max_size_bytes: int = 10 * 1024 * 1024,
) -> DocumentResult:
    """Extracts, validates, and normalizes plain text from bytes or string content.

    Args:
        file_content: Raw bytes or string content of the uploaded text file.
        filename: Name of the uploaded file (must end in .txt).
        max_size_bytes: Maximum allowed file size in bytes (default 10 MB).

    Returns:
        DocumentResult with normalized text and a single PageContent (page 1).

    Raises:
        TypeError: If file_content is not bytes or str.
        UnsupportedFileTypeError: If filename does not end in .txt.
        FileSizeLimitExceededError: If byte size exceeds max_size_bytes.
        EmptyDocumentError: If file is empty or contains only whitespace.
        InvalidEncodingError: If bytes cannot be decoded as valid UTF-8.
    """
    clean_filename = sanitize_filename(filename)

    # Validate file extension
    if not clean_filename.lower().endswith(".txt"):
        raise UnsupportedFileTypeError(
            f"Unsupported file format for '{clean_filename}'. The text extractor only accepts .txt files."
        )

    # Validate input type
    if isinstance(file_content, str):
        raw_bytes = file_content.encode("utf-8")
    elif isinstance(file_content, bytes):
        raw_bytes = file_content
    else:
        raise TypeError(
            f"Expected bytes or str for file_content, got {type(file_content).__name__}"
        )

    # Check for empty file before size checks
    if len(raw_bytes) == 0:
        raise EmptyDocumentError(
            f"The file '{clean_filename}' is empty (0 bytes). Please upload a document with text."
        )

    # Check file size limit
    if len(raw_bytes) > max_size_bytes:
        size_mb = len(raw_bytes) / (1024 * 1024)
        limit_mb = max_size_bytes / (1024 * 1024)
        raise FileSizeLimitExceededError(
            f"File '{clean_filename}' ({size_mb:.2f} MB) exceeds the maximum allowed size limit of {limit_mb:.1f} MB."
        )

    # Decode UTF-8 (utf-8-sig transparently strips UTF-8 BOM if present)
    try:
        decoded_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidEncodingError(
            f"Failed to decode '{clean_filename}' as UTF-8. VerityAI requires UTF-8 encoded plain-text files. "
            f"Decoding error: {exc.reason} at byte position {exc.start}."
        ) from exc

    # Normalize line breaks while preserving paragraphs and intentional formatting
    normalized_text = decoded_text.replace("\r\n", "\n").replace("\r", "\n")

    # Reject whitespace-only text
    if not normalized_text.strip():
        raise EmptyDocumentError(
            f"The file '{clean_filename}' contains only whitespace or blank lines with no readable text."
        )

    total_chars = len(normalized_text)
    page = PageContent(page_number=1, text=normalized_text, char_count=total_chars)

    return DocumentResult(
        filename=clean_filename,
        doc_type="txt",
        total_pages=1,
        extracted_text=normalized_text,
        pages=[page],
        char_count=total_chars,
        status="success",
    )

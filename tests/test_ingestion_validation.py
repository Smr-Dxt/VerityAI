"""
Unit and edge-case validation tests for unified document ingestion
Phase 2: Local Document Ingestion
"""

import sys
from pathlib import Path
import pytest
import pymupdf

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from verityai.ingestion import (
    CorruptedFileError,
    DocumentResult,
    EmptyDocumentError,
    EncryptedPDFError,
    FileSizeLimitExceededError,
    InvalidEncodingError,
    UnsupportedFileTypeError,
    ingest_document,
    safe_ingest_document,
)
from verityai.config import get_config


def _create_sample_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Sample PDF validation content.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_ingest_document_txt_dispatch():
    """Verify that ingest_document correctly dispatches plain-text files."""
    result = ingest_document(b"Research claim A.", filename="study.txt")
    assert result.status == "success"
    assert result.doc_type == "txt"
    assert result.total_pages == 1
    assert "Research claim A." in result.extracted_text


def test_ingest_document_pdf_dispatch():
    """Verify that ingest_document correctly dispatches PDF files."""
    pdf_bytes = _create_sample_pdf()
    result = ingest_document(pdf_bytes, filename="study.pdf")
    assert result.status == "success"
    assert result.doc_type == "pdf"
    assert result.total_pages == 1
    assert "Sample PDF validation content." in result.extracted_text


def test_ingest_document_case_insensitive_extensions():
    """Verify that uppercase file extensions (.TXT, .PDF) are accepted."""
    txt_res = ingest_document(b"Upper case extension", filename="DATA.TXT")
    assert txt_res.status == "success"
    assert txt_res.doc_type == "txt"

    pdf_res = ingest_document(_create_sample_pdf(), filename="DOCUMENT.PDF")
    assert pdf_res.status == "success"
    assert pdf_res.doc_type == "pdf"


@pytest.mark.parametrize("bad_name", [
    "document.docx",
    "table.csv",
    "script.py",
    "image.png",
    "data.json",
    "archive.zip",
    "noextension",
])
def test_ingest_document_unsupported_extensions(bad_name):
    """Verify that all non-supported file formats raise UnsupportedFileTypeError."""
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        ingest_document(b"Some generic data", filename=bad_name)
    assert "Unsupported file type" in str(exc_info.value)


def test_ingest_document_null_byte_binary_detection_in_txt():
    """Verify that binary content disguised as a .txt file is caught via null byte detection."""
    binary_payload = b"GIF89a\x00\x01\x00\x01\x80\x00\x00\xff\xff\xff"
    with pytest.raises(InvalidEncodingError) as exc_info:
        ingest_document(binary_payload, filename="disguised_image.txt")
    assert "appears to be a binary file" in str(exc_info.value)


def test_ingest_document_pdf_signature_check():
    """Verify that a non-PDF file renamed to .pdf fails signature validation."""
    fake_payload = b"Plain ASCII text that does not start with the PDF signature."
    with pytest.raises(CorruptedFileError) as exc_info:
        ingest_document(fake_payload, filename="fake.pdf")
    assert "missing PDF signature" in str(exc_info.value)


def test_safe_ingest_document_success():
    """Verify safe_ingest_document returns a successful DocumentResult for valid documents."""
    result = safe_ingest_document(b"Safe test text.", filename="safe.txt")
    assert result.status == "success"
    assert result.error_message is None
    assert result.extracted_text == "Safe test text."


def test_safe_ingest_document_unsupported_format():
    """Verify safe_ingest_document captures UnsupportedFileTypeError into a typed error result."""
    result = safe_ingest_document(b"data", filename="paper.docx")
    assert result.status == "error"
    assert result.error_message is not None
    assert "Unsupported file type" in result.error_message
    assert result.total_pages == 0
    assert result.extracted_text == ""


def test_safe_ingest_document_empty_file():
    """Verify safe_ingest_document captures EmptyDocumentError without crashing."""
    result = safe_ingest_document(b"", filename="empty.txt")
    assert result.status == "error"
    assert "empty (0 bytes)" in result.error_message


def test_safe_ingest_document_encrypted_pdf():
    """Verify safe_ingest_document captures EncryptedPDFError without crashing."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Classified")
    encrypted_bytes = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="key")
    doc.close()

    result = safe_ingest_document(encrypted_bytes, filename="secure.pdf")
    assert result.status == "error"
    assert "password-protected" in result.error_message


def test_safe_ingest_document_scanned_pdf():
    """Verify safe_ingest_document captures ScannedPDFError without crashing."""
    doc = pymupdf.open()
    doc.new_page()  # Blank page
    blank_bytes = doc.tobytes()
    doc.close()

    result = safe_ingest_document(blank_bytes, filename="scan.pdf")
    assert result.status == "error"
    assert "no extractable text" in result.error_message
    assert "Local OCR will be supported in a future phase" in result.error_message


def test_safe_ingest_document_size_limit():
    """Verify safe_ingest_document captures FileSizeLimitExceededError."""
    large_payload = b"X" * 1024
    result = safe_ingest_document(large_payload, filename="big.txt", max_size_bytes=200)
    assert result.status == "error"
    assert "exceeds the maximum allowed size limit" in result.error_message

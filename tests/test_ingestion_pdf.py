"""
Unit tests for PDF (.pdf) Document Ingestion
Phase 2: Local Document Ingestion

These tests use synthetic in-memory PDFs generated dynamically via PyMuPDF.
No proprietary, private, or large static documents are committed to version control.
"""

import io
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
    PageContent,
    ScannedPDFError,
    UnsupportedFileTypeError,
    extract_pdf,
)


def _create_synthetic_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate a clean synthetic PDF in memory with specified pages."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _create_encrypted_pdf(text: str, password: str = "pass123") -> bytes:
    """Helper to generate an encrypted PDF in memory."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password)
    doc.close()
    return pdf_bytes


def test_extract_pdf_single_page():
    """Verify that a single-page PDF is extracted with correct text and page metadata."""
    sample_text = "VerityAI: Offline-First Agentic Research Assistant."
    pdf_bytes = _create_synthetic_pdf([sample_text])

    result = extract_pdf(pdf_bytes, filename="single_page.pdf")

    assert isinstance(result, DocumentResult)
    assert result.status == "success"
    assert result.filename == "single_page.pdf"
    assert result.doc_type == "pdf"
    assert result.total_pages == 1
    assert len(result.pages) == 1

    page1 = result.pages[0]
    assert isinstance(page1, PageContent)
    assert page1.page_number == 1
    assert sample_text in page1.text
    assert page1.char_count == len(page1.text)
    assert result.char_count == page1.char_count


def test_extract_pdf_multi_page():
    """Verify that multi-page PDFs extract page-by-page text with 1-indexed numbering."""
    pages_input = [
        "Page 1: Introduction and problem formulation.",
        "Page 2: Methodology and algorithmic pipeline.",
        "Page 3: Experimental empirical benchmarks and findings.",
    ]
    pdf_bytes = _create_synthetic_pdf(pages_input)

    result = extract_pdf(pdf_bytes, filename="multi_page_paper.pdf")

    assert result.status == "success"
    assert result.total_pages == 3
    assert len(result.pages) == 3

    # Check 1-based page numbering and page-isolated text
    for i, expected_snippet in enumerate(pages_input, start=1):
        page_item = result.pages[i - 1]
        assert page_item.page_number == i
        assert expected_snippet in page_item.text

    assert "Introduction" in result.pages[0].text
    assert "Methodology" in result.pages[1].text
    assert "Experimental" in result.pages[2].text
    assert result.char_count == sum(p.char_count for p in result.pages)


def test_extract_pdf_file_like_stream():
    """Verify that file-like streams (e.g. io.BytesIO from Streamlit uploads) work."""
    pdf_bytes = _create_synthetic_pdf(["Stream test content."])
    stream = io.BytesIO(pdf_bytes)

    result = extract_pdf(stream, filename="streamed.pdf")
    assert result.status == "success"
    assert "Stream test content." in result.extracted_text


def test_extract_pdf_empty_bytes():
    """Verify that a 0-byte file raises an EmptyDocumentError."""
    with pytest.raises(EmptyDocumentError) as exc_info:
        extract_pdf(b"", filename="empty.pdf")
    assert "empty (0 bytes)" in str(exc_info.value)


def test_extract_pdf_no_extractable_text_scanned_pdf():
    """Verify that PDFs with blank pages or no text raise ScannedPDFError with an OCR notice."""
    doc = pymupdf.open()
    doc.new_page()  # Blank page with zero text
    blank_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(ScannedPDFError) as exc_info:
        extract_pdf(blank_bytes, filename="scanned_scan.pdf")
    assert "no extractable text" in str(exc_info.value)
    assert "Local OCR will be supported in a future phase" in str(exc_info.value)


def test_extract_pdf_encrypted():
    """Verify that password-protected PDFs raise EncryptedPDFError."""
    encrypted_bytes = _create_encrypted_pdf("Secret text", password="mypassword")

    with pytest.raises(EncryptedPDFError) as exc_info:
        extract_pdf(encrypted_bytes, filename="protected.pdf")
    assert "password-protected" in str(exc_info.value)


def test_extract_pdf_corrupted_bytes():
    """Verify that corrupt bytes raise CorruptedFileError without crashing."""
    corrupted_bytes = b"%PDF-1.4\n%MalformedGarbageDataNotValid"

    with pytest.raises(CorruptedFileError) as exc_info:
        extract_pdf(corrupted_bytes, filename="corrupted.pdf")
    assert "corrupted, truncated, or not a valid PDF" in str(exc_info.value)


def test_extract_pdf_fake_extension_renamed_txt():
    """Verify that a plain text file renamed to .pdf is detected as corrupt."""
    fake_pdf_bytes = b"This is just ASCII text and not a PDF file at all."

    with pytest.raises(CorruptedFileError) as exc_info:
        extract_pdf(fake_pdf_bytes, filename="fake.pdf")
    assert "corrupted, truncated, or not a valid PDF" in str(exc_info.value)


def test_extract_pdf_unsupported_extension():
    """Verify that passing a non-.pdf filename raises UnsupportedFileTypeError."""
    pdf_bytes = _create_synthetic_pdf(["Sample"])

    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        extract_pdf(pdf_bytes, filename="sample.docx")
    assert "The PDF extractor only accepts .pdf files" in str(exc_info.value)


def test_extract_pdf_page_count_limit():
    """Verify that exceeding the max_pages limit raises FileSizeLimitExceededError."""
    pdf_bytes = _create_synthetic_pdf(["P1", "P2", "P3", "P4", "P5"])

    with pytest.raises(FileSizeLimitExceededError) as exc_info:
        extract_pdf(pdf_bytes, filename="too_long.pdf", max_pages=3)
    assert "exceeds the maximum allowed limit of 3 pages" in str(exc_info.value)


def test_extract_pdf_file_size_limit():
    """Verify that exceeding max_size_bytes raises FileSizeLimitExceededError."""
    pdf_bytes = _create_synthetic_pdf(["Small test content"])

    with pytest.raises(FileSizeLimitExceededError) as exc_info:
        extract_pdf(pdf_bytes, filename="oversized.pdf", max_size_bytes=100)
    assert "exceeds the maximum allowed size limit" in str(exc_info.value)


def test_extract_pdf_incorrect_input_type():
    """Verify that invalid input types raise a TypeError."""
    with pytest.raises(TypeError) as exc_info:
        extract_pdf(99999, filename="bad_type.pdf")  # type: ignore
    assert "Expected bytes or file-like object" in str(exc_info.value)

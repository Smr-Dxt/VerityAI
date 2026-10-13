"""
Unit tests for Plain-Text (.txt) Document Ingestion
Phase 2: Local Document Ingestion
"""

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from verityai.ingestion import (
    DocumentResult,
    EmptyDocumentError,
    FileSizeLimitExceededError,
    InvalidEncodingError,
    PageContent,
    UnsupportedFileTypeError,
    extract_txt,
    sanitize_filename,
)
from verityai.config import get_config


def test_extract_txt_valid_utf8():
    """Verify that a standard UTF-8 text file is ingested properly."""
    content = b"This is a valid UTF-8 research note.\nIt has factual observations."
    result = extract_txt(content, filename="research_notes.txt")

    assert isinstance(result, DocumentResult)
    assert result.status == "success"
    assert result.filename == "research_notes.txt"
    assert result.doc_type == "txt"
    assert result.total_pages == 1
    assert len(result.pages) == 1

    page = result.pages[0]
    assert isinstance(page, PageContent)
    assert page.page_number == 1
    assert "This is a valid UTF-8 research note." in page.text
    assert result.char_count == len(page.text)
    assert result.extracted_text == page.text


def test_extract_txt_unicode():
    """Verify that multi-language Unicode characters and symbols decode accurately."""
    unicode_str = (
        "Multilingual Research:\n"
        "French: Déjà vu et modèle vérifié.\n"
        "German: Über die Quantenmechanik und Maßstäbe.\n"
        "Greek: Επιστημονική μέθοδος.\n"
        "Japanese: オフライン推論システム.\n"
        "Symbols: ∑(x_i) ≥ 0, 🔬 ⚖️ ⚡\n"
    )
    content = unicode_str.encode("utf-8")
    result = extract_txt(content, filename="multilingual.txt")

    assert result.status == "success"
    assert "Déjà vu" in result.extracted_text
    assert "Über die Quantenmechanik" in result.extracted_text
    assert "Επιστημονική" in result.extracted_text
    assert "オフライン" in result.extracted_text
    assert "🔬 ⚖️ ⚡" in result.extracted_text


def test_extract_txt_utf8_bom():
    """Verify that a UTF-8 Byte Order Mark (BOM) is transparently stripped."""
    # UTF-8 BOM is 0xEF, 0xBB, 0xBF
    bom_bytes = b"\xef\xbb\xbfTitle: Document with BOM header\nContent follows."
    result = extract_txt(bom_bytes, filename="bom_doc.txt")

    assert result.status == "success"
    assert not result.extracted_text.startswith("\ufeff")
    assert result.extracted_text.startswith("Title: Document with BOM header")


def test_extract_txt_multiline_preserved():
    """Verify that line breaks, carriage returns, and paragraphs are preserved."""
    raw_content = b"Paragraph 1\r\n\r\nParagraph 2\nLine 2b\rLine 2c\n\nParagraph 3"
    result = extract_txt(raw_content, filename="lines.txt")

    assert result.status == "success"
    assert "\r" not in result.extracted_text  # Carriage returns normalized
    assert "Paragraph 1\n\nParagraph 2\nLine 2b\nLine 2c\n\nParagraph 3" == result.extracted_text


def test_extract_txt_empty_file():
    """Verify that a 0-byte file raises an EmptyDocumentError."""
    with pytest.raises(EmptyDocumentError) as exc_info:
        extract_txt(b"", filename="empty.txt")
    assert "empty (0 bytes)" in str(exc_info.value)


def test_extract_txt_whitespace_only():
    """Verify that a document containing only spaces, tabs, and newlines is rejected."""
    whitespace_bytes = b"   \n\t  \r\n   \t  \n"
    with pytest.raises(EmptyDocumentError) as exc_info:
        extract_txt(whitespace_bytes, filename="blank.txt")
    assert "whitespace or blank lines" in str(exc_info.value)


def test_extract_txt_invalid_encoding():
    """Verify that invalid UTF-8 bytes raise an InvalidEncodingError with details."""
    invalid_bytes = b"Valid prefix\x80\x81\xff\xfeInvalid bytes"
    with pytest.raises(InvalidEncodingError) as exc_info:
        extract_txt(invalid_bytes, filename="bad_encoding.txt")
    assert "Failed to decode" in str(exc_info.value)
    assert "UTF-8" in str(exc_info.value)


def test_extract_txt_unsupported_extension():
    """Verify that non-.txt filenames are rejected by the text parser."""
    content = b"Some plain text"
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        extract_txt(content, filename="document.pdf")
    assert "The text extractor only accepts .txt files" in str(exc_info.value)


def test_extract_txt_file_size_limit():
    """Verify that files exceeding the size limit raise FileSizeLimitExceededError."""
    payload = b"A" * 1024  # 1 KB
    limit = 500  # 500 bytes limit
    with pytest.raises(FileSizeLimitExceededError) as exc_info:
        extract_txt(payload, filename="large.txt", max_size_bytes=limit)
    assert "exceeds the maximum allowed size limit" in str(exc_info.value)


def test_extract_txt_incorrect_input_type():
    """Verify that passing an invalid type raises a TypeError."""
    with pytest.raises(TypeError) as exc_info:
        extract_txt(12345, filename="bad_type.txt")  # type: ignore
    assert "Expected bytes or str" in str(exc_info.value)


def test_extract_txt_string_input():
    """Verify that string inputs can be passed directly to extract_txt."""
    text_input = "Direct string content for ingestion."
    result = extract_txt(text_input, filename="direct_str.txt")
    assert result.status == "success"
    assert result.extracted_text == text_input


def test_extract_txt_filename_sanitization():
    """Verify that untrusted filenames with path traversal characters are sanitized."""
    content = b"Safe research content."
    result = extract_txt(content, filename="../../etc/shadow.txt")
    assert result.filename == "shadow.txt"

    result_windows = extract_txt(content, filename=r"..\..\Windows\System32\cmd.txt")
    assert result_windows.filename == "cmd.txt"


def test_sanitize_filename_utility():
    """Verify standalone behavior of sanitize_filename."""
    assert sanitize_filename("safe_file.txt") == "safe_file.txt"
    assert sanitize_filename("folder/nested/doc.txt") == "doc.txt"
    assert sanitize_filename("") == "unnamed_document"
    assert sanitize_filename(None) == "unnamed_document"  # type: ignore


def test_config_integration_max_file_size():
    """Verify AppConfig max_file_size_bytes integrates smoothly."""
    config = get_config()
    assert config.max_file_size_bytes == 10 * 1024 * 1024  # 10 MB default
    assert config.max_pdf_pages == 500  # 500 page limit default

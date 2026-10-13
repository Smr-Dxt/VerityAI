"""
Streamlit UI and Integration Tests for Document Ingestion
Phase 2: Local Document Ingestion

These tests verify that the Streamlit application starts without exceptions,
navigates to the Documents view, and correctly displays document metadata,
previews, page selectors, and actionable error messages.
"""

import sys
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

from verityai.ingestion import DocumentResult, PageContent


def test_streamlit_app_launches_cleanly():
    """Verify that app.py compiles and runs without any uncaught exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=15.0)
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Research Workspace"


def test_documents_view_renders_uploader():
    """Verify that switching to the Documents view renders the file uploader."""
    at = AppTest.from_file(APP_PATH, default_timeout=15.0)
    at.run()
    at.sidebar.radio[0].set_value("Documents").run()

    assert not at.exception
    assert len(at.file_uploader) == 1
    uploader = at.file_uploader[0]
    assert uploader.key == "doc_uploader"
    assert "local ingestion" in uploader.label


def test_documents_view_displays_txt_metadata_and_preview():
    """Verify that a successfully ingested TXT document displays summary metadata and text preview."""
    at = AppTest.from_file(APP_PATH, default_timeout=15.0)
    at.run()

    # Seed session state with a successfully extracted TXT document
    mock_txt_result = DocumentResult(
        filename="test_paper.txt",
        doc_type="txt",
        total_pages=1,
        extracted_text="Empirical findings on attention mechanisms.",
        pages=[PageContent(page_number=1, text="Empirical findings on attention mechanisms.", char_count=43)],
        char_count=43,
        status="success",
    )
    at.session_state["active_document"] = mock_txt_result
    at.sidebar.radio[0].set_value("Documents").run()

    assert not at.exception
    assert len(at.success) >= 1
    assert "test_paper.txt" in at.success[0].value
    # Text area preview should contain the extracted text
    assert len(at.text_area) >= 1
    assert "Empirical findings on attention mechanisms." in at.text_area[0].value


def test_documents_view_displays_pdf_page_selector():
    """Verify that a multi-page PDF renders the page-by-page selector and page content."""
    at = AppTest.from_file(APP_PATH, default_timeout=15.0)
    at.run()

    # Seed session state with a 2-page PDF
    mock_pdf_result = DocumentResult(
        filename="multi_page_study.pdf",
        doc_type="pdf",
        total_pages=2,
        extracted_text="Page 1 text.\n\nPage 2 text.",
        pages=[
            PageContent(page_number=1, text="Page 1 text.", char_count=12),
            PageContent(page_number=2, text="Page 2 text.", char_count=12),
        ],
        char_count=24,
        status="success",
    )
    at.session_state["active_document"] = mock_pdf_result
    at.sidebar.radio[0].set_value("Documents").run()

    assert not at.exception
    # Should render a selectbox for page selection
    assert len(at.selectbox) >= 1
    assert len(at.text_area) >= 1
    assert "Page 1 text." in at.text_area[0].value


def test_documents_view_displays_actionable_error_alert():
    """Verify that an ingestion failure displays an error alert without crashing."""
    at = AppTest.from_file(APP_PATH, default_timeout=15.0)
    at.run()

    # Seed session state with a failed document result
    mock_error_result = DocumentResult(
        filename="scanned_paper.pdf",
        doc_type="pdf",
        total_pages=0,
        extracted_text="",
        pages=[],
        char_count=0,
        status="error",
        error_message="The PDF 'scanned_paper.pdf' contains no extractable text. It may be a scanned document.",
    )
    at.session_state["active_document"] = mock_error_result
    at.sidebar.radio[0].set_value("Documents").run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "no extractable text" in at.error[0].value


def test_research_workspace_reflects_active_document():
    """Verify that Research Workspace displays the active document corpus status when loaded."""
    at = AppTest.from_file(APP_PATH, default_timeout=15.0)
    at.run()

    mock_doc = DocumentResult(
        filename="notes.txt",
        doc_type="txt",
        total_pages=1,
        extracted_text="Content",
        pages=[PageContent(page_number=1, text="Content", char_count=7)],
        char_count=7,
        status="success",
    )
    at.session_state["active_document"] = mock_doc
    at.sidebar.radio[0].set_value("Research Workspace").run()

    assert not at.exception
    # Verify markdown cards display the filename
    markdown_content = " ".join(m.value for m in at.markdown)
    assert "notes.txt" in markdown_content
    assert "Active Ingested Document" in markdown_content

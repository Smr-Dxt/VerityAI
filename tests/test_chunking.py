"""
Unit tests for the source-aware document chunking module.
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval
"""

import pytest

from verityai.chunking import (
    ChunkMetadata,
    DocumentChunk,
    chunk_document,
    generate_doc_id,
    split_text_into_chunks,
)
from verityai.ingestion import DocumentResult, PageContent


# ==============================================================================
# Text Splitting Unit Tests
# ==============================================================================


def test_split_text_empty_and_whitespace():
    """Empty or whitespace-only strings must return an empty list."""
    assert split_text_into_chunks("") == []
    assert split_text_into_chunks("   \n\t  ") == []


def test_split_text_short_document():
    """Text shorter than chunk_size should produce exactly one chunk."""
    text = "VerityAI is an offline-first research assistant."
    chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_word_boundary_preservation():
    """Splitting should snap to whitespace rather than cutting words in half."""
    text = "The quick brown fox jumps over the lazy dog repeatedly until evening."
    # chunk_size=30 would slice 'repeatedly' if not word-aware
    chunks = split_text_into_chunks(text, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 1

    for chunk in chunks:
        # No leading or trailing whitespace
        assert chunk == chunk.strip()
        # Words must not be arbitrarily broken with orphaned letters
        words = chunk.split()
        for w in words:
            assert w in text


def test_split_text_overlap():
    """Consecutive chunks must share overlapping text."""
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
        "omicron pi rho sigma tau upsilon phi chi psi omega."
    )
    chunks = split_text_into_chunks(text, chunk_size=60, chunk_overlap=20)
    assert len(chunks) >= 2

    # Verify that the end of chunk 0 appears near the start of chunk 1
    words_chunk_0 = set(chunks[0].split())
    words_chunk_1 = set(chunks[1].split())
    shared_words = words_chunk_0.intersection(words_chunk_1)
    assert len(shared_words) > 0, "Adjacent overlapping chunks must share common words"


def test_split_text_invalid_parameters():
    """Invalid chunk_size or chunk_overlap values must raise ValueError."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_text_into_chunks("Some text", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_text_into_chunks("Some text", chunk_size=-10, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        split_text_into_chunks("Some text", chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
        split_text_into_chunks("Some text", chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
        split_text_into_chunks("Some text", chunk_size=100, chunk_overlap=150)


def test_split_text_no_whitespace_fallback():
    """Extremely long tokens without whitespace must split cleanly at chunk_size limit."""
    long_string = "A" * 250
    chunks = split_text_into_chunks(long_string, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 100


# ==============================================================================
# Document Chunking Tests (Provenance & Metadata)
# ==============================================================================


def test_chunk_document_pdf_multipage_provenance():
    """PDF chunks must strictly preserve 1-indexed page boundaries without cross-page mixing."""
    page1 = PageContent(
        page_number=1,
        text="Introduction to offline artificial intelligence and edge computing.",
        char_count=67,
    )
    page2 = PageContent(
        page_number=2,
        text="Methodology and mathematical foundations for vector search indexing.",
        char_count=69,
    )
    page3 = PageContent(
        page_number=3,
        text="Conclusion and performance benchmarks on RTX 3050 mobile hardware.",
        char_count=68,
    )

    doc = DocumentResult(
        filename="research_paper.pdf",
        doc_type="pdf",
        total_pages=3,
        extracted_text=f"{page1.text}\n{page2.text}\n{page3.text}",
        pages=[page1, page2, page3],
        char_count=len(page1.text) + len(page2.text) + len(page3.text),
        status="success",
    )

    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 3
    # Verify exact 1-indexed page preservation
    assert chunks[0].metadata.page_number == 1
    assert chunks[1].metadata.page_number == 2
    assert chunks[2].metadata.page_number == 3

    # Verify deterministic chunk IDs contain page numbers
    assert "_p1_c0" in chunks[0].metadata.chunk_id
    assert "_p2_c0" in chunks[1].metadata.chunk_id
    assert "_p3_c0" in chunks[2].metadata.chunk_id

    # Verify filename and doc_type
    for c in chunks:
        assert c.metadata.filename == "research_paper.pdf"
        assert c.metadata.doc_type == "pdf"
        assert c.metadata.char_count == len(c.text)
        assert c.metadata.word_count == len(c.text.split())


def test_chunk_document_txt_provenance():
    """TXT document chunks must carry doc_type='txt' and page_number=1."""
    text_content = (
        "VerityAI Plain Text Research Notes.\n"
        "This note documents the retrieval mechanisms planned for Phase 3.\n"
        "We are integrating sentence transformers and FAISS indices for offline RAG."
    )
    page = PageContent(page_number=1, text=text_content, char_count=len(text_content))
    doc = DocumentResult(
        filename="notes.txt",
        doc_type="txt",
        total_pages=1,
        extracted_text=text_content,
        pages=[page],
        char_count=len(text_content),
        status="success",
    )

    chunks = chunk_document(doc, chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2

    for idx, chunk in enumerate(chunks):
        assert chunk.metadata.filename == "notes.txt"
        assert chunk.metadata.doc_type == "txt"
        assert chunk.metadata.page_number == 1
        assert chunk.metadata.chunk_index == idx
        assert chunk.metadata.chunk_id.endswith(f"_c{idx}")
        assert chunk.metadata.char_count == len(chunk.text)


def test_deterministic_chunk_ids():
    """Identical documents must always yield identical doc_id and chunk_ids."""
    page = PageContent(page_number=1, text="Deterministic content test.", char_count=27)
    doc_a = DocumentResult(
        filename="sample.txt",
        doc_type="txt",
        total_pages=1,
        extracted_text="Deterministic content test.",
        pages=[page],
        char_count=27,
        status="success",
    )
    doc_b = DocumentResult(
        filename="sample.txt",
        doc_type="txt",
        total_pages=1,
        extracted_text="Deterministic content test.",
        pages=[page],
        char_count=27,
        status="success",
    )

    chunks_a = chunk_document(doc_a)
    chunks_b = chunk_document(doc_b)

    assert len(chunks_a) == len(chunks_b) == 1
    assert chunks_a[0].metadata.doc_id == chunks_b[0].metadata.doc_id
    assert chunks_a[0].metadata.chunk_id == chunks_b[0].metadata.chunk_id


def test_chunk_document_empty_or_error_status():
    """Documents with error status or empty pages must safely return empty lists."""
    error_doc = DocumentResult(
        filename="bad.pdf",
        doc_type="pdf",
        total_pages=0,
        extracted_text="",
        pages=[],
        char_count=0,
        status="error",
        error_message="Corrupted file",
    )
    assert chunk_document(error_doc) == []

    empty_doc = DocumentResult(
        filename="empty.txt",
        doc_type="txt",
        total_pages=1,
        extracted_text="   \n  ",
        pages=[PageContent(page_number=1, text="   ", char_count=3)],
        char_count=3,
        status="success",
    )
    assert chunk_document(empty_doc) == []

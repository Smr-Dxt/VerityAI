"""
Unit tests for FAISS-based semantic vector retrieval.
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval
"""

import pytest

from verityai.chunking import ChunkMetadata, DocumentChunk
from verityai.embeddings import MockEmbedder
from verityai.retrieval_semantic import FAISSRetriever, RetrievalResult


# ==============================================================================
# Test Fixtures & Helpers
# ==============================================================================


def create_sample_chunk(
    text: str,
    doc_id: str = "doc_test123",
    filename: str = "research_paper.pdf",
    page_number: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    """Helper to build test DocumentChunk instances with valid metadata."""
    metadata = ChunkMetadata(
        chunk_id=f"{doc_id}_p{page_number}_c{chunk_index}",
        doc_id=doc_id,
        filename=filename,
        doc_type="pdf" if filename.endswith(".pdf") else "txt",
        page_number=page_number,
        chunk_index=chunk_index,
        char_count=len(text),
        word_count=len(text.split()),
    )
    return DocumentChunk(text=text, metadata=metadata)


# ==============================================================================
# FAISSRetriever Unit Tests
# ==============================================================================


def test_empty_retriever_returns_empty_list():
    """Querying an empty or unindexed retriever must safely return an empty list."""
    embedder = MockEmbedder(dimension=64)
    retriever = FAISSRetriever(embedder=embedder)

    assert retriever.num_indexed == 0
    results = retriever.retrieve("Any query text", top_k=5)
    assert results == []


def test_invalid_query_validation():
    """Empty or whitespace-only queries must raise ValueError."""
    retriever = FAISSRetriever(embedder=MockEmbedder(dimension=64))

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        retriever.retrieve("    \t\n  ")


def test_invalid_top_k_validation():
    """Non-positive top_k arguments must raise ValueError."""
    retriever = FAISSRetriever(embedder=MockEmbedder(dimension=64))

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("Valid query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("Valid query", top_k=-3)


def test_index_and_retrieve_ranking():
    """Retrieved results must be ranked in descending order of similarity score."""
    embedder = MockEmbedder(dimension=64)
    retriever = FAISSRetriever(embedder=embedder)

    c1 = create_sample_chunk("Offline artificial intelligence architectures.", page_number=1, chunk_index=0)
    c2 = create_sample_chunk("Neural network quantization techniques on GPU.", page_number=2, chunk_index=1)
    c3 = create_sample_chunk("Vector database indexing using FAISS Inner Product.", page_number=3, chunk_index=2)

    retriever.index_chunks([c1, c2, c3])
    assert retriever.num_indexed == 3

    # Query for exact match with chunk 2
    results = retriever.retrieve("Neural network quantization techniques on GPU.", top_k=3)

    assert len(results) == 3
    assert all(isinstance(r, RetrievalResult) for r in results)

    # Top match must be c2 with rank 1
    assert results[0].rank == 1
    assert results[0].chunk.metadata.chunk_id == c2.metadata.chunk_id
    assert results[0].chunk.metadata.page_number == 2
    assert results[0].score >= results[1].score >= results[2].score


def test_provenance_preservation():
    """Every retrieved result must preserve exact source metadata (filename, page, ID)."""
    embedder = MockEmbedder(dimension=64)
    retriever = FAISSRetriever(embedder=embedder)

    chunk = create_sample_chunk(
        text="Citation grounding ensures factual verification of retrieved passages.",
        filename="benchmark_report.pdf",
        page_number=42,
        chunk_index=7,
    )

    retriever.index_chunks([chunk])
    results = retriever.retrieve("factual verification", top_k=1)

    assert len(results) == 1
    match = results[0]

    assert match.chunk.metadata.filename == "benchmark_report.pdf"
    assert match.chunk.metadata.page_number == 42
    assert match.chunk.metadata.chunk_index == 7
    assert match.chunk.metadata.doc_type == "pdf"
    assert match.chunk.text == chunk.text


def test_top_k_larger_than_corpus():
    """When top_k exceeds total indexed chunks, return all available chunks without crashing."""
    embedder = MockEmbedder(dimension=64)
    retriever = FAISSRetriever(embedder=embedder)

    chunks = [
        create_sample_chunk("First chunk", chunk_index=0),
        create_sample_chunk("Second chunk", chunk_index=1),
    ]
    retriever.index_chunks(chunks)

    results = retriever.retrieve("chunk", top_k=100)
    assert len(results) == 2


def test_clear_retriever():
    """Clearing the retriever resets indexed state and yields empty queries."""
    embedder = MockEmbedder(dimension=64)
    retriever = FAISSRetriever(embedder=embedder)

    chunk = create_sample_chunk("Some technical text")
    retriever.index_chunks([chunk])
    assert retriever.num_indexed == 1

    retriever.clear()
    assert retriever.num_indexed == 0
    assert retriever.retrieve("Some technical text") == []


def test_score_range_and_normalization():
    """Similarity scores for normalized vectors must lie within [-1.0, 1.0]."""
    embedder = MockEmbedder(dimension=128, normalize=True)
    retriever = FAISSRetriever(embedder=embedder)

    chunks = [
        create_sample_chunk("Passage A", chunk_index=0),
        create_sample_chunk("Passage B", chunk_index=1),
        create_sample_chunk("Passage C", chunk_index=2),
    ]
    retriever.index_chunks(chunks)

    results = retriever.retrieve("Passage query", top_k=3)
    for r in results:
        assert -1.0 - 1e-4 <= r.score <= 1.0 + 1e-4

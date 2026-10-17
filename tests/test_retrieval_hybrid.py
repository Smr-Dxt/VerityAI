"""
Unit tests for BM25 lexical retrieval and hybrid search (BM25 + FAISS + RRF).
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval
"""

import pytest

from verityai.chunking import ChunkMetadata, DocumentChunk
from verityai.embeddings import MockEmbedder
from verityai.retrieval_hybrid import BM25Retriever, HybridRetrievalResult, HybridRetriever
from verityai.retrieval_semantic import FAISSRetriever


# ==============================================================================
# Helpers
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
# BM25Retriever Unit Tests
# ==============================================================================


def test_bm25_empty_retriever():
    """Empty BM25 retriever must return empty results."""
    bm25 = BM25Retriever()
    assert bm25.num_indexed == 0
    assert bm25.retrieve("query") == []


def test_bm25_exact_keyword_matching():
    """BM25 must accurately identify and rank exact unique lexical tokens."""
    c1 = create_sample_chunk("Standard transformer attention mechanism.", chunk_index=0)
    c2 = create_sample_chunk("NVIDIA RTX-3050-Mobile hardware specifications.", chunk_index=1)
    c3 = create_sample_chunk("Quantized weight matrix GGUF-Q4_K_M representation.", chunk_index=2)

    bm25 = BM25Retriever()
    bm25.index_chunks([c1, c2, c3])
    assert bm25.num_indexed == 3

    results = bm25.retrieve("GGUF-Q4_K_M", top_k=2)
    assert len(results) == 1
    assert results[0].chunk.metadata.chunk_id == c3.metadata.chunk_id
    assert results[0].score > 0.0
    assert results[0].rank == 1


def test_bm25_no_matches_returns_empty():
    """When no terms match in the corpus, BM25 returns an empty list."""
    c1 = create_sample_chunk("Machine learning and artificial intelligence.", chunk_index=0)
    bm25 = BM25Retriever()
    bm25.index_chunks([c1])

    results = bm25.retrieve("xylophone submarine", top_k=5)
    assert results == []


def test_bm25_validation():
    """BM25 must validate inputs."""
    bm25 = BM25Retriever()
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        bm25.retrieve("   ")

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        bm25.retrieve("valid", top_k=0)


# ==============================================================================
# HybridRetriever (Reciprocal Rank Fusion) Unit Tests
# ==============================================================================


def test_hybrid_empty_retriever():
    """Empty hybrid retriever returns empty results."""
    embedder = MockEmbedder(dimension=64)
    hybrid = HybridRetriever(semantic_retriever=FAISSRetriever(embedder=embedder))
    assert hybrid.num_indexed == 0
    assert hybrid.retrieve("query") == []


def test_hybrid_deduplication_and_provenance():
    """Hybrid results must deduplicate chunks by chunk_id and retain page metadata."""
    embedder = MockEmbedder(dimension=64)
    hybrid = HybridRetriever(semantic_retriever=FAISSRetriever(embedder=embedder))

    c1 = create_sample_chunk(
        "Low rank adaptation LoRA fine-tuning methodology.",
        filename="methods.pdf",
        page_number=14,
        chunk_index=0,
    )
    c2 = create_sample_chunk(
        "Evaluation benchmarks on technical document retrieval.",
        filename="methods.pdf",
        page_number=15,
        chunk_index=1,
    )

    hybrid.index_chunks([c1, c2])

    results = hybrid.retrieve("LoRA fine-tuning", top_k=5)
    assert len(results) <= 2

    # Check deduplication: all chunk_ids must be unique
    chunk_ids = [r.chunk.metadata.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))

    # Provenance check
    top_match = results[0]
    assert isinstance(top_match, HybridRetrievalResult)
    assert top_match.chunk.metadata.filename == "methods.pdf"
    assert top_match.chunk.metadata.page_number in (14, 15)
    assert top_match.rrf_score > 0.0


def test_hybrid_consensus_boosting():
    """A chunk ranked highly by BOTH semantic and BM25 search must receive a boosted RRF score."""
    embedder = MockEmbedder(dimension=64)
    hybrid = HybridRetriever(semantic_retriever=FAISSRetriever(embedder=embedder), rrf_k=60)

    # c1 is exact lexical match and semantic match
    c1 = create_sample_chunk("Evaluation of quantized 4-bit transformer attention.", chunk_index=0)
    c2 = create_sample_chunk("General introduction to machine learning principles.", chunk_index=1)
    c3 = create_sample_chunk("Random text without relevance.", chunk_index=2)

    hybrid.index_chunks([c1, c2, c3])

    results = hybrid.retrieve("quantized 4-bit transformer", top_k=3)
    assert len(results) >= 1

    top_result = results[0]
    # c1 should be top match
    assert top_result.chunk.metadata.chunk_id == c1.metadata.chunk_id
    # Ranked in both semantic and keyword search
    assert top_result.semantic_rank is not None
    assert top_result.keyword_rank is not None
    # RRF score should equal the sum of both contributions
    expected_score = (1.0 / (60 + top_result.semantic_rank)) + (1.0 / (60 + top_result.keyword_rank))
    assert pytest.approx(top_result.rrf_score, rel=1e-5) == expected_score


def test_hybrid_configurable_weights():
    """Setting keyword_weight=0 or semantic_weight=0 isolates the respective search modality."""
    embedder = MockEmbedder(dimension=64)
    hybrid = HybridRetriever(semantic_retriever=FAISSRetriever(embedder=embedder))

    c1 = create_sample_chunk("Hardware constraints on RTX 3050 mobile GPU.", chunk_index=0)
    hybrid.index_chunks([c1])

    # Pure semantic
    res_sem = hybrid.retrieve("RTX 3050", semantic_weight=1.0, keyword_weight=0.0)
    assert len(res_sem) == 1
    assert res_sem[0].semantic_rank is not None
    assert res_sem[0].keyword_rank is None

    # Pure keyword
    res_kw = hybrid.retrieve("RTX 3050", semantic_weight=0.0, keyword_weight=1.0)
    assert len(res_kw) == 1
    assert res_kw[0].semantic_rank is None
    assert res_kw[0].keyword_rank is not None


def test_hybrid_validation():
    """HybridRetriever must validate query, top_k, weights, and rrf_k."""
    embedder = MockEmbedder(dimension=64)
    hybrid = HybridRetriever(semantic_retriever=FAISSRetriever(embedder=embedder))

    with pytest.raises(ValueError, match="cannot be empty"):
        hybrid.retrieve("")

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        hybrid.retrieve("query", top_k=-1)

    with pytest.raises(ValueError, match="Retrieval weights must be non-negative"):
        hybrid.retrieve("query", semantic_weight=-0.5)

    with pytest.raises(ValueError, match="rrf_k must be positive"):
        HybridRetriever(semantic_retriever=FAISSRetriever(embedder=embedder), rrf_k=0)

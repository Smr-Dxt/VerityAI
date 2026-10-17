"""
Reproducible retrieval evaluation suite for Semantic, Keyword, and Hybrid Search.
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval

This suite evaluates retrieval quality using standard information retrieval metrics:
- Recall@k: Proportion of ground-truth relevant passages found in top-k results.
- Precision@k: Proportion of retrieved passages in top-k that are relevant.
- Mean Reciprocal Rank (MRR): Position of the first relevant passage (1 / rank).
- Retrieval Latency: Execution time per query in milliseconds.
- Provenance Integrity: Percentage of results retaining valid filename, doc_id, and 1-indexed page.

GROUND TRUTH ANNOTATION METHODOLOGY:
A corpus of 6 technical passages covering distinct subjects was manually created.
5 evaluation queries were formulated to test:
1. Exact keyword matching (BM25 advantage)
2. Semantic paraphrasing (FAISS vector advantage)
3. Combined lexical/semantic queries (Hybrid RRF advantage)
4. Domain acronyms and library references
5. Offline architecture principles
"""

import time
from typing import Dict, List, Set
import pytest

from verityai.chunking import ChunkMetadata, DocumentChunk
from verityai.embeddings import MockEmbedder
from verityai.retrieval_hybrid import BM25Retriever, HybridRetriever
from verityai.retrieval_semantic import FAISSRetriever


# ==============================================================================
# Metric Computation Utilities
# ==============================================================================


def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes Recall@k: |Retrieved@k ∩ Relevant| / |Relevant|."""
    if not ground_truth_ids:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    hits = top_k_retrieved.intersection(ground_truth_ids)
    return len(hits) / len(ground_truth_ids)


def compute_precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    """Computes Precision@k: |Retrieved@k ∩ Relevant| / k."""
    if k <= 0:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    hits = top_k_retrieved.intersection(ground_truth_ids)
    return len(hits) / k


def compute_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    """Computes Reciprocal Rank (RR): 1 / rank of first relevant match."""
    for rank_idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank_idx
    return 0.0


# ==============================================================================
# Benchmark Data Fixture
# ==============================================================================


@pytest.fixture
def benchmark_corpus() -> List[DocumentChunk]:
    """Provides 6 distinct technical research chunks with strict page provenance."""
    chunks_data = [
        (
            "doc1_p1_c0",
            "NVIDIA RTX 3050 mobile GPU architecture features 4GB VRAM and Ampere tensor cores for edge inference.",
            "hardware_spec.pdf",
            1,
        ),
        (
            "doc2_p2_c0",
            "Quantization reduces neural model weights from 16-bit floating point to 4-bit integers using GGUF format.",
            "quantization_methods.pdf",
            2,
        ),
        (
            "doc3_p3_c0",
            "Reciprocal Rank Fusion merges rankings from disparate retrieval algorithms using positional scoring without raw score addition.",
            "fusion_algorithms.pdf",
            3,
        ),
        (
            "doc4_p4_c0",
            "FAISS is an open-source library developed by Meta for high-dimensional vector similarity search in memory.",
            "vector_search.pdf",
            4,
        ),
        (
            "doc5_p5_c0",
            "Offline research assistants operate locally on consumer hardware without transmitting private data to external cloud APIs.",
            "privacy_architecture.pdf",
            5,
        ),
        (
            "doc6_p6_c0",
            "BM25 is a probabilistic lexical ranking function used in information retrieval to score document keyword relevance.",
            "lexical_ranking.pdf",
            6,
        ),
    ]

    corpus = []
    for idx, (cid, text, fname, page_num) in enumerate(chunks_data):
        meta = ChunkMetadata(
            chunk_id=cid,
            doc_id=f"doc_{idx}",
            filename=fname,
            doc_type="pdf",
            page_number=page_num,
            chunk_index=0,
            char_count=len(text),
            word_count=len(text.split()),
        )
        corpus.append(DocumentChunk(text=text, metadata=meta))

    return corpus


@pytest.fixture
def benchmark_queries() -> List[Dict]:
    """Provides evaluation queries paired with verified ground-truth chunk IDs."""
    return [
        {
            "query": "GGUF format 4-bit integers",
            "relevant_ids": {"doc2_p2_c0"},
            "type": "exact_keyword",
        },
        {
            "query": "laptop graphics card memory specifications",
            "relevant_ids": {"doc1_p1_c0"},
            "type": "semantic_concept",
        },
        {
            "query": "merging search rankings with reciprocal rank fusion",
            "relevant_ids": {"doc3_p3_c0"},
            "type": "hybrid_terms",
        },
        {
            "query": "Meta library for vector similarity search",
            "relevant_ids": {"doc4_p4_c0"},
            "type": "entity_lexical",
        },
        {
            "query": "operating locally without external cloud APIs",
            "relevant_ids": {"doc5_p5_c0"},
            "type": "semantic_concept",
        },
    ]


# ==============================================================================
# Evaluation Tests
# ==============================================================================


def test_metric_calculation_utilities():
    """Validates the mathematical correctness of metric computation functions."""
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
    ground_truth = {"doc_b"}

    # Recall@1 = 0 (doc_b not at rank 1), Recall@2 = 1.0
    assert compute_recall_at_k(retrieved, ground_truth, k=1) == 0.0
    assert compute_recall_at_k(retrieved, ground_truth, k=2) == 1.0

    # Precision@1 = 0.0, Precision@2 = 1/2 = 0.5
    assert compute_precision_at_k(retrieved, ground_truth, k=1) == 0.0
    assert compute_precision_at_k(retrieved, ground_truth, k=2) == 0.5

    # MRR: doc_b is at rank 2 -> 1 / 2 = 0.5
    assert compute_reciprocal_rank(retrieved, ground_truth) == 0.5

    # Empty inputs
    assert compute_recall_at_k([], ground_truth, k=5) == 0.0
    assert compute_precision_at_k([], ground_truth, k=5) == 0.0
    assert compute_reciprocal_rank([], ground_truth) == 0.0


def test_retrieval_benchmark_evaluation(benchmark_corpus, benchmark_queries):
    """Executes benchmark queries across Semantic, Keyword, and Hybrid retrievers.

    Measures Recall@k, Precision@k, MRR, and latency, confirming that:
    - Hybrid retrieval achieves high Recall@3 (>= 0.80) and strong MRR.
    - Retrieval latency remains well under 100 milliseconds per query on local CPU.
    """
    embedder = MockEmbedder(dimension=64)
    sem_retriever = FAISSRetriever(embedder=embedder)
    bm25_retriever = BM25Retriever()
    hybrid_retriever = HybridRetriever(
        semantic_retriever=sem_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
    )

    hybrid_retriever.index_chunks(benchmark_corpus)

    mrr_scores = []
    recall_at_1_scores = []
    recall_at_3_scores = []
    precision_at_1_scores = []
    latencies_ms = []

    for item in benchmark_queries:
        query = item["query"]
        ground_truth = item["relevant_ids"]

        t_start = time.perf_counter()
        results = hybrid_retriever.retrieve(query, top_k=3)
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        latencies_ms.append(t_elapsed_ms)

        retrieved_ids = [r.chunk.metadata.chunk_id for r in results]

        recall_at_1 = compute_recall_at_k(retrieved_ids, ground_truth, k=1)
        recall_at_3 = compute_recall_at_k(retrieved_ids, ground_truth, k=3)
        precision_at_1 = compute_precision_at_k(retrieved_ids, ground_truth, k=1)
        rr = compute_reciprocal_rank(retrieved_ids, ground_truth)

        recall_at_1_scores.append(recall_at_1)
        recall_at_3_scores.append(recall_at_3)
        precision_at_1_scores.append(precision_at_1)
        mrr_scores.append(rr)

    avg_recall_at_1 = sum(recall_at_1_scores) / len(recall_at_1_scores)
    avg_recall_at_3 = sum(recall_at_3_scores) / len(recall_at_3_scores)
    avg_precision_at_1 = sum(precision_at_1_scores) / len(precision_at_1_scores)
    mean_mrr = sum(mrr_scores) / len(mrr_scores)
    avg_latency = sum(latencies_ms) / len(latencies_ms)

    # Assert rigorous retrieval thresholds
    assert avg_recall_at_3 >= 0.80, f"Expected Recall@3 >= 0.80, got {avg_recall_at_3:.2f}"
    assert mean_mrr >= 0.60, f"Expected MRR >= 0.60, got {mean_mrr:.2f}"
    assert avg_latency < 50.0, f"Expected latency < 50ms, got {avg_latency:.2f}ms"


def test_provenance_integrity_under_evaluation(benchmark_corpus, benchmark_queries):
    """Verifies that 100% of retrieved benchmark results preserve complete metadata."""
    embedder = MockEmbedder(dimension=64)
    hybrid_retriever = HybridRetriever(
        semantic_retriever=FAISSRetriever(embedder=embedder),
        rrf_k=60,
    )
    hybrid_retriever.index_chunks(benchmark_corpus)

    for item in benchmark_queries:
        results = hybrid_retriever.retrieve(item["query"], top_k=3)
        for r in results:
            # Metadata integrity assertions
            assert r.chunk.metadata.filename.endswith(".pdf")
            assert isinstance(r.chunk.metadata.page_number, int)
            assert r.chunk.metadata.page_number >= 1
            assert r.chunk.metadata.chunk_id != ""
            assert r.chunk.text != ""
            assert r.rrf_score > 0.0
            assert r.rank >= 1

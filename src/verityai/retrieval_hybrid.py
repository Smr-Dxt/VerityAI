"""
VerityAI - Hybrid Document Retrieval Module (BM25 + FAISS + RRF)
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval

This module combines dense semantic retrieval (FAISS) and sparse lexical retrieval (BM25)
using Reciprocal Rank Fusion (RRF).

MATHEMATICAL FOUNDATION & SCORE FUSION CONSTRAINTS:
- Score Incommensurability: BM25 scores (unbounded positive floats) and cosine similarity
  scores (bounded in [-1.0, 1.0]) cannot be directly summed or compared without arbitrary,
  uncalibrated scaling.
- Reciprocal Rank Fusion (RRF): RRF merges disparate retrievers purely by their rank
  positions, assigning score contributions:
      RRF_score(d) = w_sem * (1 / (k + rank_sem(d))) + w_kw * (1 / (k + rank_kw(d)))
  where k is a rank-smoothing constant (standard default k=60).
- Deduplication: Chunks retrieved by both algorithms are unified under their stable chunk_id,
  rewarding consensus matches with higher combined rank.
- Provenance Preservation: Fused results maintain all original metadata (filename, page_number,
  chunk_id) and document the source ranks and raw scores.
"""

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Callable, List, Optional
import numpy as np

from verityai.chunking import DocumentChunk
from verityai.retrieval_semantic import FAISSRetriever, RetrievalResult


# ==============================================================================
# Lucene-Standard BM25 Engine
# ==============================================================================


class LuceneBM25Index:
    """In-memory Lucene-standard BM25 implementation.

    Uses the Robertson-Spärck Jones formulation with the +1.0 Lucene modification:
        IDF(q) = ln(1.0 + (N - df + 0.5) / (df + 0.5))
    This guarantees strictly non-negative IDF scores across all corpus sizes (even N=1),
    avoiding the negative IDF and zero division artifacts of standard BM25Okapi on small
    corpora.
    """

    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = (
            sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0.0
        )
        self.doc_freqs: Counter = Counter()
        for doc in corpus:
            self.doc_freqs.update(set(doc))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Calculates BM25 relevance scores for all documents in the corpus."""
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            df = self.doc_freqs[token]
            # Lucene-standard strictly positive IDF
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.corpus):
                tf = doc.count(token)
                if tf > 0:
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (
                        1.0 - self.b + self.b * (self.doc_lens[i] / (self.avgdl or 1.0))
                    )
                    scores[i] += idf * (numerator / denominator)
        return scores



# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass(frozen=True)
class HybridRetrievalResult:
    """A fused retrieval result combining semantic and keyword evidence.

    Attributes:
        chunk: Source DocumentChunk with complete metadata (filename, page_number, chunk_id).
        rrf_score: Reciprocal Rank Fusion score (higher = stronger consensus relevance).
        rank: 1-indexed final hybrid ranking position (1 = top match).
        semantic_rank: Original 1-indexed rank from FAISS semantic search (None if absent).
        keyword_rank: Original 1-indexed rank from BM25 lexical search (None if absent).
        semantic_score: Raw cosine similarity score (None if absent).
        keyword_score: Raw BM25 score (None if absent).
    """

    chunk: DocumentChunk
    rrf_score: float
    rank: int
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None


# ==============================================================================
# Default Tokenizer
# ==============================================================================


def default_word_tokenizer(text: str) -> List[str]:
    """Lightweight regex-based word tokenizer for BM25 lexical matching.

    Extracts lowercased alphanumeric word tokens, preserving acronyms and digits
    without external natural language processing libraries.

    Args:
        text: Input string to tokenize.

    Returns:
        List of lowercased token strings.
    """
    if not text:
        return []
    return re.findall(r"\b\w+\b", text.lower())


# ==============================================================================
# BM25 Lexical Retriever
# ==============================================================================


class BM25Retriever:
    """In-memory BM25 lexical search retriever powered by rank-bm25."""

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        """Initializes BM25Retriever.

        Args:
            tokenizer: Optional custom tokenization function. Defaults to default_word_tokenizer.
        """
        self.tokenizer = tokenizer or default_word_tokenizer
        self._bm25: Optional[LuceneBM25Index] = None
        self._chunks: List[DocumentChunk] = []

    @property
    def num_indexed(self) -> int:
        """Returns the number of indexed document chunks."""
        return len(self._chunks)

    def clear(self) -> None:
        """Clears the BM25 index and chunk references."""
        self._bm25 = None
        self._chunks = []

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Builds an in-memory BM25 index over the provided chunks.

        Args:
            chunks: List of DocumentChunk instances to index.

        Raises:
            TypeError: If chunks is not a list.
        """
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list of DocumentChunk instances, got {type(chunks)}")

        self.clear()

        if not chunks:
            return

        tokenized_corpus = [self.tokenizer(c.text) for c in chunks]
        self._bm25 = LuceneBM25Index(tokenized_corpus)
        self._chunks = list(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieves top-k chunks matching the query using BM25 lexical scoring.

        Args:
            query: Plain text search query.
            top_k: Number of ranked passages to return (must be > 0).

        Returns:
            List of RetrievalResult instances sorted by descending BM25 score.

        Raises:
            ValueError: If query is empty or whitespace-only, or if top_k <= 0.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self._bm25 is None or len(self._chunks) == 0:
            return []

        tokens = self.tokenizer(query.strip())
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)

        # Filter and rank chunks with positive score (lexical overlap)
        scored_pairs = [
            (idx, float(score))
            for idx, score in enumerate(raw_scores)
            if score > 0.0
        ]

        if not scored_pairs:
            return []

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = scored_pairs[:top_k]

        return [
            RetrievalResult(
                chunk=self._chunks[idx],
                score=score,
                rank=rank_idx,
            )
            for rank_idx, (idx, score) in enumerate(top_pairs, start=1)
        ]


# ==============================================================================
# Hybrid Retriever (FAISS + BM25 + Reciprocal Rank Fusion)
# ==============================================================================


class HybridRetriever:
    """Hybrid document retriever combining FAISS semantic search and BM25 keyword search.

    Uses Reciprocal Rank Fusion (RRF) to merge and deduplicate search results based
    on rank positions rather than uncalibrated raw score addition.
    """

    def __init__(
        self,
        semantic_retriever: FAISSRetriever,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf_k: int = 60,
    ) -> None:
        """Initializes the HybridRetriever.

        Args:
            semantic_retriever: Configured FAISSRetriever instance.
            bm25_retriever: Optional BM25Retriever instance (creates default if omitted).
            rrf_k: Rank-smoothing constant for Reciprocal Rank Fusion (default 60).

        Raises:
            ValueError: If rrf_k <= 0.
        """
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")

        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k

    @property
    def num_indexed(self) -> int:
        """Returns the number of chunks indexed in the hybrid system."""
        return self.semantic_retriever.num_indexed

    def clear(self) -> None:
        """Clears both the semantic and lexical indices."""
        self.semantic_retriever.clear()
        self.bm25_retriever.clear()

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Indexes document chunks into both FAISS and BM25 indices simultaneously.

        Args:
            chunks: List of DocumentChunk instances to index.
        """
        self.semantic_retriever.index_chunks(chunks)
        self.bm25_retriever.index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ) -> List[HybridRetrievalResult]:
        """Executes hybrid retrieval combining semantic and keyword evidence via RRF.

        Args:
            query: Plain-text search query.
            top_k: Number of combined results to return (must be > 0).
            semantic_weight: Multiplier weight for semantic rank contribution (default 1.0).
            keyword_weight: Multiplier weight for keyword rank contribution (default 1.0).

        Returns:
            List of HybridRetrievalResult instances sorted by descending RRF score.

        Raises:
            ValueError: If query is empty, top_k <= 0, or weights are negative.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if semantic_weight < 0.0 or keyword_weight < 0.0:
            raise ValueError("Retrieval weights must be non-negative.")

        if self.num_indexed == 0:
            return []

        # Retrieve a candidate pool from both retrievers
        candidate_depth = min(max(top_k * 3, 20), self.num_indexed)

        semantic_results = (
            self.semantic_retriever.retrieve(query, top_k=candidate_depth)
            if semantic_weight > 0.0
            else []
        )
        keyword_results = (
            self.bm25_retriever.retrieve(query, top_k=candidate_depth)
            if keyword_weight > 0.0
            else []
        )

        chunk_by_id = {}
        rrf_scores = {}
        sem_ranks = {}
        kw_ranks = {}
        sem_scores = {}
        kw_scores = {}

        # Accumulate semantic rank contributions
        for res in semantic_results:
            cid = res.chunk.metadata.chunk_id
            chunk_by_id[cid] = res.chunk
            sem_ranks[cid] = res.rank
            sem_scores[cid] = res.score
            score_contrib = semantic_weight * (1.0 / (self.rrf_k + res.rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib

        # Accumulate keyword rank contributions
        for res in keyword_results:
            cid = res.chunk.metadata.chunk_id
            chunk_by_id[cid] = res.chunk
            kw_ranks[cid] = res.rank
            kw_scores[cid] = res.score
            score_contrib = keyword_weight * (1.0 / (self.rrf_k + res.rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib

        if not rrf_scores:
            return []

        # Sort deduplicated chunks by RRF score descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        fused_results: List[HybridRetrievalResult] = []
        for final_rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            fused_results.append(
                HybridRetrievalResult(
                    chunk=chunk_by_id[cid],
                    rrf_score=float(rrf_scores[cid]),
                    rank=final_rank,
                    semantic_rank=sem_ranks.get(cid),
                    keyword_rank=kw_ranks.get(cid),
                    semantic_score=sem_scores.get(cid),
                    keyword_score=kw_scores.get(cid),
                )
            )

        return fused_results

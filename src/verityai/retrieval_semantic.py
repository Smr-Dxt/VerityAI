"""
VerityAI - Semantic Document Retrieval Module
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval

This module provides in-memory dense vector indexing and semantic similarity search
using FAISS (Facebook AI Similarity Search).

ARCHITECTURE CONSTRAINTS & DESIGN:
- Index Type: faiss.IndexFlatIP (Exact Inner Product search).
- Metric: Because embeddings are L2-normalized unit vectors, the inner product equals
  cosine similarity in the range [-1.0, 1.0].
- Score Semantics: Scores represent geometric vector proximity, NOT calibrated probabilities
  of correctness or truth.
- Provenance Guarantee: Strict 1-to-1 deterministic positional mapping between FAISS
  row indices and source DocumentChunk metadata (filename, page_number, chunk_id).
- Privacy & Ephemerality: Indices are maintained purely in RAM and never written to disk,
  preventing document leakage.
"""

from dataclasses import dataclass
from typing import List, Optional
import faiss
import numpy as np

from verityai.chunking import DocumentChunk
from verityai.embeddings import BaseEmbedder


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked search result carrying passage text, similarity score, and source provenance.

    Attributes:
        chunk: The source DocumentChunk instance (includes text, filename, page_number, chunk_id).
        score: Geometric similarity score (inner product / cosine similarity in [-1.0, 1.0]).
               NOTE: This is a ranking metric, NOT a calibrated probability.
        rank: 1-indexed retrieval ranking (1 = highest scoring match).
    """

    chunk: DocumentChunk
    score: float
    rank: int


# ==============================================================================
# FAISS Vector Retriever
# ==============================================================================


class FAISSRetriever:
    """In-memory semantic vector retriever powered by FAISS.

    Provides deterministic indexing, query embedding, and top-k vector retrieval
    while preserving page-level document provenance.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        """Initializes the FAISSRetriever with an embedding service.

        Args:
            embedder: BaseEmbedder instance (e.g. LocalEmbedder or MockEmbedder).
        """
        self.embedder = embedder
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: List[DocumentChunk] = []

    @property
    def num_indexed(self) -> int:
        """Returns the number of document chunks currently indexed."""
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        """Returns the vector dimensionality of the embedder and index."""
        return self.embedder.dimension

    def clear(self) -> None:
        """Clears the in-memory index and releases indexed chunk references."""
        self._index = None
        self._chunks = []

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Builds an in-memory FAISS IndexFlatIP from a list of DocumentChunks.

        Maintains a deterministic 1-to-1 correspondence: FAISS row i corresponds
        strictly to chunks[i].

        Args:
            chunks: List of DocumentChunk instances to index.

        Raises:
            TypeError: If chunks is not a list.
            ValueError: If embedding dimensionality is incompatible with index.
        """
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list of DocumentChunk instances, got {type(chunks)}")

        self.clear()

        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match retriever dimension {self.dimension}"
            )

        # FAISS requires contiguous C-order float32 arrays
        contiguous_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Create Exact Inner Product index
        index = faiss.IndexFlatIP(self.dimension)
        index.add(contiguous_embeddings)

        self._index = index
        self._chunks = list(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieves top-k semantically relevant chunks for a search query.

        Args:
            query: Plain text research query string.
            top_k: Number of ranked passages to return (must be > 0).

        Returns:
            List of RetrievalResult instances sorted by relevance (descending score).

        Raises:
            ValueError: If query is empty or whitespace-only, or if top_k <= 0.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self._index is None or len(self._chunks) == 0:
            return []

        k = min(top_k, len(self._chunks))

        # Generate query vector
        q_vec = self.embedder.embed_query(query.strip())
        q_matrix = np.ascontiguousarray(np.expand_dims(q_vec, axis=0), dtype=np.float32)

        # Perform inner product search on FAISS index
        distances, indices = self._index.search(q_matrix, k)

        results: List[RetrievalResult] = []
        for rank_idx, (score, chunk_idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if chunk_idx < 0 or chunk_idx >= len(self._chunks):
                continue

            results.append(
                RetrievalResult(
                    chunk=self._chunks[chunk_idx],
                    score=float(score),
                    rank=rank_idx,
                )
            )

        return results

"""
VerityAI - Local Text Embeddings Module
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval

This module generates dense vector representations (embeddings) from document text chunks
and queries using local Sentence Transformers models.

MODEL DETAILS & ARCHITECTURE CONSTRAINTS:
- Selected Model: sentence-transformers/all-MiniLM-L6-v2
- License: Apache 2.0 (permissive open-source for academic, personal, and commercial use)
- Footprint: ~80 MB disk storage, ~120 MB RAM footprint
- Dimension: 384-dimensional dense vectors
- Execution: Optimized for CPU inference; automatically leverages CUDA (RTX 3050 4GB) if available
- Offline Privacy: Once downloaded or cached locally, all inferences run 100% offline
  with zero external network requests and zero logging of document text.
"""

from abc import ABC, abstractmethod
import hashlib
from typing import List, Optional
import numpy as np


# ==============================================================================
# Abstract Base Embedder
# ==============================================================================


class BaseEmbedder(ABC):
    """Abstract interface defining required methods for text embedding services."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the fixed embedding dimensionality (e.g. 384)."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of text strings into a 2D float32 numpy array of shape (N, dimension)."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D float32 numpy array of shape (dimension,)."""
        pass


# ==============================================================================
# Production Local Embedder (Sentence Transformers)
# ==============================================================================


class LocalEmbedder(BaseEmbedder):
    """Local, offline embedding generator powered by Sentence Transformers.

    Features:
    - Lazy loading: Postpones model initialization until first embedding invocation.
    - Hardware adaptive: Auto-detects NVIDIA CUDA GPU acceleration when available,
      falling back cleanly to CPU inference.
    - L2 normalization: Normalizes output vectors so inner products equal cosine similarity.
    - Validation: Guardrails against empty inputs, dimensional mismatches, and NaNs.
    - Privacy: Never logs document or query text.
    """

    def __init__(
        self,
        model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        cache_folder: Optional[str] = None,
    ) -> None:
        """Initializes the LocalEmbedder configuration.

        Args:
            model_name_or_path: HuggingFace model repo identifier or local directory path.
            device: Execution target ("cuda", "cpu", or None for auto-detection).
            normalize_embeddings: Whether to L2-normalize vectors to unit length.
            batch_size: Number of passages encoded per forward pass.
            cache_folder: Optional local directory where downloaded weights are cached.
        """
        self.model_name_or_path = model_name_or_path
        self._configured_device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.cache_folder = cache_folder

        self._model = None
        self._resolved_device: Optional[str] = None
        self._dimension: int = 384  # Default for all-MiniLM-L6-v2

    @property
    def is_loaded(self) -> bool:
        """Returns True if the underlying model has been loaded into memory."""
        return self._model is not None

    @property
    def device(self) -> str:
        """Returns the resolved device ('cuda' or 'cpu')."""
        if self._resolved_device is None:
            self._resolve_device()
        return self._resolved_device  # type: ignore

    @property
    def dimension(self) -> int:
        """Returns the vector dimensionality."""
        return self._dimension

    def _resolve_device(self) -> None:
        """Determines whether to execute on CUDA or CPU."""
        if self._configured_device:
            self._resolved_device = self._configured_device
            return

        try:
            import torch

            if torch.cuda.is_available():
                self._resolved_device = "cuda"
            else:
                self._resolved_device = "cpu"
        except ImportError:
            self._resolved_device = "cpu"

    def load(self) -> None:
        """Explicitly loads the model into memory if not already initialized."""
        if self._model is not None:
            return

        self._resolve_device()

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_name_or_path=self.model_name_or_path,
            device=self._resolved_device,
            cache_folder=self.cache_folder,
        )
        # Dynamically record dimension from loaded model
        try:
            dim = self._model.get_sentence_embedding_dimension()
            if dim is not None:
                self._dimension = int(dim)
        except Exception:
            pass

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a batch of text chunks.

        Args:
            texts: List of string passages to encode.

        Returns:
            2D numpy array of shape (len(texts), dimension) and dtype float32.

        Raises:
            TypeError: If texts is not a list.
            ValueError: If embeddings contain NaN or infinite values.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list of strings, got {type(texts)}")

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # Ensure all elements are strings
        cleaned_texts = [str(t) if t is not None else "" for t in texts]

        self.load()
        assert self._model is not None

        raw_embeddings = self._model.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        # Validate shape and numerical integrity
        if embeddings.shape != (len(texts), self.dimension):
            raise ValueError(
                f"Embedding shape mismatch: expected ({len(texts)}, {self.dimension}), got {embeddings.shape}"
            )

        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            raise ValueError("Generated embeddings contain NaN or infinite values.")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generates an embedding vector for a single research query.

        Args:
            query: Query string.

        Returns:
            1D numpy array of shape (dimension,) and dtype float32.

        Raises:
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")

        results = self.embed_texts([query.strip()])
        return results[0]


# ==============================================================================
# Deterministic Mock Embedder (For Fast Unit Testing & CI)
# ==============================================================================


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedding generator for fast, offline unit tests.

    Uses a cryptographic hash of each text string to seed a pseudo-random
    generator, creating reproducible, unit-normalized vectors without downloading
    any model weights or requiring network access.
    """

    def __init__(self, dimension: int = 384, normalize: bool = True) -> None:
        """Initializes the mock embedder.

        Args:
            dimension: Dimensionality of mock vectors (default 384).
            normalize: Whether to normalize generated vectors to L2 norm 1.0.
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dim = dimension
        self.normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates deterministic mock vectors for the given texts."""
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        matrix = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Seed pseudo-random generator with text hash
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            if self.normalize:
                norm = np.linalg.norm(vec)
                if norm > 1e-12:
                    vec = vec / norm
            matrix[i] = vec

        return matrix

    def embed_query(self, query: str) -> np.ndarray:
        """Generates a deterministic mock vector for a query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only.")
        return self.embed_texts([query.strip()])[0]

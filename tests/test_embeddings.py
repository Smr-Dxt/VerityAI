"""
Unit tests for local text embedding generation.
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval
"""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from verityai.embeddings import BaseEmbedder, LocalEmbedder, MockEmbedder


# ==============================================================================
# MockEmbedder Tests (Fast Deterministic Unit Tests)
# ==============================================================================


def test_mock_embedder_dimensions_and_shape():
    """MockEmbedder must produce output matching the configured dimension."""
    embedder = MockEmbedder(dimension=384)
    assert embedder.dimension == 384

    texts = ["First passage", "Second passage", "Third passage"]
    vecs = embedder.embed_texts(texts)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (3, 384)
    assert vecs.dtype == np.float32


def test_mock_embedder_determinism():
    """Embedding the exact same text must produce the exact same vector."""
    embedder = MockEmbedder(dimension=128)
    text = "Deterministic vector generation check."

    vec_1 = embedder.embed_texts([text])
    vec_2 = embedder.embed_texts([text])

    np.testing.assert_array_equal(vec_1, vec_2)


def test_mock_embedder_different_texts_differ():
    """Different texts must produce distinct vectors."""
    embedder = MockEmbedder(dimension=128)
    vecs = embedder.embed_texts(["Apples", "Oranges"])

    assert not np.array_equal(vecs[0], vecs[1])


def test_mock_embedder_l2_normalization():
    """When normalize=True, all generated vectors must have an L2 norm of 1.0."""
    embedder = MockEmbedder(dimension=384, normalize=True)
    vecs = embedder.embed_texts(["Alpha", "Beta", "Gamma"])

    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-5)


def test_mock_embedder_empty_input():
    """Empty list must return an empty 2D array with shape (0, dimension)."""
    embedder = MockEmbedder(dimension=384)
    vecs = embedder.embed_texts([])
    assert vecs.shape == (0, 384)
    assert vecs.dtype == np.float32


def test_mock_embedder_embed_query():
    """embed_query must return a 1D vector of shape (dimension,)."""
    embedder = MockEmbedder(dimension=384)
    query_vec = embedder.embed_query("What is the speed of light?")

    assert isinstance(query_vec, np.ndarray)
    assert query_vec.shape == (384,)
    assert query_vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(query_vec), 1.0, atol=1e-5)


def test_mock_embedder_query_validation():
    """Empty or whitespace-only queries must raise ValueError."""
    embedder = MockEmbedder()
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        embedder.embed_query("")

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        embedder.embed_query("   \n\t  ")


def test_mock_embedder_type_errors():
    """Invalid input types must raise appropriate exceptions."""
    embedder = MockEmbedder()
    with pytest.raises(TypeError, match="must be a list of strings"):
        embedder.embed_texts("not a list")  # type: ignore

    with pytest.raises(ValueError, match="dimension must be positive"):
        MockEmbedder(dimension=0)


# ==============================================================================
# LocalEmbedder Unit Tests (Isolated with Mocked SentenceTransformer)
# ==============================================================================


def test_local_embedder_lazy_loading():
    """LocalEmbedder must not initialize the neural model until first use."""
    embedder = LocalEmbedder(model_name_or_path="mock-repo/mock-model")
    assert not embedder.is_loaded
    assert embedder._model is None


def test_local_embedder_empty_input_no_model_call():
    """Passing an empty list returns (0, dimension) without triggering model loading."""
    embedder = LocalEmbedder()
    vecs = embedder.embed_texts([])
    assert vecs.shape == (0, 384)
    assert not embedder.is_loaded


def test_local_embedder_query_empty_validation():
    """Query validation must reject blank strings before loading model."""
    embedder = LocalEmbedder()
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        embedder.embed_query("   ")
    assert not embedder.is_loaded


def test_local_embedder_mocked_execution():
    """Verifies LocalEmbedder forwards parameters correctly to SentenceTransformer."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st_cls:
        mock_instance = MagicMock()
        mock_st_cls.return_value = mock_instance
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        # Return mock normalized float32 vectors
        fake_vecs = np.ones((2, 384), dtype=np.float32) / np.sqrt(384)
        mock_instance.encode.return_value = fake_vecs

        embedder = LocalEmbedder(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            normalize_embeddings=True,
            batch_size=16,
        )

        res = embedder.embed_texts(["Document chunk one", "Document chunk two"])

        # Model loaded and configured with CPU
        assert embedder.is_loaded
        mock_st_cls.assert_called_once_with(
            model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            cache_folder=None,
        )

        # encode called with correct parameters
        mock_instance.encode.assert_called_once_with(
            ["Document chunk one", "Document chunk two"],
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        assert res.shape == (2, 384)
        assert res.dtype == np.float32


def test_local_embedder_detects_nan_values():
    """If model outputs corrupted NaN values, LocalEmbedder must raise ValueError."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st_cls:
        mock_instance = MagicMock()
        mock_st_cls.return_value = mock_instance
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        corrupted_vecs = np.full((1, 384), np.nan, dtype=np.float32)
        mock_instance.encode.return_value = corrupted_vecs

        embedder = LocalEmbedder(device="cpu")
        with pytest.raises(ValueError, match="contain NaN or infinite values"):
            embedder.embed_texts(["Some text"])


# ==============================================================================
# Optional Real Model Integration Test
# ==============================================================================


@pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="Set RUN_REAL_MODEL_TESTS=1 to run real neural model download and encoding",
)
def test_real_model_download_and_embed():
    """Optional end-to-end integration test verifying real all-MiniLM-L6-v2 model execution."""
    embedder = LocalEmbedder(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    vecs = embedder.embed_texts(["VerityAI local embedding test on Windows."])
    assert vecs.shape == (1, 384)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-4)

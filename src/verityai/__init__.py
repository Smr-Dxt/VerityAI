"""
VerityAI - Offline-First Agentic Research Assistant
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval
"""

__app_name__ = "VerityAI"
__version__ = "0.3.0"
__phase__ = "Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval"

from verityai.chunking import (
    ChunkMetadata,
    DocumentChunk,
    chunk_document,
    generate_doc_id,
    split_text_into_chunks,
)
from verityai.config import AppConfig, get_config
from verityai.embeddings import BaseEmbedder, LocalEmbedder, MockEmbedder
from verityai.retrieval_hybrid import BM25Retriever, HybridRetrievalResult, HybridRetriever
from verityai.retrieval_semantic import FAISSRetriever, RetrievalResult
from verityai.ingestion import (
    CorruptedFileError,
    DocumentResult,
    EmptyDocumentError,
    EncryptedPDFError,
    FileSizeLimitExceededError,
    IngestionError,
    InvalidEncodingError,
    PageContent,
    ScannedPDFError,
    UnsupportedFileTypeError,
    extract_pdf,
    extract_txt,
    ingest_document,
    safe_ingest_document,
)

__all__ = [
    "__app_name__",
    "__version__",
    "__phase__",
    "AppConfig",
    "get_config",
    "DocumentResult",
    "PageContent",
    "ChunkMetadata",
    "DocumentChunk",
    "chunk_document",
    "split_text_into_chunks",
    "generate_doc_id",
    "BaseEmbedder",
    "LocalEmbedder",
    "MockEmbedder",
    "FAISSRetriever",
    "RetrievalResult",
    "BM25Retriever",
    "HybridRetriever",
    "HybridRetrievalResult",
    "IngestionError",
    "UnsupportedFileTypeError",
    "FileSizeLimitExceededError",
    "EmptyDocumentError",
    "InvalidEncodingError",
    "CorruptedFileError",
    "EncryptedPDFError",
    "ScannedPDFError",
    "extract_txt",
    "extract_pdf",
    "ingest_document",
    "safe_ingest_document",
]

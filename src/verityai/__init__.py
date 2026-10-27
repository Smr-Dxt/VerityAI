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
from verityai.citation_validator import (
    CitationErrorCode,
    CitationValidationReport,
    CitationValidationResult,
    CitationValidator,
    validate_citation,
    validate_citations,
)
from verityai.config import AppConfig, get_config
from verityai.embeddings import BaseEmbedder, LocalEmbedder, MockEmbedder
from verityai.evidence import (
    Citation,
    CitationReference,
    EvidencePassage,
    VerificationOutcome,
    VerificationResult,
    VerificationStatus,
)
from verityai.verifier import (
    BaseClaimVerifier,
    DeterministicClaimVerifier,
    verify_claim,
)
from verityai.contradiction import (
    BaseContradictionDetector,
    ContradictionReasonCode,
    ContradictionReport,
    ContradictionStatus,
    CrossDocumentConflict,
    DeterministicContradictionDetector,
    detect_cross_document_contradictions,
)
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
    "EvidencePassage",
    "CitationReference",
    "Citation",
    "VerificationStatus",
    "VerificationOutcome",
    "VerificationResult",
    "CitationErrorCode",
    "CitationValidationResult",
    "CitationValidationReport",
    "CitationValidator",
    "validate_citation",
    "validate_citations",
    "BaseClaimVerifier",
    "DeterministicClaimVerifier",
    "verify_claim",
    "ContradictionStatus",
    "ContradictionReasonCode",
    "CrossDocumentConflict",
    "ContradictionReport",
    "BaseContradictionDetector",
    "DeterministicContradictionDetector",
    "detect_cross_document_contradictions",
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



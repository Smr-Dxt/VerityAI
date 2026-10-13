"""
VerityAI - Offline-First Agentic Research Assistant
Phase 2: Local Document Ingestion
"""

__app_name__ = "VerityAI"
__version__ = "0.2.0"
__phase__ = "Phase 2: Local Document Ingestion"

from verityai.config import AppConfig, get_config
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

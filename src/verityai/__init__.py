"""
VerityAI - Offline-First Agentic Research Assistant
Phase 1: Project Foundation
"""

__app_name__ = "VerityAI"
__version__ = "0.1.0"
__phase__ = "Phase 1: Project Foundation"

from verityai.config import AppConfig, get_config
from verityai.ingestion import (
    DocumentResult,
    EmptyDocumentError,
    FileSizeLimitExceededError,
    IngestionError,
    InvalidEncodingError,
    PageContent,
    UnsupportedFileTypeError,
    extract_txt,
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
    "extract_txt",
]

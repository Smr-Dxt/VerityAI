"""
VerityAI - Source-Aware Document Chunking Module
Phase 3: Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval

This module splits ingested DocumentResult instances (TXT and PDF) into manageable,
overlapping chunks while strictly preserving source provenance (filename, doc_id,
and 1-indexed PDF page numbers).

NOTE ON TOKEN VS. CHARACTER COUNT:
This implementation uses character-based windows with word-boundary snapping
rather than a model-specific subword tokenizer (e.g. Byte-Pair Encoding or WordPiece).
As a general rule of thumb for English text:
    - 1 token ≈ 4 characters (approx. 0.75 words).
    - Default chunk_size = 1,200 characters corresponds to approx. 250–300 tokens
      (approx. 200–250 words).
    - Default chunk_overlap = 200 characters corresponds to approx. 40–50 tokens
      (approx. 35–45 words).
Using character boundaries with word snapping avoids loading heavy neural tokenizers
during document ingestion while preventing sliced words.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

from verityai.ingestion import DocumentResult, PageContent


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata detailing the provenance and geometry of a text chunk.

    Attributes:
        chunk_id: Stable, deterministic identifier for this chunk (e.g., "doc_a1b2_p1_c0").
        doc_id: Deterministic document identifier derived from filename and content hash.
        filename: Sanitized original document filename.
        doc_type: Document format ("pdf" or "txt").
        page_number: 1-indexed page number for PDFs (or 1 for TXT documents).
        chunk_index: 0-indexed sequence number of this chunk within the overall document.
        char_count: Exact character count of the chunk's text.
        word_count: Number of whitespace-separated words in the chunk's text.
    """

    chunk_id: str
    doc_id: str
    filename: str
    doc_type: str
    page_number: Optional[int]
    chunk_index: int
    char_count: int
    word_count: int


@dataclass(frozen=True)
class DocumentChunk:
    """A single retrievable text passage paired with complete source metadata.

    Attributes:
        text: Plain-text passage content.
        metadata: ChunkMetadata specifying source file, page, and chunk index.
    """

    text: str
    metadata: ChunkMetadata


# ==============================================================================
# Helper Functions
# ==============================================================================


def generate_doc_id(filename: str, content: str) -> str:
    """Generates a stable, deterministic identifier for a document.

    Uses SHA-256 over the filename and content to ensure identical documents
    always receive the exact same document ID.

    Args:
        filename: Document filename.
        content: Extracted text content.

    Returns:
        Deterministic string identifier with prefix "doc_" and 12 hex characters.
    """
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8"))
    hasher.update(b"::")
    hasher.update(content.encode("utf-8"))
    digest = hasher.hexdigest()[:12]
    return f"doc_{digest}"


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[str]:
    """Splits text into overlapping windows without slicing words midway.

    Slides a window across the text, looking backwards from the target boundary
    to find whitespace, ensuring chunks end cleanly on full words.

    Args:
        text: Raw text string to split.
        chunk_size: Maximum target window length in characters (must be > 0).
        chunk_overlap: Overlap length in characters (must satisfy 0 <= overlap < chunk_size).

    Returns:
        List of non-empty text chunk strings.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    clean_text = text.strip()
    if not clean_text:
        return []

    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks: List[str] = []
    text_len = len(clean_text)
    start = 0

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            # Reached the end of the text
            tail = clean_text[start:].strip()
            if tail:
                chunks.append(tail)
            break

        # Search backwards from target end to find a whitespace boundary
        window = clean_text[start:end]
        match = re.search(r"\s+(?=\S*$)", window)

        if match and match.start() > 0:
            split_point = start + match.start()
        else:
            # Fallback: No whitespace found in window (e.g. single giant word/URL)
            split_point = end

        chunk_str = clean_text[start:split_point].strip()
        if chunk_str:
            chunks.append(chunk_str)

        # Compute next start position using overlap
        if chunk_overlap > 0:
            raw_next = split_point - chunk_overlap
            # Snap next_start to a word start so the next chunk doesn't start mid-word
            space_pos = clean_text.rfind(" ", start, max(start, raw_next))
            if space_pos != -1 and space_pos >= start:
                next_start = space_pos + 1
            else:
                next_start = raw_next
        else:
            next_start = split_point

        # Ensure guaranteed forward progress to prevent infinite loops
        if next_start <= start:
            next_start = split_point

        # Advance past any leading whitespace
        while next_start < text_len and clean_text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks


# ==============================================================================
# Document Chunking Dispatcher
# ==============================================================================


def chunk_document(
    doc: DocumentResult,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[DocumentChunk]:
    """Splits an ingested DocumentResult into source-aware DocumentChunk items.

    Strictly preserves source-level provenance:
    - PDF documents: Chunks each PageContent independently. Chunks never cross
      page boundaries, ensuring every chunk has an unambiguous 1-indexed page_number.
    - TXT documents: Chunks across the document body. Page number is 1, matching
      Phase 2 single-page representation.

    Args:
        doc: DocumentResult instance produced by Phase 2 ingestion.
        chunk_size: Target maximum chunk length in characters (default 1,200).
        chunk_overlap: Overlap length in characters between adjacent chunks (default 200).

    Returns:
        List of DocumentChunk instances with deterministic IDs and complete provenance.

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size.
    """
    if doc.status == "error" or not doc.pages or not doc.extracted_text.strip():
        return []

    doc_id = generate_doc_id(doc.filename, doc.extracted_text)
    chunks: List[DocumentChunk] = []
    global_chunk_idx = 0

    for page in doc.pages:
        page_text = page.text.strip()
        if not page_text:
            continue

        raw_chunks = split_text_into_chunks(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for page_chunk_idx, chunk_text in enumerate(raw_chunks):
            if doc.doc_type == "pdf":
                chunk_id = f"{doc_id}_p{page.page_number}_c{page_chunk_idx}"
                page_num: Optional[int] = page.page_number
            else:
                chunk_id = f"{doc_id}_c{global_chunk_idx}"
                page_num = 1

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                filename=doc.filename,
                doc_type=doc.doc_type,
                page_number=page_num,
                chunk_index=global_chunk_idx,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
            )

            chunks.append(DocumentChunk(text=chunk_text, metadata=metadata))
            global_chunk_idx += 1

    return chunks

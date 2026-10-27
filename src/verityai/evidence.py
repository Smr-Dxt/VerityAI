"""
VerityAI - Typed Evidence and Verification Models
Phase 4: Agentic Verification and Citations (Chunk 1)

This module defines typed, immutable data models for:
- Evidence passages (extracted or retrieved textual evidence with provenance)
- Citation references (structured citations pointing to source documents and pages)
- Verification outcomes (claim verification status and supporting evidence)

These models bridge Phase 3 hybrid retrieval results and Phase 4 agentic verification.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from verityai.chunking import DocumentChunk
from verityai.retrieval_semantic import RetrievalResult
from verityai.retrieval_hybrid import HybridRetrievalResult


# ==============================================================================
# Verification Status Enumeration
# ==============================================================================


class VerificationStatus(str, Enum):
    """Verification verdict status for factual claims.

    Attributes:
        SUPPORTED: The claim is directly corroborated and confirmed by evidence.
        CONTRADICTED: The claim conflicts with or is refuted by evidence.
        PARTIALLY_SUPPORTED: The claim is partially supported, but has unverified or mixed aspects.
        UNVERIFIED: Insufficient or inconclusive evidence to verify or refute the claim.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNVERIFIED = "unverified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNVERIFIABLE = "unverifiable"

    def __str__(self) -> str:
        return self.value


# ==============================================================================
# Evidence Passage Model
# ==============================================================================


@dataclass(frozen=True)
class EvidencePassage:
    """A retrievable evidence passage with complete document provenance.

    Represents a specific passage or chunk extracted from a source document
    to be evaluated in claim verification.

    Attributes:
        text: Plain-text excerpt or passage content (non-empty).
        doc_id: Deterministic document identifier.
        filename: Original sanitized document filename.
        chunk_id: Stable identifier for the underlying chunk.
        page_number: 1-indexed page number for PDFs (or None for TXT documents).
        score: Optional retrieval score (e.g. cosine similarity, BM25, or RRF).
        rank: Optional 1-indexed retrieval ranking.
        doc_type: Optional document format ("pdf", "txt", etc.).
    """

    text: str
    doc_id: str
    filename: str
    chunk_id: str
    page_number: Optional[int] = None
    score: Optional[float] = None
    rank: Optional[int] = None
    doc_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Validates model attributes upon initialization."""
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Evidence text must be a non-empty string.")

        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("doc_id must be a non-empty string.")

        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must be a non-empty string.")

        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string.")

        if self.page_number is not None:
            if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
                raise TypeError(
                    f"page_number must be an integer or None, got {type(self.page_number).__name__}"
                )
            if self.page_number < 1:
                raise ValueError(
                    f"page_number must be a positive integer (>= 1), got {self.page_number}"
                )

        if self.rank is not None:
            if isinstance(self.rank, bool) or not isinstance(self.rank, int):
                raise TypeError(
                    f"rank must be an integer or None, got {type(self.rank).__name__}"
                )
            if self.rank < 1:
                raise ValueError(f"rank must be a positive integer (>= 1), got {self.rank}")

        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise TypeError(
                    f"score must be a numeric float or int, got {type(self.score).__name__}"
                )

    @classmethod
    def from_chunk(
        cls,
        chunk: DocumentChunk,
        score: Optional[float] = None,
        rank: Optional[int] = None,
    ) -> "EvidencePassage":
        """Constructs an EvidencePassage directly from a DocumentChunk.

        For TXT documents, page_number is normalized to None unless doc_type is PDF.
        """
        page_num = (
            chunk.metadata.page_number
            if chunk.metadata.doc_type == "pdf"
            else None
        )
        return cls(
            text=chunk.text,
            doc_id=chunk.metadata.doc_id,
            filename=chunk.metadata.filename,
            chunk_id=chunk.metadata.chunk_id,
            page_number=page_num,
            score=score,
            rank=rank,
            doc_type=chunk.metadata.doc_type,
        )

    @classmethod
    def from_retrieval_result(cls, result: RetrievalResult) -> "EvidencePassage":
        """Constructs an EvidencePassage from a semantic RetrievalResult."""
        return cls.from_chunk(
            chunk=result.chunk,
            score=result.score,
            rank=result.rank,
        )

    @classmethod
    def from_hybrid_result(cls, result: HybridRetrievalResult) -> "EvidencePassage":
        """Constructs an EvidencePassage from a HybridRetrievalResult."""
        return cls.from_chunk(
            chunk=result.chunk,
            score=result.rrf_score,
            rank=result.rank,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the evidence passage to a dictionary."""
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "score": self.score,
            "rank": self.rank,
            "doc_type": self.doc_type,
        }


# ==============================================================================
# Citation Reference Model
# ==============================================================================


@dataclass(frozen=True)
class CitationReference:
    """A formal citation referencing a source document, page, and chunk.

    Used to ground verified answers in verifiable source documents.

    Attributes:
        citation_id: Identifier label (e.g. "1", "[1]", "cite_1").
        doc_id: Deterministic document identifier.
        filename: Source document filename.
        chunk_id: Stable chunk identifier cited.
        page_number: 1-indexed page number for PDFs (or None for TXT documents).
        quote: Optional exact quote or supporting text snippet.
        rank: Optional retrieval rank of the cited evidence.
        score: Optional retrieval score of the cited evidence.
    """

    citation_id: str
    doc_id: str
    filename: str
    chunk_id: str
    page_number: Optional[int] = None
    quote: Optional[str] = None
    rank: Optional[int] = None
    score: Optional[float] = None

    def __post_init__(self) -> None:
        """Validates citation attributes upon initialization."""
        if not isinstance(self.citation_id, str) or not self.citation_id.strip():
            raise ValueError("citation_id must be a non-empty string.")

        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("doc_id must be a non-empty string.")

        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must be a non-empty string.")

        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string.")

        if self.page_number is not None:
            if isinstance(self.page_number, bool) or not isinstance(self.page_number, int):
                raise TypeError(
                    f"page_number must be an integer or None, got {type(self.page_number).__name__}"
                )
            if self.page_number < 1:
                raise ValueError(
                    f"page_number must be a positive integer (>= 1), got {self.page_number}"
                )

        if self.rank is not None:
            if isinstance(self.rank, bool) or not isinstance(self.rank, int):
                raise TypeError(
                    f"rank must be an integer or None, got {type(self.rank).__name__}"
                )
            if self.rank < 1:
                raise ValueError(f"rank must be a positive integer (>= 1), got {self.rank}")

        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise TypeError(
                    f"score must be a numeric float or int, got {type(self.score).__name__}"
                )

    @classmethod
    def from_evidence(
        cls,
        evidence: EvidencePassage,
        citation_id: str = "1",
        quote: Optional[str] = None,
    ) -> "CitationReference":
        """Constructs a CitationReference from an EvidencePassage."""
        return cls(
            citation_id=citation_id,
            doc_id=evidence.doc_id,
            filename=evidence.filename,
            chunk_id=evidence.chunk_id,
            page_number=evidence.page_number,
            quote=quote,
            rank=evidence.rank,
            score=evidence.score,
        )

    def format_inline(self) -> str:
        """Returns a short inline citation string (e.g. '[report.pdf, p. 3]' or '[notes.txt]')."""
        if self.page_number is not None:
            return f"[{self.filename}, p. {self.page_number}]"
        return f"[{self.filename}]"

    def format_reference(self) -> str:
        """Returns a formatted bibliographic citation entry."""
        page_str = f", p. {self.page_number}" if self.page_number is not None else ""
        quote_str = f': "{self.quote}"' if self.quote else ""
        return f"[{self.citation_id}] {self.filename}{page_str}{quote_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the citation reference to a dictionary."""
        return {
            "citation_id": self.citation_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "quote": self.quote,
            "rank": self.rank,
            "score": self.score,
        }


# ==============================================================================
# Verification Outcome Model
# ==============================================================================


@dataclass(frozen=True)
class VerificationOutcome:
    """The structured result of verifying a factual claim against evidence.

    Attributes:
        claim: Plain-text claim statement being verified (non-empty).
        status: VerificationStatus outcome verdict.
        evidence: Tuple of EvidencePassage items evaluated for this claim.
        citations: Tuple of CitationReference items supporting the verdict.
        confidence: Optional confidence score between 0.0 and 1.0.
        explanation: Optional plain-text explanation of the reasoning.
        contradiction_details: Optional description of conflicting information if contradicted.
    """

    claim: str
    status: VerificationStatus
    evidence: Sequence[EvidencePassage] = field(default_factory=tuple)
    citations: Sequence[CitationReference] = field(default_factory=tuple)
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    contradiction_details: Optional[str] = None

    def __post_init__(self) -> None:
        """Validates and standardizes verification outcome fields."""
        # Handle string status conversion first so status is known
        if isinstance(self.status, str):
            try:
                object.__setattr__(self, "status", VerificationStatus(self.status.lower()))
            except ValueError:
                valid_statuses = [s.value for s in VerificationStatus]
                raise ValueError(
                    f"Invalid status '{self.status}'. Must be one of: {valid_statuses}"
                )
        elif not isinstance(self.status, VerificationStatus):
            raise TypeError(
                f"status must be a VerificationStatus or str, got {type(self.status).__name__}"
            )

        # Allow empty or whitespace claim only when status is UNVERIFIABLE
        if not isinstance(self.claim, str) or (
            not self.claim.strip() and self.status != VerificationStatus.UNVERIFIABLE
        ):
            raise ValueError("claim must be a non-empty string.")

        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise TypeError(
                    f"confidence must be a float or int, got {type(self.confidence).__name__}"
                )
            conf_val = float(self.confidence)
            if not (0.0 <= conf_val <= 1.0):
                raise ValueError(
                    f"confidence must be between 0.0 and 1.0, got {self.confidence}"
                )

        if not isinstance(self.evidence, (list, tuple)):
            raise TypeError(
                f"evidence must be a list or tuple of EvidencePassage, got {type(self.evidence).__name__}"
            )
        for item in self.evidence:
            if not isinstance(item, EvidencePassage):
                raise TypeError(
                    f"Each item in evidence must be an EvidencePassage, got {type(item).__name__}"
                )
        object.__setattr__(self, "evidence", tuple(self.evidence))

        if not isinstance(self.citations, (list, tuple)):
            raise TypeError(
                f"citations must be a list or tuple of CitationReference, got {type(self.citations).__name__}"
            )
        for item in self.citations:
            if not isinstance(item, CitationReference):
                raise TypeError(
                    f"Each item in citations must be a CitationReference, got {type(item).__name__}"
                )
        object.__setattr__(self, "citations", tuple(self.citations))

    @property
    def is_verified(self) -> bool:
        """Returns True if the claim is supported by evidence."""
        return self.status == VerificationStatus.SUPPORTED

    @property
    def is_contradicted(self) -> bool:
        """Returns True if the claim is contradicted by evidence."""
        return self.status == VerificationStatus.CONTRADICTED

    @property
    def is_unverified(self) -> bool:
        """Returns True if evidence is insufficient to verify the claim."""
        return self.status in (
            VerificationStatus.UNVERIFIED,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.UNVERIFIABLE,
        )

    @property
    def is_insufficient_evidence(self) -> bool:
        """Returns True if available evidence does not establish whether the claim is true."""
        return self.status in (
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.UNVERIFIED,
        )

    @property
    def is_unverifiable(self) -> bool:
        """Returns True if verification could not be completed due to invalid input or limitations."""
        return self.status == VerificationStatus.UNVERIFIABLE

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the verification outcome to a dictionary."""
        return {
            "claim": self.claim,
            "status": self.status.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "explanation": self.explanation,
            "contradiction_details": self.contradiction_details,
            "is_verified": self.is_verified,
            "is_contradicted": self.is_contradicted,
            "is_unverified": self.is_unverified,
            "is_insufficient_evidence": self.is_insufficient_evidence,
            "is_unverifiable": self.is_unverifiable,
        }


# Aliases for convenience
Citation = CitationReference
VerificationResult = VerificationOutcome

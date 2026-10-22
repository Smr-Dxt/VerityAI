"""
VerityAI - Deterministic Citation Validation Module
Phase 4: Agentic Verification and Citations (Chunk 2)

This module provides deterministic citation identity validation. It verifies that
citation references point strictly and faithfully to genuine evidence passages
present in the current operation's supplied evidence pool.

KEY ARCHITECTURAL GUARANTEES:
1. Grounded Provenance: Citations are never valid merely because they have the right
   syntactic shape; they must resolve to a genuine evidence record in the active pool.
2. Separation of Concerns: Identity validation verifies provenance (doc_id, chunk_id,
   filename, page_number), strictly separate from semantic claim verification.
3. Strict PDF & TXT Support: PDF citations must match exact 1-indexed source pages;
   TXT citations must not invent non-existent page numbers.
4. Privacy: Error messages and reports log only structural metadata (doc_id, chunk_id,
   page_number) and never leak private document text contents.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from verityai.chunking import DocumentChunk
from verityai.evidence import CitationReference, EvidencePassage
from verityai.retrieval_hybrid import HybridRetrievalResult
from verityai.retrieval_semantic import RetrievalResult


# ==============================================================================
# Citation Error Codes
# ==============================================================================


class CitationErrorCode(str, Enum):
    """Machine-readable error codes for citation validation failures."""

    EMPTY_EVIDENCE = "EMPTY_EVIDENCE"
    MALFORMED_CITATION = "MALFORMED_CITATION"
    MISSING_FIELDS = "MISSING_FIELDS"
    UNKNOWN_DOCUMENT = "UNKNOWN_DOCUMENT"
    UNKNOWN_CHUNK = "UNKNOWN_CHUNK"
    FILENAME_MISMATCH = "FILENAME_MISMATCH"
    MISSING_PAGE_NUMBER = "MISSING_PAGE_NUMBER"
    PAGE_NUMBER_MISMATCH = "PAGE_NUMBER_MISMATCH"
    UNEXPECTED_PAGE_NUMBER = "UNEXPECTED_PAGE_NUMBER"
    DUPLICATE_CITATION = "DUPLICATE_CITATION"

    def __str__(self) -> str:
        return self.value


# ==============================================================================
# Validation Result Models
# ==============================================================================


@dataclass(frozen=True)
class CitationValidationResult:
    """Outcome of validating an individual citation reference against evidence.

    Attributes:
        citation: The CitationReference evaluated (or None if input was malformed).
        is_valid: True if the citation genuinely matches a valid evidence record.
        error_code: CitationErrorCode identifying the failure category, or None if valid.
        reason: Plain-text explanation of failure (no private document text), or None if valid.
        matched_evidence: The genuine EvidencePassage matched if valid, or None.
        is_duplicate: True if this citation appeared more than once in the collection.
    """

    citation: Optional[CitationReference]
    is_valid: bool
    error_code: Optional[CitationErrorCode] = None
    reason: Optional[str] = None
    matched_evidence: Optional[EvidencePassage] = None
    is_duplicate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the validation result to a dictionary."""
        return {
            "citation": self.citation.to_dict() if self.citation else None,
            "is_valid": self.is_valid,
            "error_code": str(self.error_code) if self.error_code else None,
            "reason": self.reason,
            "matched_chunk_id": self.matched_evidence.chunk_id if self.matched_evidence else None,
            "is_duplicate": self.is_duplicate,
        }


@dataclass(frozen=True)
class CitationValidationReport:
    """Aggregated validation report for a collection of citations.

    Attributes:
        results: Tuple of individual CitationValidationResult items.
    """

    results: Tuple[CitationValidationResult, ...] = field(default_factory=tuple)

    @property
    def is_all_valid(self) -> bool:
        """Returns True if at least one citation was checked and all citations are valid."""
        return len(self.results) > 0 and all(r.is_valid for r in self.results)

    @property
    def total_citations(self) -> int:
        """Returns total number of citations evaluated."""
        return len(self.results)

    @property
    def valid_count(self) -> int:
        """Returns count of valid citations."""
        return sum(1 for r in self.results if r.is_valid)

    @property
    def invalid_count(self) -> int:
        """Returns count of invalid citations."""
        return sum(1 for r in self.results if not r.is_valid)

    @property
    def duplicate_count(self) -> int:
        """Returns count of duplicate citations."""
        return sum(1 for r in self.results if r.is_duplicate)

    @property
    def valid_citations(self) -> List[CitationReference]:
        """Returns list of valid CitationReference items."""
        return [r.citation for r in self.results if r.is_valid and r.citation is not None]

    @property
    def invalid_citations(self) -> List[Tuple[Optional[CitationReference], str]]:
        """Returns list of (citation, reason) pairs for invalid citations."""
        return [
            (r.citation, r.reason or "Unknown validation error")
            for r in self.results
            if not r.is_valid
        ]

    def summary(self) -> str:
        """Returns a concise human-readable validation summary."""
        if self.total_citations == 0:
            return "No citations to validate."
        if self.is_all_valid:
            return f"All {self.valid_count} citation(s) are valid and grounded in evidence."
        return (
            f"Citation validation failed: {self.valid_count}/{self.total_citations} valid, "
            f"{self.invalid_count} invalid ({self.duplicate_count} duplicate(s))."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report to dictionary."""
        return {
            "total_citations": self.total_citations,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "duplicate_count": self.duplicate_count,
            "is_all_valid": self.is_all_valid,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }


# ==============================================================================
# Citation Validator Engine
# ==============================================================================


class CitationValidator:
    """Validates that citations refer strictly and deterministically to genuine evidence."""

    def __init__(
        self,
        evidence_pool: Optional[
            Sequence[Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]]
        ] = None,
        allow_duplicates: bool = False,
    ) -> None:
        """Initializes the CitationValidator with an optional evidence pool.

        Args:
            evidence_pool: Collection of evidence passages, document chunks, or retrieval results.
            allow_duplicates: If True, duplicate citations are flagged as duplicates but marked valid.
                             If False (default), duplicate citations are marked invalid.
        """
        self.allow_duplicates = allow_duplicates
        self._index: Dict[Tuple[str, str], EvidencePassage] = {}
        self._known_doc_ids: Set[str] = set()

        if evidence_pool is not None:
            self.load_evidence(evidence_pool)

    @property
    def evidence_count(self) -> int:
        """Returns the number of indexed evidence passages."""
        return len(self._index)

    def load_evidence(
        self,
        evidence_pool: Sequence[
            Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]
        ],
    ) -> None:
        """Loads and indexes evidence items by (doc_id, chunk_id).

        Args:
            evidence_pool: Sequence of evidence objects to index.

        Raises:
            TypeError: If an item in evidence_pool is of an unsupported type.
        """
        self._index.clear()
        self._known_doc_ids.clear()

        for item in evidence_pool:
            passage = self._normalize_evidence(item)
            key = (passage.doc_id, passage.chunk_id)
            self._index[key] = passage
            self._known_doc_ids.add(passage.doc_id)

    @staticmethod
    def _normalize_evidence(
        item: Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult],
    ) -> EvidencePassage:
        """Normalizes any supported evidence representation into an EvidencePassage."""
        if isinstance(item, EvidencePassage):
            return item
        if isinstance(item, DocumentChunk):
            return EvidencePassage.from_chunk(item)
        if isinstance(item, RetrievalResult):
            return EvidencePassage.from_retrieval_result(item)
        if isinstance(item, HybridRetrievalResult):
            return EvidencePassage.from_hybrid_result(item)
        raise TypeError(
            f"Unsupported evidence type: {type(item).__name__}. "
            f"Must be EvidencePassage, DocumentChunk, RetrievalResult, or HybridRetrievalResult."
        )

    def validate_citation(
        self,
        citation: Any,
        seen_ids: Optional[Set[str]] = None,
        seen_targets: Optional[Set[Tuple[str, str, Optional[int]]]] = None,
    ) -> CitationValidationResult:
        """Validates a single citation against the loaded evidence pool.

        Args:
            citation: The citation reference to validate.
            seen_ids: Optional set of already-seen citation_ids for duplicate tracking.
            seen_targets: Optional set of already-seen (doc_id, chunk_id, page_number) tuples.

        Returns:
            CitationValidationResult with validity boolean, matched passage, and error details.
        """
        # 1. Guard against empty evidence collection
        if len(self._index) == 0:
            return CitationValidationResult(
                citation=citation if isinstance(citation, CitationReference) else None,
                is_valid=False,
                error_code=CitationErrorCode.EMPTY_EVIDENCE,
                reason="The supplied evidence collection is empty; cannot validate citation.",
            )

        # 2. Guard against non-CitationReference or None inputs
        if citation is None or not isinstance(citation, CitationReference):
            return CitationValidationResult(
                citation=None,
                is_valid=False,
                error_code=CitationErrorCode.MALFORMED_CITATION,
                reason=f"Malformed citation input: expected CitationReference, got {type(citation).__name__}.",
            )

        # 3. Check for empty string fields
        if (
            not citation.citation_id.strip()
            or not citation.doc_id.strip()
            or not citation.chunk_id.strip()
            or not citation.filename.strip()
        ):
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                error_code=CitationErrorCode.MISSING_FIELDS,
                reason="Citation reference contains empty or missing required identifier fields.",
            )

        # 4. Check for unknown document ID
        if citation.doc_id not in self._known_doc_ids:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                error_code=CitationErrorCode.UNKNOWN_DOCUMENT,
                reason=f"Document ID '{citation.doc_id}' not found in supplied evidence collection.",
            )

        # 5. Check for unknown chunk ID within document
        lookup_key = (citation.doc_id, citation.chunk_id)
        if lookup_key not in self._index:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                error_code=CitationErrorCode.UNKNOWN_CHUNK,
                reason=f"Chunk ID '{citation.chunk_id}' not found in document '{citation.doc_id}'.",
            )

        matched_evidence = self._index[lookup_key]

        # 6. Check filename match
        if citation.filename != matched_evidence.filename:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                error_code=CitationErrorCode.FILENAME_MISMATCH,
                reason=(
                    f"Citation filename '{citation.filename}' does not match "
                    f"source evidence filename '{matched_evidence.filename}'."
                ),
                matched_evidence=matched_evidence,
            )

        # 7. Check page number consistency
        is_pdf_source = (
            matched_evidence.doc_type == "pdf"
            or matched_evidence.page_number is not None
        )

        if is_pdf_source:
            if citation.page_number is None:
                return CitationValidationResult(
                    citation=citation,
                    is_valid=False,
                    error_code=CitationErrorCode.MISSING_PAGE_NUMBER,
                    reason=(
                        f"Citation for PDF document '{citation.filename}' is missing page number "
                        f"(source recorded page {matched_evidence.page_number})."
                    ),
                    matched_evidence=matched_evidence,
                )
            if citation.page_number != matched_evidence.page_number:
                return CitationValidationResult(
                    citation=citation,
                    is_valid=False,
                    error_code=CitationErrorCode.PAGE_NUMBER_MISMATCH,
                    reason=(
                        f"Cited page number {citation.page_number} does not match "
                        f"source page {matched_evidence.page_number} for '{citation.filename}'."
                    ),
                    matched_evidence=matched_evidence,
                )
        else:
            # Plain text source (page_number is None)
            if citation.page_number is not None:
                return CitationValidationResult(
                    citation=citation,
                    is_valid=False,
                    error_code=CitationErrorCode.UNEXPECTED_PAGE_NUMBER,
                    reason=(
                        f"Citation specifies page {citation.page_number} for plain text "
                        f"document '{citation.filename}', which has no page numbers."
                    ),
                    matched_evidence=matched_evidence,
                )

        # 8. Check duplicate tracking
        target_tuple = (citation.doc_id, citation.chunk_id, citation.page_number)
        is_duplicate = False

        if seen_ids is not None and seen_targets is not None:
            if citation.citation_id in seen_ids or target_tuple in seen_targets:
                is_duplicate = True
            seen_ids.add(citation.citation_id)
            seen_targets.add(target_tuple)

        if is_duplicate and not self.allow_duplicates:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                error_code=CitationErrorCode.DUPLICATE_CITATION,
                reason=(
                    f"Duplicate citation: citation ID '{citation.citation_id}' referencing "
                    f"chunk '{citation.chunk_id}' was already cited."
                ),
                matched_evidence=matched_evidence,
                is_duplicate=True,
            )

        # 9. All identity checks passed
        return CitationValidationResult(
            citation=citation,
            is_valid=True,
            error_code=None,
            reason=None,
            matched_evidence=matched_evidence,
            is_duplicate=is_duplicate,
        )

    def validate_citations(
        self,
        citations: Sequence[Any],
    ) -> CitationValidationReport:
        """Validates an entire sequence of citations against the evidence pool.

        Args:
            citations: Sequence of citation objects to evaluate.

        Returns:
            CitationValidationReport containing individual and aggregated results.
        """
        results: List[CitationValidationResult] = []
        seen_ids: Set[str] = set()
        seen_targets: Set[Tuple[str, str, Optional[int]]] = set()

        for cite in citations:
            res = self.validate_citation(
                citation=cite,
                seen_ids=seen_ids,
                seen_targets=seen_targets,
            )
            results.append(res)

        return CitationValidationReport(results=tuple(results))


# ==============================================================================
# Convenience Functions
# ==============================================================================


def validate_citations(
    citations: Sequence[Any],
    evidence_pool: Sequence[
        Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]
    ],
    allow_duplicates: bool = False,
) -> CitationValidationReport:
    """Validates a collection of citations against a supplied evidence pool.

    Args:
        citations: Sequence of citations to validate.
        evidence_pool: Sequence of evidence objects acting as the ground-truth pool.
        allow_duplicates: If True, duplicate citations are flagged as duplicate but marked valid.

    Returns:
        CitationValidationReport with full validation metrics and itemized results.
    """
    validator = CitationValidator(
        evidence_pool=evidence_pool,
        allow_duplicates=allow_duplicates,
    )
    return validator.validate_citations(citations)


def validate_citation(
    citation: Any,
    evidence_pool: Sequence[
        Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]
    ],
) -> CitationValidationResult:
    """Validates a single citation against a supplied evidence pool.

    Args:
        citation: Single citation to validate.
        evidence_pool: Sequence of evidence objects acting as the ground-truth pool.

    Returns:
        CitationValidationResult.
    """
    validator = CitationValidator(evidence_pool=evidence_pool)
    return validator.validate_citation(citation)

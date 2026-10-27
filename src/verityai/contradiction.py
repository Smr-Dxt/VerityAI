"""
VerityAI - Cross-Document Contradiction Detection Module
Phase 4: Agentic Verification and Citations (Chunk 4)

This module provides deterministic cross-document contradiction detection
between evidence passages retrieved from different source documents.

KEY ARCHITECTURAL GUARANTEES:
1. Honest Context Sensitivity: Differences in numbers or words are NOT
   automatically contradictions. Differing dates, versions, units, or measurement
   contexts represent distinct operational conditions (CONTEXT_MISMATCH).
2. Explicit Categorization: Distinguishes between:
   - POTENTIAL_CONTRADICTION: Genuine conflict on identical predicate & context.
   - NO_CONTRADICTION_DETECTED: Mutually compatible or context-differentiated assertions.
   - INCONCLUSIVE: Ambiguous, partial overlap, or unverified claims where the
     system honestly declines to guess.
3. Provenance Preservation: Retains full document IDs, filenames, chunk IDs,
   and 1-indexed PDF page numbers on both sides of any comparison.
4. Determinism & Duplicate Prevention: Symmetrically deduplicates pair evaluations,
   ignores self-comparisons, and enforces canonical sorting for stable results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from verityai.chunking import DocumentChunk
from verityai.citation_validator import CitationValidator
from verityai.evidence import EvidencePassage
from verityai.retrieval_hybrid import HybridRetrievalResult
from verityai.retrieval_semantic import RetrievalResult
from verityai.verifier import (
    _ANTONYMS,
    normalize_statement,
    split_into_sentences,
    tokenize_words,
)


# ==============================================================================
# Enumerations
# ==============================================================================


class ContradictionStatus(str, Enum):
    """Verdict status for cross-document conflict evaluation."""

    POTENTIAL_CONTRADICTION = "potential_contradiction"
    NO_CONTRADICTION_DETECTED = "no_contradiction_detected"
    INCONCLUSIVE = "inconclusive"

    def __str__(self) -> str:
        return self.value


class ContradictionReasonCode(str, Enum):
    """Machine-readable reason code describing the nature of a comparison outcome."""

    # Contradiction reasons
    POLARITY_CONFLICT = "POLARITY_CONFLICT"
    NEGATION_CONFLICT = "NEGATION_CONFLICT"
    NUMERIC_CONFLICT = "NUMERIC_CONFLICT"
    TEMPORAL_VALUE_CONFLICT = "TEMPORAL_VALUE_CONFLICT"

    # Non-contradiction reasons
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    COMPATIBLE_STATEMENTS = "COMPATIBLE_STATEMENTS"
    IDENTICAL_PASSAGES = "IDENTICAL_PASSAGES"

    # Inconclusive reasons
    AMBIGUOUS_EVIDENCE = "AMBIGUOUS_EVIDENCE"
    INVALID_INPUT = "INVALID_INPUT"

    def __str__(self) -> str:
        return self.value


# ==============================================================================
# Domain Lexicons for Context & Temporal Detection
# ==============================================================================

_MONTHS: Set[str] = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

_DAYS: Set[str] = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}

_COMMON_UNITS: Set[str] = {
    "meter", "meters", "m", "feet", "foot", "ft", "inch", "inches", "in",
    "mile", "miles", "km", "kilometer", "kilometers",
    "kg", "kilogram", "kilograms", "lb", "lbs", "pound", "pounds",
    "byte", "bytes", "kb", "mb", "gb", "tb", "pb",
    "second", "seconds", "sec", "secs", "s",
    "minute", "minutes", "min", "mins",
    "hour", "hours", "hr", "hrs", "h",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "percent", "%", "usd", "eur", "gbp", "dollars",
}


# ==============================================================================
# Result Models
# ==============================================================================


@dataclass(frozen=True)
class CrossDocumentConflict:
    """The structured comparison outcome between two evidence passages.

    Attributes:
        passage_a: The first EvidencePassage.
        passage_b: The second EvidencePassage.
        status: ContradictionStatus verdict.
        reason_code: ContradictionReasonCode specifying the outcome category.
        statement_a: The specific statement from passage A involved in the comparison.
        statement_b: The specific statement from passage B involved in the comparison.
        explanation: Clear human-readable explanation of the verdict.
        confidence: Confidence score between 0.0 and 1.0.
    """

    passage_a: EvidencePassage
    passage_b: EvidencePassage
    status: ContradictionStatus
    reason_code: ContradictionReasonCode
    statement_a: Optional[str] = None
    statement_b: Optional[str] = None
    explanation: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        """Validates outcome attributes upon initialization."""
        if not isinstance(self.passage_a, EvidencePassage):
            raise TypeError(f"passage_a must be an EvidencePassage, got {type(self.passage_a).__name__}")
        if not isinstance(self.passage_b, EvidencePassage):
            raise TypeError(f"passage_b must be an EvidencePassage, got {type(self.passage_b).__name__}")

        if isinstance(self.status, str):
            try:
                object.__setattr__(self, "status", ContradictionStatus(self.status.lower()))
            except ValueError:
                valid_statuses = [s.value for s in ContradictionStatus]
                raise ValueError(f"Invalid status '{self.status}'. Must be one of: {valid_statuses}")
        elif not isinstance(self.status, ContradictionStatus):
            raise TypeError(f"status must be a ContradictionStatus or str, got {type(self.status).__name__}")

        if isinstance(self.reason_code, str):
            try:
                object.__setattr__(self, "reason_code", ContradictionReasonCode(self.reason_code.upper()))
            except ValueError:
                valid_codes = [c.value for c in ContradictionReasonCode]
                raise ValueError(f"Invalid reason_code '{self.reason_code}'. Must be one of: {valid_codes}")
        elif not isinstance(self.reason_code, ContradictionReasonCode):
            raise TypeError(f"reason_code must be a ContradictionReasonCode, got {type(self.reason_code).__name__}")

        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise TypeError(f"confidence must be numeric, got {type(self.confidence).__name__}")
            conf_val = float(self.confidence)
            if not (0.0 <= conf_val <= 1.0):
                raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    @property
    def is_contradiction(self) -> bool:
        """Returns True if a potential contradiction was identified."""
        return self.status == ContradictionStatus.POTENTIAL_CONTRADICTION

    @property
    def is_inconclusive(self) -> bool:
        """Returns True if evidence was inconclusive or ambiguous."""
        return self.status == ContradictionStatus.INCONCLUSIVE

    @property
    def is_compatible(self) -> bool:
        """Returns True if statements are mutually compatible or context-differentiated."""
        return self.status == ContradictionStatus.NO_CONTRADICTION_DETECTED

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the conflict outcome to a dictionary."""
        return {
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "is_contradiction": self.is_contradiction,
            "is_inconclusive": self.is_inconclusive,
            "is_compatible": self.is_compatible,
            "statement_a": self.statement_a,
            "statement_b": self.statement_b,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "source_a": {
                "doc_id": self.passage_a.doc_id,
                "filename": self.passage_a.filename,
                "chunk_id": self.passage_a.chunk_id,
                "page_number": self.passage_a.page_number,
                "text": self.passage_a.text,
            },
            "source_b": {
                "doc_id": self.passage_b.doc_id,
                "filename": self.passage_b.filename,
                "chunk_id": self.passage_b.chunk_id,
                "page_number": self.passage_b.page_number,
                "text": self.passage_b.text,
            },
        }


@dataclass(frozen=True)
class ContradictionReport:
    """Aggregated cross-document contradiction report over an evidence pool.

    Attributes:
        conflicts: Tuple of CrossDocumentConflict items flagged as POTENTIAL_CONTRADICTION.
        inconclusive: Tuple of CrossDocumentConflict items flagged as INCONCLUSIVE.
        total_pairs_evaluated: Total count of distinct passage pairs evaluated.
        status: Overall collection verdict.
        summary: Concise human-readable summary.
    """

    conflicts: Tuple[CrossDocumentConflict, ...] = field(default_factory=tuple)
    inconclusive: Tuple[CrossDocumentConflict, ...] = field(default_factory=tuple)
    total_pairs_evaluated: int = 0
    status: ContradictionStatus = ContradictionStatus.NO_CONTRADICTION_DETECTED
    summary: str = ""

    @property
    def has_contradictions(self) -> bool:
        """Returns True if at least one potential contradiction was detected."""
        return len(self.conflicts) > 0

    @property
    def conflict_count(self) -> int:
        """Returns the number of potential contradictions detected."""
        return len(self.conflicts)

    @property
    def inconclusive_count(self) -> int:
        """Returns the number of inconclusive comparisons."""
        return len(self.inconclusive)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the report to a dictionary."""
        return {
            "status": self.status.value,
            "has_contradictions": self.has_contradictions,
            "conflict_count": self.conflict_count,
            "inconclusive_count": self.inconclusive_count,
            "total_pairs_evaluated": self.total_pairs_evaluated,
            "summary": self.summary,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "inconclusive": [inc.to_dict() for inc in self.inconclusive],
        }


# ==============================================================================
# Abstract Contradiction Detector Interface
# ==============================================================================


class BaseContradictionDetector(ABC):
    """Abstract base class for cross-document contradiction detectors."""

    @abstractmethod
    def compare_passages(
        self,
        passage_a: Any,
        passage_b: Any,
    ) -> CrossDocumentConflict:
        """Compares two evidence passages for mutual contradiction."""
        pass

    @abstractmethod
    def detect_contradictions(
        self,
        evidence: Optional[
            Sequence[Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]]
        ],
    ) -> ContradictionReport:
        """Scans an entire evidence pool for cross-document contradictions."""
        pass


# ==============================================================================
# Deterministic Contradiction Detector Implementation
# ==============================================================================


class DeterministicContradictionDetector(BaseContradictionDetector):
    """A conservative, deterministic rule-based cross-document contradiction detector.

    CAPABILITIES:
    - Polarity / Antonym conflicts on matching predicates under identical context.
    - Negation conflicts (affirmation vs negation on identical propositions).
    - Numerical discrepancies on identical measurements and context.
    - Temporal value conflicts (e.g. conflicting effective months without date qualification).
    - Context-awareness: Recognizes version differences, explicit temporal dates,
      and unit variations as CONTEXT_MISMATCH (non-contradictions).
    - Ambiguous / unverified statements are conservatively marked INCONCLUSIVE.
    - Deterministic canonical pair ordering prevents duplicate comparisons.

    DOCUMENTED LIMITATIONS:
    - Does NOT perform arbitrary natural language inference or deep semantic reasoning.
    - Complex paraphrasing and implicit contextual entailment will be classified
      as INCONCLUSIVE until neural NLI / LLM components are introduced in future chunks.
    """

    def __init__(
        self,
        cross_document_only: bool = True,
        include_inconclusive_in_report: bool = True,
    ) -> None:
        """Initializes the deterministic contradiction detector.

        Args:
            cross_document_only: If True, only compares passages originating from different
                                 documents (doc_id_a != doc_id_b).
            include_inconclusive_in_report: If True, tracks inconclusive pairs in report.
        """
        self.cross_document_only = cross_document_only
        self.include_inconclusive_in_report = include_inconclusive_in_report

    def compare_passages(
        self,
        passage_a: Any,
        passage_b: Any,
    ) -> CrossDocumentConflict:
        """Compares two passages and determines whether they contradict, differ in context, or are compatible."""
        # 1. Validate inputs and metadata
        p_a, err_a = self._safe_normalize(passage_a)
        p_b, err_b = self._safe_normalize(passage_b)

        if err_a or err_b or p_a is None or p_b is None:
            dummy_a = p_a or EvidencePassage(text="invalid", doc_id="unknown", filename="unknown.txt", chunk_id="unknown")
            dummy_b = p_b or EvidencePassage(text="invalid", doc_id="unknown", filename="unknown.txt", chunk_id="unknown")
            return CrossDocumentConflict(
                passage_a=dummy_a,
                passage_b=dummy_b,
                status=ContradictionStatus.INCONCLUSIVE,
                reason_code=ContradictionReasonCode.INVALID_INPUT,
                explanation=f"Invalid or missing source metadata: {err_a or err_b}",
                confidence=0.0,
            )

        # 2. Check identical passages
        if p_a.doc_id == p_b.doc_id and p_a.chunk_id == p_b.chunk_id:
            return CrossDocumentConflict(
                passage_a=p_a,
                passage_b=p_b,
                status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                reason_code=ContradictionReasonCode.IDENTICAL_PASSAGES,
                explanation="Identical passage evaluated against itself; no cross-document conflict.",
                confidence=1.0,
            )

        if normalize_statement(p_a.text) == normalize_statement(p_b.text):
            return CrossDocumentConflict(
                passage_a=p_a,
                passage_b=p_b,
                status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                reason_code=ContradictionReasonCode.IDENTICAL_PASSAGES,
                explanation="Passages contain identical text content; no cross-document conflict.",
                confidence=1.0,
            )

        # 3. Sentence-level cross-comparison
        sentences_a = split_into_sentences(p_a.text)
        sentences_b = split_into_sentences(p_b.text)

        best_conflict: Optional[CrossDocumentConflict] = None
        has_context_mismatch = False
        context_mismatch_explanation: Optional[str] = None
        has_ambiguity = False

        for sa in sentences_a:
            for sb in sentences_b:
                eval_res = self._evaluate_sentence_pair(sa, sb, p_a, p_b)
                if eval_res.status == ContradictionStatus.POTENTIAL_CONTRADICTION:
                    # Return immediately on first verified direct conflict
                    return eval_res
                elif eval_res.reason_code == ContradictionReasonCode.CONTEXT_MISMATCH:
                    has_context_mismatch = True
                    context_mismatch_explanation = eval_res.explanation
                elif eval_res.status == ContradictionStatus.INCONCLUSIVE:
                    has_ambiguity = True

        # If a verified context mismatch explains differing assertions
        if has_context_mismatch:
            return CrossDocumentConflict(
                passage_a=p_a,
                passage_b=p_b,
                status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                reason_code=ContradictionReasonCode.CONTEXT_MISMATCH,
                explanation=context_mismatch_explanation or "Statements differ in version, date, or operational context.",
                confidence=1.0,
            )

        # If there is semantic ambiguity / partial overlap
        if has_ambiguity:
            return CrossDocumentConflict(
                passage_a=p_a,
                passage_b=p_b,
                status=ContradictionStatus.INCONCLUSIVE,
                reason_code=ContradictionReasonCode.AMBIGUOUS_EVIDENCE,
                explanation="Passages share topical overlap but evidence is inconclusive to confirm conflict.",
                confidence=0.0,
            )

        # Non-conflicting, mutually compatible statements
        return CrossDocumentConflict(
            passage_a=p_a,
            passage_b=p_b,
            status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
            reason_code=ContradictionReasonCode.COMPATIBLE_STATEMENTS,
            explanation="No conflicting assertions found between passages; statements appear compatible.",
            confidence=1.0,
        )

    def detect_contradictions(
        self,
        evidence: Optional[
            Sequence[Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]]
        ],
    ) -> ContradictionReport:
        """Scans the provided evidence pool for cross-document contradictions."""
        if evidence is None or len(evidence) == 0:
            return ContradictionReport(
                conflicts=(),
                inconclusive=(),
                total_pairs_evaluated=0,
                status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                summary="No evidence passages provided to evaluate.",
            )

        # Normalize and validate all passages
        normalized_passages: List[EvidencePassage] = []
        for item in evidence:
            passage, err = self._safe_normalize(item)
            if err or passage is None:
                # Malformed item; return inconclusive report immediately
                dummy_err = EvidencePassage(
                    text="invalid", doc_id="invalid", filename="invalid.txt", chunk_id="invalid"
                )
                err_conflict = CrossDocumentConflict(
                    passage_a=dummy_err,
                    passage_b=dummy_err,
                    status=ContradictionStatus.INCONCLUSIVE,
                    reason_code=ContradictionReasonCode.INVALID_INPUT,
                    explanation=f"Evidence collection contains invalid item: {err}",
                    confidence=0.0,
                )
                return ContradictionReport(
                    conflicts=(),
                    inconclusive=(err_conflict,),
                    total_pairs_evaluated=0,
                    status=ContradictionStatus.INCONCLUSIVE,
                    summary=f"Contradiction detection aborted due to malformed input: {err}",
                )
            normalized_passages.append(passage)

        # Deduplicate identical passages within the collection by (doc_id, chunk_id)
        seen_keys: Set[Tuple[str, str]] = set()
        deduped: List[EvidencePassage] = []
        for p in normalized_passages:
            key = (p.doc_id, p.chunk_id)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(p)

        # Sort canonically for strict determinism across repeated runs
        deduped.sort(key=lambda p: (p.doc_id, p.chunk_id, p.page_number or 0))

        conflicts: List[CrossDocumentConflict] = []
        inconclusive: List[CrossDocumentConflict] = []
        pairs_evaluated = 0

        # Symmetrically evaluate all unique pairs (i < j)
        n = len(deduped)
        for i in range(n):
            for j in range(i + 1, n):
                pa = deduped[i]
                pb = deduped[j]

                # If cross-document only, skip intra-document comparisons
                if self.cross_document_only and pa.doc_id == pb.doc_id:
                    continue

                pairs_evaluated += 1
                conflict = self.compare_passages(pa, pb)

                if conflict.status == ContradictionStatus.POTENTIAL_CONTRADICTION:
                    conflicts.append(conflict)
                elif conflict.status == ContradictionStatus.INCONCLUSIVE:
                    if self.include_inconclusive_in_report:
                        inconclusive.append(conflict)

        # Formulate overall report status and summary
        if conflicts:
            overall_status = ContradictionStatus.POTENTIAL_CONTRADICTION
            summary = (
                f"Detected {len(conflicts)} potential cross-document contradiction(s) "
                f"across {pairs_evaluated} pair(s) evaluated."
            )
        elif inconclusive:
            overall_status = ContradictionStatus.INCONCLUSIVE
            summary = (
                f"No direct contradictions verified; {len(inconclusive)} comparison(s) "
                f"yielded inconclusive evidence across {pairs_evaluated} pair(s) evaluated."
            )
        else:
            overall_status = ContradictionStatus.NO_CONTRADICTION_DETECTED
            summary = f"No contradictions detected across {pairs_evaluated} pair(s) evaluated."

        return ContradictionReport(
            conflicts=tuple(conflicts),
            inconclusive=tuple(inconclusive),
            total_pairs_evaluated=pairs_evaluated,
            status=overall_status,
            summary=summary,
        )

    # --------------------------------------------------------------------------
    # Internal Evaluation Helpers
    # --------------------------------------------------------------------------

    def _evaluate_sentence_pair(
        self,
        sa: str,
        sb: str,
        pa: EvidencePassage,
        pb: EvidencePassage,
    ) -> CrossDocumentConflict:
        """Evaluates a pair of normalized sentences for conflict or context mismatch."""
        tokens_a = tokenize_words(sa)
        tokens_b = tokenize_words(sb)

        # 1. Context Qualifier Checks (Version, Temporal, Units)
        # Check for explicit differing version context
        v_a = self._extract_version(sa)
        v_b = self._extract_version(sb)
        if v_a and v_b and v_a != v_b:
            return CrossDocumentConflict(
                passage_a=pa,
                passage_b=pb,
                status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                reason_code=ContradictionReasonCode.CONTEXT_MISMATCH,
                statement_a=sa,
                statement_b=sb,
                explanation=f"Statements apply to different software versions ('{v_a}' vs '{v_b}').",
                confidence=1.0,
            )

        # Check for explicit differing year/quarter temporal context
        t_a = self._extract_temporal_qualifier(sa)
        t_b = self._extract_temporal_qualifier(sb)
        if t_a and t_b and t_a != t_b:
            return CrossDocumentConflict(
                passage_a=pa,
                passage_b=pb,
                status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                reason_code=ContradictionReasonCode.CONTEXT_MISMATCH,
                statement_a=sa,
                statement_b=sb,
                explanation=f"Statements apply to different timeframes or dates ('{t_a}' vs '{t_b}').",
                confidence=1.0,
            )

        # Check for differing unit measurement context
        u_a = self._extract_units(tokens_a)
        u_b = self._extract_units(tokens_b)
        if u_a and u_b and u_a != u_b:
            return CrossDocumentConflict(
                passage_a=pa,
                passage_b=pb,
                status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                reason_code=ContradictionReasonCode.CONTEXT_MISMATCH,
                statement_a=sa,
                statement_b=sb,
                explanation=f"Statements express measurements in different units ({u_a} vs {u_b}).",
                confidence=1.0,
            )

        # 2. Contradiction: Direct Negation
        # A has 'not', B lacks 'not' on identical remaining proposition
        if "not" in tokens_a and "not" not in tokens_b:
            a_no_not = [t for t in tokens_a if t != "not"]
            if a_no_not == tokens_b:
                return CrossDocumentConflict(
                    passage_a=pa,
                    passage_b=pb,
                    status=ContradictionStatus.POTENTIAL_CONTRADICTION,
                    reason_code=ContradictionReasonCode.NEGATION_CONFLICT,
                    statement_a=sa,
                    statement_b=sb,
                    explanation=(
                        f"Direct negation: '{pa.filename}' negates statement in '{pb.filename}' "
                        f"under identical context."
                    ),
                    confidence=1.0,
                )

        # B has 'not', A lacks 'not' on identical remaining proposition
        if "not" in tokens_b and "not" not in tokens_a:
            b_no_not = [t for t in tokens_b if t != "not"]
            if b_no_not == tokens_a:
                return CrossDocumentConflict(
                    passage_a=pa,
                    passage_b=pb,
                    status=ContradictionStatus.POTENTIAL_CONTRADICTION,
                    reason_code=ContradictionReasonCode.NEGATION_CONFLICT,
                    statement_a=sa,
                    statement_b=sb,
                    explanation=(
                        f"Direct negation: '{pb.filename}' negates statement in '{pa.filename}' "
                        f"under identical context."
                    ),
                    confidence=1.0,
                )

        # 3. Contradiction: Antonym Polarity Swap
        # Identical frame except exactly one antonym token
        if len(tokens_a) == len(tokens_b) and len(tokens_a) >= 2:
            mismatches: List[Tuple[int, str, str]] = []
            for idx, (ta, tb) in enumerate(zip(tokens_a, tokens_b)):
                if ta != tb:
                    mismatches.append((idx, ta, tb))

            if len(mismatches) == 1:
                _, ta, tb = mismatches[0]
                if _ANTONYMS.get(ta) == tb:
                    return CrossDocumentConflict(
                        passage_a=pa,
                        passage_b=pb,
                        status=ContradictionStatus.POTENTIAL_CONTRADICTION,
                        reason_code=ContradictionReasonCode.POLARITY_CONFLICT,
                        statement_a=sa,
                        statement_b=sb,
                        explanation=(
                            f"Polarity conflict: '{pa.filename}' states '{ta}', but '{pb.filename}' "
                            f"states opposite '{tb}' for the same subject and conditions."
                        ),
                        confidence=1.0,
                    )

                # 4. Contradiction: Numeric Discrepancy on identical predicate
                if self._is_numeric(ta) and self._is_numeric(tb):
                    return CrossDocumentConflict(
                        passage_a=pa,
                        passage_b=pb,
                        status=ContradictionStatus.POTENTIAL_CONTRADICTION,
                        reason_code=ContradictionReasonCode.NUMERIC_CONFLICT,
                        statement_a=sa,
                        statement_b=sb,
                        explanation=(
                            f"Numerical discrepancy: '{pa.filename}' reports value '{ta}', while '{pb.filename}' "
                            f"reports '{tb}' for the same measurement."
                        ),
                        confidence=1.0,
                    )

                # 5. Contradiction: Temporal Value Conflict (conflicting months or days on identical predicate)
                if (ta in _MONTHS and tb in _MONTHS) or (ta in _DAYS and tb in _DAYS):
                    return CrossDocumentConflict(
                        passage_a=pa,
                        passage_b=pb,
                        status=ContradictionStatus.POTENTIAL_CONTRADICTION,
                        reason_code=ContradictionReasonCode.TEMPORAL_VALUE_CONFLICT,
                        statement_a=sa,
                        statement_b=sb,
                        explanation=(
                            f"Temporal value conflict: '{pa.filename}' specifies '{ta}', while '{pb.filename}' "
                            f"specifies '{tb}' for the same event or policy."
                        ),
                        confidence=1.0,
                    )

        # 6. Ambiguous / Topical Overlap Check
        # If sentences share significant topic words (>= 2 content words) or both contain numeric values
        # on shared subjects without satisfying direct contradiction rules
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "with", "by", "from", "as",
        }
        overlap = set(tokens_a).intersection(set(tokens_b)) - stopwords
        has_numbers = any(self._is_numeric(t) for t in tokens_a) and any(self._is_numeric(t) for t in tokens_b)
        if len(overlap) >= 2 or (len(overlap) >= 1 and has_numbers):
            return CrossDocumentConflict(
                passage_a=pa,
                passage_b=pb,
                status=ContradictionStatus.INCONCLUSIVE,
                reason_code=ContradictionReasonCode.AMBIGUOUS_EVIDENCE,
                statement_a=sa,
                statement_b=sb,
                explanation="Statements share overlapping terms but differ in assertion details.",
                confidence=0.0,
            )

        return CrossDocumentConflict(
            passage_a=pa,
            passage_b=pb,
            status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
            reason_code=ContradictionReasonCode.COMPATIBLE_STATEMENTS,
            statement_a=sa,
            statement_b=sb,
            explanation="Statements do not conflict.",
            confidence=1.0,
        )

    # --------------------------------------------------------------------------
    # Extraction & Parsing Utilities
    # --------------------------------------------------------------------------

    @staticmethod
    def _safe_normalize(item: Any) -> Tuple[Optional[EvidencePassage], Optional[str]]:
        """Safely normalizes any supported evidence item and validates metadata."""
        if item is None:
            return None, "Evidence item is None."

        try:
            passage = CitationValidator._normalize_evidence(item)
        except Exception as exc:
            return None, f"Unsupported or malformed evidence item: {exc}"

        if (
            not isinstance(passage.doc_id, str)
            or not passage.doc_id.strip()
            or not isinstance(passage.filename, str)
            or not passage.filename.strip()
            or not isinstance(passage.chunk_id, str)
            or not passage.chunk_id.strip()
            or not isinstance(passage.text, str)
            or not passage.text.strip()
        ):
            return None, "Evidence passage has missing or empty required metadata (doc_id, filename, chunk_id, or text)."

        return passage, None

    @staticmethod
    def _is_numeric(token: str) -> bool:
        """Returns True if token represents a valid integer or floating point number."""
        try:
            float(token)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_version(text: str) -> Optional[str]:
        """Extracts version identifier if present (e.g. 'v1.0', 'version 2.4', 'v3')."""
        match = re.search(r"\b(?:v(?:ersion)?\.?\s*(\d+(?:\.\d+)*)|\bv(\d+))\b", text, re.IGNORECASE)
        if match:
            return match.group(1) or match.group(2)
        return None

    @staticmethod
    def _extract_temporal_qualifier(text: str) -> Optional[str]:
        """Extracts year, quarter, or explicit temporal anchor (e.g. '2020', 'q3', '2024')."""
        match = re.search(r"\b(?:in|as of|effective|since|during|for)?\s*(19\d\d|20\d\d|q[1-4])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return None

    @staticmethod
    def _extract_units(tokens: List[str]) -> Set[str]:
        """Extracts recognized measurement unit tokens from a list of tokens."""
        return set(tokens).intersection(_COMMON_UNITS)


# ==============================================================================
# Convenience Functions
# ==============================================================================


def detect_cross_document_contradictions(
    evidence: Optional[
        Sequence[Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]]
    ],
    detector: Optional[BaseContradictionDetector] = None,
    cross_document_only: bool = True,
) -> ContradictionReport:
    """Scans evidence passages for cross-document contradictions using default or custom detector.

    Args:
        evidence: Sequence of evidence items to evaluate.
        detector: Optional custom BaseContradictionDetector instance.
        cross_document_only: If True, only compares passages from distinct documents.

    Returns:
        ContradictionReport detailing any conflicts, inconclusive cases, and overall status.
    """
    engine = detector if detector is not None else DeterministicContradictionDetector(
        cross_document_only=cross_document_only
    )
    return engine.detect_contradictions(evidence)

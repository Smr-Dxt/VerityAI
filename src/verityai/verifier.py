"""
VerityAI - Claim-Level Evidence Verification Module
Phase 4: Agentic Verification and Citations (Chunk 3)

This module provides a modular, typed claim verification architecture.
It defines an extensible BaseClaimVerifier interface and a deterministic
reference implementation (DeterministicClaimVerifier).

VERIFICATION OUTCOMES:
- SUPPORTED: The evidence directly substantiates and corroborates the claim.
- CONTRADICTED: The evidence directly conflicts with or refutes the claim.
- INSUFFICIENT_EVIDENCE: The available evidence is inconclusive, ambiguous,
  unrelated, or contains mixed signals. The verifier honestly declines to guess.
- UNVERIFIABLE: Verification cannot proceed due to invalid input (empty claim,
  malformed evidence, missing source metadata, or broken citation references).

HONEST VERIFICATION GUARANTEES:
1. Grounded Provenance: Citation references must resolve strictly through the
   existing CitationValidator.
2. No Keyword Guessing: Superficial word overlap between a claim and an evidence
   passage is NOT treated as support or contradiction. Ambiguous cases return
   INSUFFICIENT_EVIDENCE.
3. Separation of Concerns: Identity validation verifies provenance metadata;
   the verifier checks propositional alignment or conflict.
4. Extensible Architecture: Clear BaseClaimVerifier contract ready for future
   local NLI models and LLM-based verification components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from verityai.chunking import DocumentChunk
from verityai.citation_validator import CitationValidator, validate_citations
from verityai.evidence import (
    CitationReference,
    EvidencePassage,
    VerificationOutcome,
    VerificationStatus,
)
from verityai.retrieval_hybrid import HybridRetrievalResult
from verityai.retrieval_semantic import RetrievalResult


# ==============================================================================
# Contraction & Polarity Lexicons
# ==============================================================================

# Common English contractions normalized for deterministic matching
_CONTRACTIONS: Dict[str, str] = {
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bdoesn't\b": "does not",
    r"\bdon't\b": "do not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bcouldn't\b": "could not",
    r"\bshouldn't\b": "should not",
}

# Curated antonym and polarity pairs for direct contradiction detection
_ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("online", "offline"),
    ("enabled", "disabled"),
    ("active", "inactive"),
    ("true", "false"),
    ("success", "failure"),
    ("successful", "unsuccessful"),
    ("passed", "failed"),
    ("valid", "invalid"),
    ("allowed", "disallowed"),
    ("permitted", "prohibited"),
    ("permitted", "forbidden"),
    ("supported", "unsupported"),
    ("compatible", "incompatible"),
    ("available", "unavailable"),
    ("connected", "disconnected"),
    ("secure", "insecure"),
    ("public", "private"),
    ("open", "closed"),
    ("started", "stopped"),
    ("present", "absent"),
    ("installed", "uninstalled"),
    ("safe", "unsafe"),
    ("encrypted", "unencrypted"),
    ("compressed", "uncompressed"),
    ("authorized", "unauthorized"),
    ("authenticated", "unauthenticated"),
    ("resolved", "unresolved"),
    ("synchronous", "asynchronous"),
    ("symmetric", "asymmetric"),
    ("deterministic", "nondeterministic"),
    ("internal", "external"),
    ("local", "remote"),
    ("primary", "secondary"),
    ("temporary", "permanent"),
    ("manual", "automatic"),
]

_ANTONYMS: Dict[str, str] = {}
for _w1, _w2 in _ANTONYM_PAIRS:
    _ANTONYMS[_w1] = _w2
    _ANTONYMS[_w2] = _w1


# ==============================================================================
# Normalization Helper Functions
# ==============================================================================


def normalize_statement(text: str) -> str:
    """Normalizes statement text for deterministic propositional evaluation.

    Expands common contractions, trims leading/trailing punctuation and quotes,
    collapses whitespace, and lowercases text.
    """
    if not text:
        return ""

    lowered = text.lower()
    for pattern, replacement in _CONTRACTIONS.items():
        lowered = re.sub(pattern, replacement, lowered)

    # Trim leading and trailing punctuation and quotes
    cleaned = re.sub(r"^[\s\"'`.,;:!?()\[\]{}]+|[\s\"'`.,;:!?()\[\]{}]+$", "", lowered)
    # Collapse multiple whitespaces
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed


def split_into_sentences(text: str) -> List[str]:
    """Splits passage text into individual candidate sentence statements."""
    if not text or not text.strip():
        return []

    # Split by standard sentence delimiters (. ! ? \n) followed by whitespace or end
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences: List[str] = []
    for s in raw_sentences:
        norm = normalize_statement(s)
        if norm:
            sentences.append(norm)
    return sentences


def tokenize_words(text: str) -> List[str]:
    """Extracts alphanumeric word tokens from normalized text."""
    return re.findall(r"\b[\w.-]+\b", text)


# ==============================================================================
# Abstract Verifier Interface
# ==============================================================================


class BaseClaimVerifier(ABC):
    """Abstract base class for claim verification components.

    Defines the contract for verifying factual assertions against evidence.
    Implementations may be deterministic, neural (NLI), or agentic LLM-driven.
    """

    @abstractmethod
    def verify(
        self,
        claim: Any,
        evidence: Optional[
            Sequence[Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]]
        ],
        citations: Optional[Sequence[CitationReference]] = None,
    ) -> VerificationOutcome:
        """Verifies a claim statement against a collection of evidence passages.

        Args:
            claim: Factual claim statement to evaluate.
            evidence: Sequence of evidence items acting as the knowledge base.
            citations: Optional sequence of CitationReference items to validate.

        Returns:
            VerificationOutcome with status (SUPPORTED, CONTRADICTED,
            INSUFFICIENT_EVIDENCE, or UNVERIFIABLE), confidence, and citations.
        """
        pass


# ==============================================================================
# Deterministic Claim Verifier Implementation
# ==============================================================================


class DeterministicClaimVerifier(BaseClaimVerifier):
    """A deterministic, rule-based claim verification engine.

    CAPABILITIES:
    - Verbatim and canonical statement corroboration: Confidently marks claims
      SUPPORTED when an evidence passage contains an exact matching proposition
      without conflicting evidence.
    - Explicit contradiction detection: Confidently marks claims CONTRADICTED
      when an evidence passage contains a direct negation (e.g. insertion/omission
      of 'not'), an antonym polarity swap ('online' vs 'offline'), or a single
      differing numerical value on an otherwise identical predicate.
    - Mixed evidence handling: Identifies multi-passage conflicting evidence
      and returns INSUFFICIENT_EVIDENCE rather than guessing.
    - Strict citation validation: Requires supplied citations to resolve cleanly
      against genuine evidence passages via CitationValidator.

    DOCUMENTED LIMITATIONS:
    - Does NOT perform arbitrary natural language inference (NLI), paraphrase
      detection, or complex multi-hop deductive reasoning.
    - Superficial keyword overlap without propositional equivalence is honestly
      classified as INSUFFICIENT_EVIDENCE.
    """

    def __init__(
        self,
        allow_numeric_contradiction: bool = True,
        auto_generate_citations: bool = True,
    ) -> None:
        """Initializes the deterministic claim verifier.

        Args:
            allow_numeric_contradiction: If True, flags single numeric value mismatches
                                         on identical sentence frames as contradictions.
            auto_generate_citations: If True, automatically creates valid citations
                                     for supported or contradicted evidence passages
                                     when no explicit citations were supplied.
        """
        self.allow_numeric_contradiction = allow_numeric_contradiction
        self.auto_generate_citations = auto_generate_citations

    def verify(
        self,
        claim: Any,
        evidence: Optional[
            Sequence[Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]]
        ],
        citations: Optional[Sequence[CitationReference]] = None,
    ) -> VerificationOutcome:
        """Evaluates a claim against evidence using deterministic verification rules."""
        # 1. Validate Claim Input
        if claim is None or not isinstance(claim, str) or not claim.strip():
            claim_repr = str(claim) if claim is not None else ""
            return VerificationOutcome(
                claim=claim_repr,
                status=VerificationStatus.UNVERIFIABLE,
                confidence=0.0,
                explanation="Claim input is invalid: must be a non-empty string.",
            )

        clean_claim = claim.strip()
        norm_claim = normalize_statement(clean_claim)

        # 2. Validate Evidence Input Collection
        if evidence is None or len(evidence) == 0:
            return VerificationOutcome(
                claim=clean_claim,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                explanation="No evidence passages were provided to verify the claim.",
            )

        # 3. Normalize Evidence Items and Check Metadata
        normalized_evidence: List[EvidencePassage] = []
        for idx, item in enumerate(evidence):
            try:
                passage = CitationValidator._normalize_evidence(item)
            except Exception as e:
                return VerificationOutcome(
                    claim=clean_claim,
                    status=VerificationStatus.UNVERIFIABLE,
                    confidence=0.0,
                    explanation=f"Evidence collection contains invalid item at index {idx}: {e}",
                )

            # Enforce non-empty provenance metadata
            if (
                not passage.doc_id.strip()
                or not passage.filename.strip()
                or not passage.chunk_id.strip()
                or not passage.text.strip()
            ):
                return VerificationOutcome(
                    claim=clean_claim,
                    status=VerificationStatus.UNVERIFIABLE,
                    confidence=0.0,
                    explanation=(
                        f"Evidence passage at index {idx} has missing or empty required metadata "
                        f"(doc_id='{passage.doc_id}', filename='{passage.filename}', "
                        f"chunk_id='{passage.chunk_id}')."
                    ),
                )

            normalized_evidence.append(passage)

        # 4. Validate Supplied Citations (if any)
        validated_citations: List[CitationReference] = []
        if citations is not None:
            if not isinstance(citations, (list, tuple)):
                return VerificationOutcome(
                    claim=clean_claim,
                    status=VerificationStatus.UNVERIFIABLE,
                    confidence=0.0,
                    explanation=f"Citations must be a list or tuple of CitationReference, got {type(citations).__name__}.",
                )

            report = validate_citations(citations=citations, evidence_pool=normalized_evidence)
            if not report.is_all_valid:
                return VerificationOutcome(
                    claim=clean_claim,
                    status=VerificationStatus.UNVERIFIABLE,
                    confidence=0.0,
                    explanation=f"Citation reference validation failed: {report.summary()}",
                    citations=(),
                )
            validated_citations = report.valid_citations

        # 5. Evaluate Claim Against Evidence Passages
        supporting_passages: List[EvidencePassage] = []
        contradicting_passages: List[Tuple[EvidencePassage, str]] = []

        for passage in normalized_evidence:
            verdict, detail = self._evaluate_passage(norm_claim, passage)
            if verdict == VerificationStatus.SUPPORTED:
                supporting_passages.append(passage)
            elif verdict == VerificationStatus.CONTRADICTED:
                contradicting_passages.append((passage, detail or "Contradicting statement in evidence."))

        # 6. Synthesize Multi-Passage Findings

        # Case A: Mixed / Conflicting Evidence across passages
        if supporting_passages and contradicting_passages:
            conflict_summary = (
                f"Conflicting evidence detected across {len(supporting_passages)} supporting passage(s) "
                f"and {len(contradicting_passages)} contradicting passage(s). Evidence is inconclusive."
            )
            return VerificationOutcome(
                claim=clean_claim,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                evidence=normalized_evidence,
                citations=tuple(validated_citations),
                confidence=0.0,
                explanation=conflict_summary,
                contradiction_details=contradicting_passages[0][1],
            )

        # Case B: Supported by Evidence
        if supporting_passages:
            # Form citations: use user-validated citations if provided, or auto-generate
            out_citations: List[CitationReference] = list(validated_citations)
            if not out_citations and self.auto_generate_citations:
                for idx, p in enumerate(supporting_passages, 1):
                    out_citations.append(
                        CitationReference.from_evidence(p, citation_id=str(idx))
                    )

            best_score = max((p.score for p in supporting_passages if p.score is not None), default=1.0)
            conf = float(best_score) if best_score is not None and 0.0 <= best_score <= 1.0 else 1.0

            return VerificationOutcome(
                claim=clean_claim,
                status=VerificationStatus.SUPPORTED,
                evidence=tuple(supporting_passages),
                citations=tuple(out_citations),
                confidence=conf,
                explanation=f"Claim is directly supported by evidence in '{supporting_passages[0].filename}'.",
            )

        # Case C: Contradicted by Evidence
        if contradicting_passages:
            out_citations = list(validated_citations)
            if not out_citations and self.auto_generate_citations:
                for idx, (p, _) in enumerate(contradicting_passages, 1):
                    out_citations.append(
                        CitationReference.from_evidence(p, citation_id=str(idx))
                    )

            return VerificationOutcome(
                claim=clean_claim,
                status=VerificationStatus.CONTRADICTED,
                evidence=tuple(p for p, _ in contradicting_passages),
                citations=tuple(out_citations),
                confidence=1.0,
                explanation=f"Claim is contradicted by evidence in '{contradicting_passages[0][0].filename}'.",
                contradiction_details=contradicting_passages[0][1],
            )

        # Case D: Insufficient Evidence (Ambiguous, Keyword Overlap Only, or Unrelated)
        return VerificationOutcome(
            claim=clean_claim,
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            evidence=tuple(normalized_evidence),
            citations=tuple(validated_citations),
            confidence=0.0,
            explanation=(
                "Available evidence does not establish whether the claim is true. "
                "No direct propositional corroboration or explicit contradiction was found."
            ),
        )

    def _evaluate_passage(
        self, norm_claim: str, passage: EvidencePassage
    ) -> Tuple[Optional[VerificationStatus], Optional[str]]:
        """Evaluates a single passage against the normalized claim statement.

        Returns:
            Tuple of (VerificationStatus, optional_contradiction_detail).
        """
        sentences = split_into_sentences(passage.text)
        claim_tokens = tokenize_words(norm_claim)

        # 1. Exact Sentence Corroboration
        for s in sentences:
            if s == norm_claim:
                return VerificationStatus.SUPPORTED, None

            # Substring match if claim is a complete independent proposition within sentence
            # bounded by punctuation, clause boundaries, or start/end
            if len(claim_tokens) >= 3 and self._is_clause_match(norm_claim, s):
                return VerificationStatus.SUPPORTED, None

        # 2. Contradiction Detection
        for s in sentences:
            is_contra, detail = self._check_contradiction(norm_claim, claim_tokens, s)
            if is_contra:
                return VerificationStatus.CONTRADICTED, detail

        # 3. Inconclusive (Insufficient evidence)
        return VerificationStatus.INSUFFICIENT_EVIDENCE, None

    def _is_clause_match(self, norm_claim: str, norm_sentence: str) -> bool:
        """Checks if normalized claim appears as a full clause within a sentence."""
        escaped = re.escape(norm_claim)
        pattern = rf"(?:^|[,\s;:\"'(-]){escaped}(?:[,\s;:\"')-]|$)"
        return bool(re.search(pattern, norm_sentence))

    def _check_contradiction(
        self, norm_claim: str, claim_tokens: List[str], norm_sentence: str
    ) -> Tuple[bool, Optional[str]]:
        """Checks if a normalized evidence sentence directly contradicts the claim.

        Mechanisms:
        1. Negation:
           - Claim has 'not', sentence asserts positive clause.
           - Claim lacks 'not', sentence asserts negated clause.
           - Token difference where only 'not' is added or removed.
        2. Antonym Polarity:
           - Replacing an antonym in the claim yields a clause asserted in the sentence.
           - Single antonym mismatch on identical token lengths.
        3. Single Numerical Discrepancy:
           - Same sentence frame with differing numeric value.
        """
        sent_tokens = tokenize_words(norm_sentence)

        # 1. Direct Negation Checks
        # A. Claim has 'not', sentence asserts positive
        if "not" in claim_tokens:
            claim_without_not = " ".join(t for t in claim_tokens if t != "not")
            if self._is_clause_match(claim_without_not, norm_sentence):
                return True, (
                    f"Direct negation: Claim asserts negation '{norm_claim}', but evidence asserts '{norm_sentence}'."
                )

        # B. Claim lacks 'not', sentence asserts negation
        aux_verbs = [
            "is", "are", "was", "were", "has", "have", "had",
            "will", "would", "can", "could", "should", "does", "do", "did"
        ]
        for aux in aux_verbs:
            pattern = rf"\b{aux}\b"
            if re.search(pattern, norm_claim):
                negated_claim = re.sub(pattern, f"{aux} not", norm_claim, count=1)
                if self._is_clause_match(negated_claim, norm_sentence):
                    return True, (
                        f"Direct negation: Claim states '{norm_claim}', but evidence states '{norm_sentence}'."
                    )

        # C. Direct token negation difference
        if "not" not in claim_tokens and "not" in sent_tokens:
            sent_without_not = [t for t in sent_tokens if t != "not"]
            if sent_without_not == claim_tokens:
                return True, (
                    f"Direct negation: Claim states '{norm_claim}', but evidence states '{norm_sentence}'."
                )

        # 2. Antonym Polarity Checks
        # A. Clause-level antonym substitution check
        for i, token in enumerate(claim_tokens):
            antonym = _ANTONYMS.get(token)
            if antonym:
                ant_tokens = list(claim_tokens)
                ant_tokens[i] = antonym
                ant_claim = " ".join(ant_tokens)
                if self._is_clause_match(ant_claim, norm_sentence):
                    return True, (
                        f"Polarity contradiction: Claim states '{token}', but evidence states opposite '{antonym}' "
                        f"in statement '{norm_sentence}'."
                    )

        # B. Token-level antonym mismatch check
        if len(claim_tokens) == len(sent_tokens) and len(claim_tokens) >= 2:
            mismatches: List[Tuple[int, str, str]] = []
            for i, (tc, ts) in enumerate(zip(claim_tokens, sent_tokens)):
                if tc != ts:
                    mismatches.append((i, tc, ts))

            if len(mismatches) == 1:
                _, tc, ts = mismatches[0]
                if _ANTONYMS.get(tc) == ts:
                    return True, (
                        f"Polarity contradiction: Claim states '{tc}', but evidence states opposite '{ts}' "
                        f"in statement '{norm_sentence}'."
                    )

                # 3. Numeric Discrepancy Check (token-level)
                if self.allow_numeric_contradiction:
                    if self._is_numeric(tc) and self._is_numeric(ts):
                        return True, (
                            f"Numerical discrepancy: Claim specifies '{tc}', but evidence states '{ts}' "
                            f"in statement '{norm_sentence}'."
                        )

        # 3. Numeric Discrepancy Check (regex frame matching)
        if self.allow_numeric_contradiction:
            num_matches = list(re.finditer(r"\b\d+(?:\.\d+)?\b", norm_claim))
            if len(num_matches) == 1:
                match = num_matches[0]
                num_val = match.group(0)
                prefix = re.escape(norm_claim[: match.start()])
                suffix = re.escape(norm_claim[match.end() :])
                num_pattern = rf"(?:^|[,\s;:\"'(-]){prefix}(\d+(?:\.\d+)?){suffix}(?:[,\s;:\"')-]|$)"
                m = re.search(num_pattern, norm_sentence)
                if m:
                    found_num = m.group(1)
                    if found_num != num_val:
                        return True, (
                            f"Numerical discrepancy: Claim specifies '{num_val}', but evidence states '{found_num}' "
                            f"in statement '{norm_sentence}'."
                        )

        return False, None

    @staticmethod
    def _is_numeric(token: str) -> bool:
        """Returns True if token represents an integer or float value."""
        try:
            float(token)
            return True
        except ValueError:
            return False


# ==============================================================================
# Convenience Functions
# ==============================================================================


def verify_claim(
    claim: Any,
    evidence: Optional[
        Sequence[Union[EvidencePassage, DocumentChunk, RetrievalResult, HybridRetrievalResult]]
    ],
    citations: Optional[Sequence[CitationReference]] = None,
    verifier: Optional[BaseClaimVerifier] = None,
) -> VerificationOutcome:
    """Verifies a claim statement against evidence using the active or default verifier.

    Args:
        claim: Plain-text factual statement to verify.
        evidence: Sequence of evidence items acting as the knowledge base.
        citations: Optional sequence of CitationReference items to validate.
        verifier: Optional custom verifier instance (defaults to DeterministicClaimVerifier).

    Returns:
        VerificationOutcome with verification status, confidence, and validated citations.
    """
    engine = verifier if verifier is not None else DeterministicClaimVerifier()
    return engine.verify(claim=claim, evidence=evidence, citations=citations)

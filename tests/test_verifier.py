"""
VerityAI - Claim-Level Evidence Verification Test Suite
Phase 4: Agentic Verification and Citations (Chunk 3)

Tests:
1. Clearly supporting evidence (exact, case-insensitive, clause, PDF/TXT provenance).
2. Clearly conflicting evidence (direct negation, antonym polarity, numeric discrepancy).
3. Unrelated claim and evidence (returns INSUFFICIENT_EVIDENCE).
4. Empty evidence collection (returns INSUFFICIENT_EVIDENCE).
5. Empty or malformed claims (returns UNVERIFIABLE).
6. Invalid citation references (returns UNVERIFIABLE via CitationValidator).
7. Missing or empty source metadata (returns UNVERIFIABLE).
8. Multiple passages with mixed/conflicting evidence (returns INSUFFICIENT_EVIDENCE).
9. Honest refusal to guess on superficial keyword overlap (returns INSUFFICIENT_EVIDENCE).
10. Stable, immutable, and predictable verification result structures.
11. Abstract BaseClaimVerifier interface contract and custom verifier support.
12. Compatibility with DocumentChunk, RetrievalResult, and HybridRetrievalResult.
"""

from typing import Any, Tuple

import pytest

from verityai.chunking import ChunkMetadata, DocumentChunk
from verityai.citation_validator import CitationValidator
from verityai.evidence import (
    CitationReference,
    EvidencePassage,
    VerificationOutcome,
    VerificationStatus,
)
from verityai.retrieval_hybrid import HybridRetrievalResult
from verityai.retrieval_semantic import RetrievalResult
from verityai.verifier import (
    BaseClaimVerifier,
    DeterministicClaimVerifier,
    normalize_statement,
    split_into_sentences,
    verify_claim,
)


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def txt_passage() -> EvidencePassage:
    """A standard plain text evidence passage fixture."""
    return EvidencePassage(
        text="The VerityAI system operates completely offline without external cloud dependencies.",
        doc_id="doc_verity_101",
        filename="architecture_overview.txt",
        chunk_id="chunk_001",
        score=0.92,
        rank=1,
        doc_type="txt",
    )


@pytest.fixture
def pdf_passage() -> EvidencePassage:
    """A standard PDF evidence passage fixture with page number."""
    return EvidencePassage(
        text="All customer biometric records are encrypted at rest using AES-256-GCM.",
        doc_id="doc_security_202",
        filename="security_compliance.pdf",
        chunk_id="chunk_sec_04",
        page_number=7,
        score=0.88,
        rank=2,
        doc_type="pdf",
    )


@pytest.fixture
def server_status_passages() -> Tuple[EvidencePassage, EvidencePassage]:
    """Passages with mutually conflicting assertions for mixed-evidence testing."""
    p1 = EvidencePassage(
        text="The primary database cluster is currently online and serving read-write traffic.",
        doc_id="doc_ops_01",
        filename="cluster_status_primary.txt",
        chunk_id="chunk_p1",
        doc_type="txt",
    )
    p2 = EvidencePassage(
        text="The primary database cluster is currently offline for emergency maintenance.",
        doc_id="doc_ops_02",
        filename="cluster_status_secondary.txt",
        chunk_id="chunk_p2",
        doc_type="txt",
    )
    return p1, p2


# ==============================================================================
# Supporting Evidence Tests
# ==============================================================================


class TestSupportingEvidence:
    """Tests for claims with clearly supporting evidence."""

    def test_verbatim_supporting_evidence(self, txt_passage: EvidencePassage) -> None:
        """Verifies exact verbatim match returns SUPPORTED with valid citation."""
        claim = "The VerityAI system operates completely offline without external cloud dependencies."
        verifier = DeterministicClaimVerifier()
        result = verifier.verify(claim=claim, evidence=[txt_passage])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.is_verified is True
        assert result.is_contradicted is False
        assert result.is_unverified is False
        assert result.confidence == 0.92
        assert len(result.evidence) == 1
        assert len(result.citations) == 1
        assert result.citations[0].doc_id == txt_passage.doc_id
        assert result.citations[0].chunk_id == txt_passage.chunk_id
        assert "architecture_overview.txt" in result.explanation

    def test_case_insensitive_and_punctuation_normalized_match(
        self, txt_passage: EvidencePassage
    ) -> None:
        """Verifies case variations and trailing punctuation are handled correctly."""
        claim = "THE VERITYAI SYSTEM OPERATES COMPLETELY OFFLINE WITHOUT EXTERNAL CLOUD DEPENDENCIES"
        verifier = DeterministicClaimVerifier()
        result = verifier.verify(claim=claim, evidence=[txt_passage])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.is_verified is True

    def test_contraction_expansion_match(self) -> None:
        """Verifies contraction normalization produces matching statements."""
        passage = EvidencePassage(
            text="The backup worker is not running during off-peak hours.",
            doc_id="doc_cron",
            filename="cron_jobs.txt",
            chunk_id="chunk_c1",
        )
        claim = "The backup worker isn't running during off-peak hours."
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.is_verified is True

    def test_pdf_passage_preserves_page_number_in_citation(
        self, pdf_passage: EvidencePassage
    ) -> None:
        """Verifies citations generated from PDF passages preserve 1-indexed page number."""
        claim = "All customer biometric records are encrypted at rest using AES-256-GCM."
        result = verify_claim(claim=claim, evidence=[pdf_passage])

        assert result.status == VerificationStatus.SUPPORTED
        assert len(result.citations) == 1
        assert result.citations[0].page_number == 7
        assert result.citations[0].filename == "security_compliance.pdf"
        assert result.citations[0].format_inline() == "[security_compliance.pdf, p. 7]"

    def test_independent_clause_matching_within_larger_sentence(self) -> None:
        """Verifies claim matching an independent clause inside a sentence is supported."""
        passage = EvidencePassage(
            text="According to the audit report, the server cluster runs on Ubuntu Linux, as verified by ops.",
            doc_id="doc_audit",
            filename="audit.txt",
            chunk_id="chunk_a1",
        )
        claim = "the server cluster runs on Ubuntu Linux"
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.is_verified is True


# ==============================================================================
# Contradicting Evidence Tests
# ==============================================================================


class TestContradictingEvidence:
    """Tests for claims with clearly conflicting evidence."""

    def test_direct_negation_evidence_has_not(self) -> None:
        """Evidence contains 'not' while claim asserts positive."""
        passage = EvidencePassage(
            text="The background sync service is not enabled by default.",
            doc_id="doc_sync",
            filename="sync_config.txt",
            chunk_id="chunk_s1",
        )
        claim = "The background sync service is enabled by default."
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.CONTRADICTED
        assert result.is_contradicted is True
        assert result.is_verified is False
        assert result.contradiction_details is not None
        assert "Direct negation" in result.contradiction_details
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == "chunk_s1"

    def test_direct_negation_claim_has_not(self) -> None:
        """Claim asserts negative while evidence asserts positive."""
        passage = EvidencePassage(
            text="The telemetry logging pipeline is active.",
            doc_id="doc_telemetry",
            filename="telemetry.txt",
            chunk_id="chunk_t1",
        )
        claim = "The telemetry logging pipeline is not active."
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.CONTRADICTED
        assert result.is_contradicted is True
        assert result.contradiction_details is not None

    def test_antonym_polarity_contradiction(self) -> None:
        """Tests contradiction via curated antonym pairs ('online' vs 'offline')."""
        passage = EvidencePassage(
            text="The primary database cluster is currently offline.",
            doc_id="doc_db",
            filename="db_status.txt",
            chunk_id="chunk_db1",
        )
        claim = "The primary database cluster is currently online."
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.CONTRADICTED
        assert result.is_contradicted is True
        assert "Polarity contradiction" in result.contradiction_details
        assert "online" in result.contradiction_details
        assert "offline" in result.contradiction_details

    def test_antonym_enabled_disabled(self) -> None:
        """Tests contradiction between 'enabled' and 'disabled'."""
        passage = EvidencePassage(
            text="Hardware acceleration is disabled on low-power devices.",
            doc_id="doc_hw",
            filename="hw.txt",
            chunk_id="chunk_hw1",
        )
        claim = "Hardware acceleration is enabled on low-power devices."
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.CONTRADICTED
        assert result.is_contradicted is True

    def test_numeric_scalar_discrepancy(self) -> None:
        """Tests contradiction when single numeric value differs on identical predicate."""
        passage = EvidencePassage(
            text="The default HTTP request timeout is 30 seconds.",
            doc_id="doc_net",
            filename="network.txt",
            chunk_id="chunk_net1",
        )
        claim = "The default HTTP request timeout is 60 seconds."
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.CONTRADICTED
        assert result.is_contradicted is True
        assert "Numerical discrepancy" in result.contradiction_details
        assert "60" in result.contradiction_details
        assert "30" in result.contradiction_details


# ==============================================================================
# Unrelated and Ambiguous / Keyword Overlap Tests
# ==============================================================================


class TestUnrelatedAndInsufficientEvidence:
    """Tests ensuring the verifier returns INSUFFICIENT_EVIDENCE instead of guessing."""

    def test_claim_unrelated_to_supplied_evidence(self, txt_passage: EvidencePassage) -> None:
        """An entirely unrelated claim must return INSUFFICIENT_EVIDENCE, not CONTRADICTED."""
        claim = "The photosynthesis process converts sunlight into glucose in chloroplasts."
        result = verify_claim(claim=claim, evidence=[txt_passage])

        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.is_insufficient_evidence is True
        assert result.is_unverified is True
        assert result.is_verified is False
        assert result.is_contradicted is False
        assert result.confidence == 0.0

    def test_superficial_keyword_overlap_returns_insufficient_evidence(self) -> None:
        """CRITICAL CONSTRAINT: High keyword overlap must NOT falsely support or contradict a claim."""
        passage = EvidencePassage(
            text="The infrastructure engineering team met in 2024 to discuss microservices architecture.",
            doc_id="doc_meeting",
            filename="meeting_notes.txt",
            chunk_id="chunk_meet1",
        )
        # Shares words: 'infrastructure', 'engineering', 'microservices', 'architecture'
        claim = "The infrastructure engineering team deployed microservices architecture into production."
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.is_insufficient_evidence is True
        assert result.is_verified is False
        assert result.is_contradicted is False
        assert "No direct propositional corroboration" in result.explanation

    def test_partially_overlapping_numeric_claim_not_falsely_contradicted(self) -> None:
        """Statements that differ in multiple words should not trigger naive numeric contradiction."""
        passage = EvidencePassage(
            text="Python 3.10 was released in October 2021 by the core development team.",
            doc_id="doc_py",
            filename="python_history.txt",
            chunk_id="chunk_py1",
        )
        claim = "Python was originally released in 1991 by Guido van Rossum."
        result = verify_claim(claim=claim, evidence=[passage])

        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.is_contradicted is False
        assert result.is_verified is False


# ==============================================================================
# Empty and Malformed Input Tests
# ==============================================================================


class TestEmptyAndMalformedInputs:
    """Tests verifying robust handling of empty and invalid inputs."""

    def test_empty_evidence_collection(self) -> None:
        """Empty evidence collection returns INSUFFICIENT_EVIDENCE."""
        claim = "The server is running."
        result = verify_claim(claim=claim, evidence=[])

        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.is_insufficient_evidence is True
        assert result.evidence == ()
        assert result.citations == ()
        assert result.confidence == 0.0

    def test_none_evidence_collection(self) -> None:
        """None evidence collection returns INSUFFICIENT_EVIDENCE."""
        claim = "The server is running."
        result = verify_claim(claim=claim, evidence=None)

        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.is_insufficient_evidence is True

    @pytest.mark.parametrize("bad_claim", ["", "   ", "\t\n", None, 12345, ["claim"]])
    def test_empty_or_malformed_claims_return_unverifiable(
        self, bad_claim: Any, txt_passage: EvidencePassage
    ) -> None:
        """Empty, non-string, or whitespace claims must return UNVERIFIABLE."""
        result = verify_claim(claim=bad_claim, evidence=[txt_passage])

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.is_unverifiable is True
        assert result.is_verified is False
        assert result.confidence == 0.0
        assert "Claim input is invalid" in result.explanation

    def test_malformed_evidence_item_type(self) -> None:
        """Evidence sequence containing unsupported types returns UNVERIFIABLE."""
        claim = "The server is running."
        result = verify_claim(claim=claim, evidence=["not an evidence object"])  # type: ignore

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.is_unverifiable is True
        assert "invalid item" in result.explanation

    @pytest.mark.parametrize(
        "bad_doc_id, bad_filename, bad_chunk_id",
        [
            ("", "valid.txt", "chunk_1"),
            ("doc_1", "", "chunk_1"),
            ("doc_1", "valid.txt", ""),
            ("   ", "valid.txt", "chunk_1"),
        ],
    )
    def test_missing_or_empty_source_metadata(
        self, bad_doc_id: str, bad_filename: str, bad_chunk_id: str
    ) -> None:
        """Evidence passages with missing essential provenance metadata return UNVERIFIABLE."""
        claim = "Any assertion."
        # Bypass frozen dataclass check via object.__setattr__ to simulate corrupted input
        p = EvidencePassage(
            text="Valid passage text content.",
            doc_id="dummy_doc",
            filename="dummy.txt",
            chunk_id="dummy_chunk",
        )
        object.__setattr__(p, "doc_id", bad_doc_id)
        object.__setattr__(p, "filename", bad_filename)
        object.__setattr__(p, "chunk_id", bad_chunk_id)

        result = verify_claim(claim=claim, evidence=[p])

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.is_unverifiable is True
        assert "missing or empty required metadata" in result.explanation


# ==============================================================================
# Citation Reference Validation Tests
# ==============================================================================


class TestCitationReferenceValidation:
    """Tests requiring citations to resolve through CitationValidator."""

    def test_valid_supplied_citation_resolves(self, txt_passage: EvidencePassage) -> None:
        """Valid CitationReference pointing to genuine evidence is verified and retained."""
        claim = "The VerityAI system operates completely offline without external cloud dependencies."
        citation = CitationReference.from_evidence(txt_passage, citation_id="c_01")
        result = verify_claim(claim=claim, evidence=[txt_passage], citations=[citation])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.is_verified is True
        assert len(result.citations) == 1
        assert result.citations[0].citation_id == "c_01"

    def test_invalid_citation_reference_unknown_document(
        self, txt_passage: EvidencePassage
    ) -> None:
        """Citation with non-existent doc_id causes UNVERIFIABLE outcome."""
        claim = "The VerityAI system operates completely offline without external cloud dependencies."
        invalid_cite = CitationReference(
            citation_id="bad_1",
            doc_id="fabricated_doc_id",
            filename="architecture_overview.txt",
            chunk_id=txt_passage.chunk_id,
        )
        result = verify_claim(claim=claim, evidence=[txt_passage], citations=[invalid_cite])

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.is_unverifiable is True
        assert "Citation reference validation failed" in result.explanation
        assert result.citations == ()

    def test_invalid_citation_reference_chunk_mismatch(
        self, txt_passage: EvidencePassage
    ) -> None:
        """Citation with wrong chunk_id causes UNVERIFIABLE outcome."""
        claim = "Valid claim."
        invalid_cite = CitationReference(
            citation_id="bad_2",
            doc_id=txt_passage.doc_id,
            filename=txt_passage.filename,
            chunk_id="chunk_nonexistent_999",
        )
        result = verify_claim(claim=claim, evidence=[txt_passage], citations=[invalid_cite])

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.is_unverifiable is True

    def test_invalid_citation_reference_page_number_mismatch(
        self, pdf_passage: EvidencePassage
    ) -> None:
        """Citation for PDF with wrong page number causes UNVERIFIABLE outcome."""
        claim = "All customer biometric records are encrypted at rest using AES-256-GCM."
        invalid_cite = CitationReference(
            citation_id="bad_3",
            doc_id=pdf_passage.doc_id,
            filename=pdf_passage.filename,
            chunk_id=pdf_passage.chunk_id,
            page_number=99,  # Genuine is 7
        )
        result = verify_claim(claim=claim, evidence=[pdf_passage], citations=[invalid_cite])

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.is_unverifiable is True

    def test_malformed_citations_argument_type(self, txt_passage: EvidencePassage) -> None:
        """Passing non-sequence citations raises or returns UNVERIFIABLE."""
        claim = "Valid claim."
        result = verify_claim(claim=claim, evidence=[txt_passage], citations="bad_type")  # type: ignore

        assert result.status == VerificationStatus.UNVERIFIABLE
        assert result.is_unverifiable is True


# ==============================================================================
# Multi-Passage and Mixed Evidence Tests
# ==============================================================================


class TestMultiPassageAndMixedEvidence:
    """Tests evaluating behavior across multiple passages with conflicting or mixed signals."""

    def test_mixed_evidence_across_passages_returns_insufficient_evidence(
        self, server_status_passages: Tuple[EvidencePassage, EvidencePassage]
    ) -> None:
        """When passage A supports and passage B contradicts, returns INSUFFICIENT_EVIDENCE."""
        p1, p2 = server_status_passages
        claim = "The primary database cluster is currently online."
        result = verify_claim(claim=claim, evidence=[p1, p2])

        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.is_insufficient_evidence is True
        assert result.is_verified is False
        assert result.is_contradicted is False
        assert "Conflicting evidence detected" in result.explanation
        assert result.contradiction_details is not None
        assert len(result.evidence) == 2

    def test_supporting_passage_alongside_neutral_passages(
        self, txt_passage: EvidencePassage, pdf_passage: EvidencePassage
    ) -> None:
        """One supporting passage among neutral passages results in SUPPORTED."""
        claim = "The VerityAI system operates completely offline without external cloud dependencies."
        result = verify_claim(claim=claim, evidence=[pdf_passage, txt_passage])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.is_verified is True
        assert len(result.evidence) == 1
        assert result.evidence[0].doc_id == txt_passage.doc_id

    def test_contradicting_passage_alongside_neutral_passages(
        self, pdf_passage: EvidencePassage
    ) -> None:
        """One contradicting passage among neutral passages results in CONTRADICTED."""
        contra_passage = EvidencePassage(
            text="The audit proved all biometric records are stored unencrypted in plaintext.",
            doc_id="doc_leak",
            filename="leak_report.txt",
            chunk_id="chunk_l1",
        )
        claim = "The audit proved all biometric records are stored encrypted in plaintext."
        # Differ by 'unencrypted' vs 'encrypted' -> polarity/antonym contradiction
        result = verify_claim(claim=claim, evidence=[pdf_passage, contra_passage])

        assert result.status == VerificationStatus.CONTRADICTED
        assert result.is_contradicted is True


# ==============================================================================
# Result Stability, Predictability, and Typing Tests
# ==============================================================================


class TestVerificationResultStructure:
    """Tests ensuring stable, predictable, typed result structures."""

    def test_outcome_immutability(self, txt_passage: EvidencePassage) -> None:
        """VerificationOutcome must be frozen and immutable."""
        claim = "The VerityAI system operates completely offline without external cloud dependencies."
        result = verify_claim(claim=claim, evidence=[txt_passage])

        with pytest.raises(Exception):
            result.status = VerificationStatus.CONTRADICTED  # type: ignore

    def test_outcome_to_dict_serialization(self, txt_passage: EvidencePassage) -> None:
        """to_dict() produces complete serialized dictionary."""
        claim = "The VerityAI system operates completely offline without external cloud dependencies."
        result = verify_claim(claim=claim, evidence=[txt_passage])
        d = result.to_dict()

        assert d["claim"] == claim
        assert d["status"] == "supported"
        assert d["is_verified"] is True
        assert d["is_contradicted"] is False
        assert d["is_unverified"] is False
        assert d["is_insufficient_evidence"] is False
        assert d["is_unverifiable"] is False
        assert isinstance(d["evidence"], list)
        assert isinstance(d["citations"], list)
        assert len(d["citations"]) == 1
        assert d["citations"][0]["chunk_id"] == "chunk_001"

    def test_abstract_verifier_cannot_be_instantiated(self) -> None:
        """BaseClaimVerifier cannot be instantiated directly without implementing verify."""
        with pytest.raises(TypeError):
            BaseClaimVerifier()  # type: ignore

    def test_custom_verifier_injection(self, txt_passage: EvidencePassage) -> None:
        """verify_claim accepts a custom BaseClaimVerifier implementation."""

        class MockAlwaysInsufficientVerifier(BaseClaimVerifier):
            def verify(self, claim: Any, evidence: Any, citations: Any = None) -> VerificationOutcome:
                return VerificationOutcome(
                    claim=str(claim),
                    status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                    explanation="Custom mock verifier always returns insufficient evidence.",
                )

        custom = MockAlwaysInsufficientVerifier()
        result = verify_claim(
            claim="Any claim statement.",
            evidence=[txt_passage],
            verifier=custom,
        )
        assert result.status == VerificationStatus.INSUFFICIENT_EVIDENCE
        assert "Custom mock verifier" in result.explanation


# ==============================================================================
# Alternative Evidence Types Compatibility Tests
# ==============================================================================


class TestAlternativeEvidenceTypes:
    """Tests compatibility with DocumentChunk, RetrievalResult, and HybridRetrievalResult."""

    def test_verify_with_document_chunk(self) -> None:
        """DocumentChunk instances are accepted and normalized."""
        chunk = DocumentChunk(
            text="The deployment pipeline runs on GitHub Actions.",
            metadata=ChunkMetadata(
                chunk_id="chunk_ci_1",
                doc_id="doc_gh_ci",
                filename="ci.txt",
                doc_type="txt",
                page_number=None,
                chunk_index=0,
                char_count=48,
                word_count=7,
            ),
        )
        claim = "The deployment pipeline runs on GitHub Actions."
        result = verify_claim(claim=claim, evidence=[chunk])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.is_verified is True
        assert result.citations[0].doc_id == "doc_gh_ci"

    def test_verify_with_semantic_retrieval_result(self) -> None:
        """Semantic RetrievalResult instances are accepted and normalized."""
        chunk = DocumentChunk(
            text="Data retention period is strictly 90 days.",
            metadata=ChunkMetadata(
                chunk_id="chunk_ret_1",
                doc_id="doc_ret",
                filename="policy.txt",
                doc_type="txt",
                page_number=None,
                chunk_index=0,
                char_count=42,
                word_count=7,
            ),
        )
        ret_result = RetrievalResult(chunk=chunk, score=0.91, rank=1)
        claim = "Data retention period is strictly 90 days."
        result = verify_claim(claim=claim, evidence=[ret_result])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.confidence == 0.91

    def test_verify_with_hybrid_retrieval_result(self) -> None:
        """HybridRetrievalResult instances are accepted and normalized."""
        chunk = DocumentChunk(
            text="Service mesh utilizes Envoy proxy sidecars.",
            metadata=ChunkMetadata(
                chunk_id="chunk_m1",
                doc_id="doc_mesh",
                filename="mesh.txt",
                doc_type="txt",
                page_number=None,
                chunk_index=0,
                char_count=43,
                word_count=6,
            ),
        )
        hybrid_res = HybridRetrievalResult(
            chunk=chunk,
            rrf_score=0.032,
            rank=1,
            semantic_rank=1,
            keyword_rank=1,
        )
        claim = "Service mesh utilizes Envoy proxy sidecars."
        result = verify_claim(claim=claim, evidence=[hybrid_res])

        assert result.status == VerificationStatus.SUPPORTED
        assert result.is_verified is True
        assert result.citations[0].filename == "mesh.txt"

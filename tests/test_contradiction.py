"""
VerityAI - Cross-Document Contradiction Detection Test Suite
Phase 4: Agentic Verification and Citations (Chunk 4)

Tests:
1. Clearly conflicting statements with matching context:
   - Polarity / Antonym conflicts ("supported" vs "unsupported", "enabled" vs "disabled")
   - Negation conflicts ("is active" vs "is not active")
   - Numeric discrepancies on identical measurement ("12 nodes" vs "18 nodes")
   - Temporal value conflicts ("takes effect in January" vs "takes effect in March")
2. Compatible non-conflicting statements.
3. Statements with differing versions or dates (CONTEXT_MISMATCH).
4. Different measurement units (CONTEXT_MISMATCH).
5. Identical passages / self-comparison exclusion (IDENTICAL_PASSAGES).
6. Symmetric and duplicate conflict prevention.
7. Missing source metadata / malformed inputs (INVALID_INPUT / INCONCLUSIVE).
8. Empty input collections.
9. Ambiguous evidence with topical overlap returning INCONCLUSIVE.
10. Full document, chunk, and page provenance preservation.
11. Deterministic and stable output ordering across runs.
12. Multi-type evidence compatibility (DocumentChunk, RetrievalResult, HybridRetrievalResult).
13. Abstract BaseContradictionDetector contract and custom detector injection.
"""

from typing import Any, List

import pytest

from verityai.chunking import ChunkMetadata, DocumentChunk
from verityai.contradiction import (
    BaseContradictionDetector,
    ContradictionReasonCode,
    ContradictionReport,
    ContradictionStatus,
    CrossDocumentConflict,
    DeterministicContradictionDetector,
    detect_cross_document_contradictions,
)
from verityai.evidence import EvidencePassage
from verityai.retrieval_hybrid import HybridRetrievalResult
from verityai.retrieval_semantic import RetrievalResult


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def doc1_passage_supported() -> EvidencePassage:
    """Passage asserting feature is supported."""
    return EvidencePassage(
        text="Hardware acceleration is supported on standard nodes.",
        doc_id="doc_infra_01",
        filename="infra_guide.txt",
        chunk_id="chunk_inf_1",
        page_number=None,
        doc_type="txt",
    )


@pytest.fixture
def doc2_passage_unsupported() -> EvidencePassage:
    """Passage asserting feature is unsupported under matching context."""
    return EvidencePassage(
        text="Hardware acceleration is unsupported on standard nodes.",
        doc_id="doc_limits_02",
        filename="limitations.pdf",
        chunk_id="chunk_lim_4",
        page_number=12,
        doc_type="pdf",
    )


# ==============================================================================
# Clearly Conflicting Statements Tests
# ==============================================================================


class TestClearlyConflictingStatements:
    """Tests for statements that directly contradict under matching context."""

    def test_polarity_antonym_contradiction(
        self,
        doc1_passage_supported: EvidencePassage,
        doc2_passage_unsupported: EvidencePassage,
    ) -> None:
        """Contradiction detected when antonym polarity differs under matching context."""
        detector = DeterministicContradictionDetector()
        result = detector.compare_passages(doc1_passage_supported, doc2_passage_unsupported)

        assert result.status == ContradictionStatus.POTENTIAL_CONTRADICTION
        assert result.is_contradiction is True
        assert result.reason_code == ContradictionReasonCode.POLARITY_CONFLICT
        assert "Polarity conflict" in result.explanation
        assert result.passage_a.doc_id == "doc_infra_01"
        assert result.passage_b.doc_id == "doc_limits_02"
        assert result.passage_b.page_number == 12

    def test_direct_negation_contradiction(self) -> None:
        """Contradiction detected when one source negates another's assertion."""
        p1 = EvidencePassage(
            text="The telemetry export daemon is active by default.",
            doc_id="doc_telemetry_a",
            filename="telemetry_v1.txt",
            chunk_id="chunk_t1",
        )
        p2 = EvidencePassage(
            text="The telemetry export daemon is not active by default.",
            doc_id="doc_telemetry_b",
            filename="telemetry_v2.txt",
            chunk_id="chunk_t2",
        )
        result = detect_cross_document_contradictions([p1, p2])

        assert result.status == ContradictionStatus.POTENTIAL_CONTRADICTION
        assert result.has_contradictions is True
        assert result.conflict_count == 1
        conflict = result.conflicts[0]
        assert conflict.reason_code == ContradictionReasonCode.NEGATION_CONFLICT
        assert conflict.is_contradiction is True

    def test_numeric_discrepancy_matching_context(self) -> None:
        """Contradiction detected when identical measurement reports different numeric values."""
        p1 = EvidencePassage(
            text="The primary storage cluster operates with 12 worker nodes.",
            doc_id="doc_cluster_alpha",
            filename="cluster_alpha.txt",
            chunk_id="chunk_c1",
        )
        p2 = EvidencePassage(
            text="The primary storage cluster operates with 18 worker nodes.",
            doc_id="doc_cluster_beta",
            filename="cluster_beta.txt",
            chunk_id="chunk_c2",
        )
        result = detect_cross_document_contradictions([p1, p2])

        assert result.has_contradictions is True
        conflict = result.conflicts[0]
        assert conflict.reason_code == ContradictionReasonCode.NUMERIC_CONFLICT
        assert "12" in conflict.explanation
        assert "18" in conflict.explanation

    def test_temporal_value_conflict_without_date_qualification(self) -> None:
        """Contradiction detected when conflicting effective months are asserted for the same policy."""
        p1 = EvidencePassage(
            text="The mandatory password rotation policy takes effect in January.",
            doc_id="doc_hr_01",
            filename="policy_draft_a.txt",
            chunk_id="chunk_hr_1",
        )
        p2 = EvidencePassage(
            text="The mandatory password rotation policy takes effect in March.",
            doc_id="doc_hr_02",
            filename="policy_draft_b.txt",
            chunk_id="chunk_hr_2",
        )
        result = detect_cross_document_contradictions([p1, p2])

        assert result.has_contradictions is True
        conflict = result.conflicts[0]
        assert conflict.reason_code == ContradictionReasonCode.TEMPORAL_VALUE_CONFLICT
        assert "january" in conflict.explanation.lower()
        assert "march" in conflict.explanation.lower()


# ==============================================================================
# Context Differentiated and Compatible Statements Tests
# ==============================================================================


class TestContextDifferentiationAndCompatibility:
    """Tests ensuring context variations (dates, versions, units) are NOT flagged as contradictions."""

    def test_differing_software_versions_are_context_mismatch(self) -> None:
        """Features differing across explicit version qualifiers are context mismatches, not contradictions."""
        p1 = EvidencePassage(
            text="In version 1.0, dark mode is unsupported on standard nodes.",
            doc_id="doc_v1",
            filename="v1_release_notes.txt",
            chunk_id="chunk_v1",
        )
        p2 = EvidencePassage(
            text="In version 2.0, dark mode is supported on standard nodes.",
            doc_id="doc_v2",
            filename="v2_release_notes.txt",
            chunk_id="chunk_v2",
        )
        detector = DeterministicContradictionDetector()
        conflict = detector.compare_passages(p1, p2)

        assert conflict.status == ContradictionStatus.NO_CONTRADICTION_DETECTED
        assert conflict.is_contradiction is False
        assert conflict.reason_code == ContradictionReasonCode.CONTEXT_MISMATCH
        assert "version" in conflict.explanation.lower()

    def test_differing_dates_or_years_are_context_mismatch(self) -> None:
        """Different numbers tied to different calendar years represent temporal progression, not conflict."""
        p1 = EvidencePassage(
            text="In 2020, total active developer accounts numbered 12.",
            doc_id="doc_2020",
            filename="annual_report_2020.txt",
            chunk_id="chunk_ar20",
        )
        p2 = EvidencePassage(
            text="In 2024, total active developer accounts numbered 18.",
            doc_id="doc_2024",
            filename="annual_report_2024.txt",
            chunk_id="chunk_ar24",
        )
        detector = DeterministicContradictionDetector()
        conflict = detector.compare_passages(p1, p2)

        assert conflict.status == ContradictionStatus.NO_CONTRADICTION_DETECTED
        assert conflict.is_contradiction is False
        assert conflict.reason_code == ContradictionReasonCode.CONTEXT_MISMATCH
        assert "timeframes or dates" in conflict.explanation.lower()

    def test_differing_measurement_units_are_context_mismatch(self) -> None:
        """Measurements expressed in differing physical units are not direct contradictions."""
        p1 = EvidencePassage(
            text="The required rack clearance is 10 meters.",
            doc_id="doc_metric",
            filename="datacenter_metric.txt",
            chunk_id="chunk_m",
        )
        p2 = EvidencePassage(
            text="The required rack clearance is 10 feet.",
            doc_id="doc_imperial",
            filename="datacenter_imperial.txt",
            chunk_id="chunk_imp",
        )
        detector = DeterministicContradictionDetector()
        conflict = detector.compare_passages(p1, p2)

        assert conflict.status == ContradictionStatus.NO_CONTRADICTION_DETECTED
        assert conflict.reason_code == ContradictionReasonCode.CONTEXT_MISMATCH
        assert "different units" in conflict.explanation

    def test_compatible_non_conflicting_statements(self) -> None:
        """Mutually compatible statements yield NO_CONTRADICTION_DETECTED."""
        p1 = EvidencePassage(
            text="The server stack runs entirely on Ubuntu Linux.",
            doc_id="doc_os",
            filename="os_spec.txt",
            chunk_id="chunk_os",
        )
        p2 = EvidencePassage(
            text="The application requires Python 3.11 or newer runtime.",
            doc_id="doc_py",
            filename="python_spec.txt",
            chunk_id="chunk_py",
        )
        report = detect_cross_document_contradictions([p1, p2])

        assert report.status == ContradictionStatus.NO_CONTRADICTION_DETECTED
        assert report.has_contradictions is False
        assert report.conflict_count == 0


# ==============================================================================
# Ambiguous Evidence & Keyword Overlap Tests
# ==============================================================================


class TestAmbiguousEvidence:
    """Tests ensuring ambiguous or partial overlap evidence returns INCONCLUSIVE."""

    def test_topical_keyword_overlap_without_equivalent_predicate_is_inconclusive(self) -> None:
        """Passages sharing keywords but lacking exact frame equivalence return INCONCLUSIVE."""
        p1 = EvidencePassage(
            text="The database cluster was upgraded with 64 gigabytes of memory.",
            doc_id="doc_upgrade",
            filename="upgrade_log.txt",
            chunk_id="chunk_up",
        )
        p2 = EvidencePassage(
            text="The analytics database cluster handles 128 queries per second.",
            doc_id="doc_perf",
            filename="perf_log.txt",
            chunk_id="chunk_perf",
        )
        detector = DeterministicContradictionDetector()
        conflict = detector.compare_passages(p1, p2)

        assert conflict.status == ContradictionStatus.INCONCLUSIVE
        assert conflict.is_inconclusive is True
        assert conflict.is_contradiction is False
        assert conflict.reason_code == ContradictionReasonCode.AMBIGUOUS_EVIDENCE

    def test_report_marks_overall_inconclusive_when_no_contradictions_and_ambiguity_present(
        self,
    ) -> None:
        """When ambiguity exists and no conflicts are proven, overall report status is INCONCLUSIVE."""
        p1 = EvidencePassage(
            text="The database cluster was upgraded with 64 gigabytes of memory.",
            doc_id="doc_up",
            filename="up.txt",
            chunk_id="chunk_u",
        )
        p2 = EvidencePassage(
            text="The analytics database cluster handles 128 queries per second.",
            doc_id="doc_pf",
            filename="pf.txt",
            chunk_id="chunk_p",
        )
        report = detect_cross_document_contradictions([p1, p2])

        assert report.status == ContradictionStatus.INCONCLUSIVE
        assert report.has_contradictions is False
        assert report.inconclusive_count == 1
        assert "inconclusive evidence" in report.summary.lower()


# ==============================================================================
# Deduplication, Identity, and Scope Tests
# ==============================================================================


class TestDeduplicationAndPairHandling:
    """Tests for symmetric duplicate prevention and self-comparison exclusion."""

    def test_passage_compared_against_itself_is_identical(
        self, doc1_passage_supported: EvidencePassage
    ) -> None:
        """Self-comparison returns IDENTICAL_PASSAGES and NO_CONTRADICTION_DETECTED."""
        detector = DeterministicContradictionDetector()
        res = detector.compare_passages(doc1_passage_supported, doc1_passage_supported)

        assert res.status == ContradictionStatus.NO_CONTRADICTION_DETECTED
        assert res.reason_code == ContradictionReasonCode.IDENTICAL_PASSAGES

    def test_duplicate_passages_in_collection_are_deduplicated(
        self,
        doc1_passage_supported: EvidencePassage,
        doc2_passage_unsupported: EvidencePassage,
    ) -> None:
        """Supplying duplicate identical passages does not multiply evaluations or conflict reports."""
        passages = [
            doc1_passage_supported,
            doc2_passage_unsupported,
            doc1_passage_supported,  # duplicate
            doc2_passage_unsupported,  # duplicate
        ]
        report = detect_cross_document_contradictions(passages)

        assert report.total_pairs_evaluated == 1
        assert report.conflict_count == 1

    def test_intra_document_comparisons_skipped_by_default(self) -> None:
        """Passages from the same document (same doc_id) are not evaluated under cross_document_only."""
        p1 = EvidencePassage(
            text="Section A claims the server is online.",
            doc_id="single_doc_01",
            filename="manual.txt",
            chunk_id="chunk_sec_a",
        )
        p2 = EvidencePassage(
            text="Section B claims the server is offline.",
            doc_id="single_doc_01",
            filename="manual.txt",
            chunk_id="chunk_sec_b",
        )
        report = detect_cross_document_contradictions([p1, p2], cross_document_only=True)

        assert report.total_pairs_evaluated == 0
        assert report.conflict_count == 0

    def test_intra_document_comparisons_evaluated_when_flag_disabled(self) -> None:
        """When cross_document_only=False, intra-document passages are evaluated."""
        p1 = EvidencePassage(
            text="The server cluster is online.",
            doc_id="single_doc_01",
            filename="manual.txt",
            chunk_id="chunk_sec_a",
        )
        p2 = EvidencePassage(
            text="The server cluster is offline.",
            doc_id="single_doc_01",
            filename="manual.txt",
            chunk_id="chunk_sec_b",
        )
        report = detect_cross_document_contradictions([p1, p2], cross_document_only=False)

        assert report.total_pairs_evaluated == 1
        assert report.conflict_count == 1


# ==============================================================================
# Edge Cases & Input Validation Tests
# ==============================================================================


class TestEdgeCasesAndInputValidation:
    """Tests for empty inputs, missing source metadata, and malformed structures."""

    def test_empty_evidence_collection(self) -> None:
        """Empty evidence returns cleanly with 0 evaluated pairs."""
        report = detect_cross_document_contradictions([])

        assert report.status == ContradictionStatus.NO_CONTRADICTION_DETECTED
        assert report.total_pairs_evaluated == 0
        assert report.conflict_count == 0
        assert "No evidence passages provided" in report.summary

    def test_none_evidence_collection(self) -> None:
        """None input returns cleanly with 0 evaluated pairs."""
        report = detect_cross_document_contradictions(None)

        assert report.total_pairs_evaluated == 0
        assert report.conflict_count == 0

    @pytest.mark.parametrize(
        "bad_doc_id, bad_filename, bad_chunk_id",
        [
            ("", "valid.txt", "chunk_1"),
            ("doc_1", "", "chunk_1"),
            ("doc_1", "valid.txt", ""),
            ("   ", "valid.txt", "chunk_1"),
        ],
    )
    def test_missing_source_metadata_handled_safely(
        self, bad_doc_id: str, bad_filename: str, bad_chunk_id: str
    ) -> None:
        """Passages with missing required metadata return INCONCLUSIVE / INVALID_INPUT without crashing."""
        p1 = EvidencePassage(
            text="Valid passage statement.",
            doc_id="dummy_doc",
            filename="dummy.txt",
            chunk_id="dummy_chunk",
        )
        object.__setattr__(p1, "doc_id", bad_doc_id)
        object.__setattr__(p1, "filename", bad_filename)
        object.__setattr__(p1, "chunk_id", bad_chunk_id)

        p2 = EvidencePassage(
            text="Another valid statement.",
            doc_id="doc_valid_02",
            filename="valid.txt",
            chunk_id="chunk_v2",
        )
        detector = DeterministicContradictionDetector()
        res = detector.compare_passages(p1, p2)

        assert res.status == ContradictionStatus.INCONCLUSIVE
        assert res.reason_code == ContradictionReasonCode.INVALID_INPUT
        assert "missing or empty required metadata" in res.explanation.lower()

    def test_malformed_evidence_item_in_collection(
        self, doc1_passage_supported: EvidencePassage
    ) -> None:
        """Unsupported item type in evidence collection returns inconclusive report."""
        report = detect_cross_document_contradictions([doc1_passage_supported, "not a passage"])  # type: ignore

        assert report.status == ContradictionStatus.INCONCLUSIVE
        assert len(report.inconclusive) == 1
        assert report.inconclusive[0].reason_code == ContradictionReasonCode.INVALID_INPUT


# ==============================================================================
# Provenance & Stability Tests
# ==============================================================================


class TestProvenanceAndStability:
    """Tests ensuring complete source provenance preservation and deterministic sorting."""

    def test_full_provenance_preserved_in_conflict_and_to_dict(
        self,
        doc1_passage_supported: EvidencePassage,
        doc2_passage_unsupported: EvidencePassage,
    ) -> None:
        """Verifies doc_id, filename, chunk_id, and PDF page_number are preserved on both sides."""
        report = detect_cross_document_contradictions([doc1_passage_supported, doc2_passage_unsupported])
        assert report.has_contradictions is True

        conflict = report.conflicts[0]
        # Check passage_a and passage_b attributes
        assert conflict.passage_a.doc_id == "doc_infra_01"
        assert conflict.passage_a.filename == "infra_guide.txt"
        assert conflict.passage_a.chunk_id == "chunk_inf_1"
        assert conflict.passage_a.page_number is None

        assert conflict.passage_b.doc_id == "doc_limits_02"
        assert conflict.passage_b.filename == "limitations.pdf"
        assert conflict.passage_b.chunk_id == "chunk_lim_4"
        assert conflict.passage_b.page_number == 12

        # Check serialized dictionary format
        d = conflict.to_dict()
        assert d["source_a"]["doc_id"] == "doc_infra_01"
        assert d["source_b"]["doc_id"] == "doc_limits_02"
        assert d["source_b"]["page_number"] == 12
        assert d["is_contradiction"] is True

        rep_dict = report.to_dict()
        assert rep_dict["has_contradictions"] is True
        assert len(rep_dict["conflicts"]) == 1

    def test_deterministic_output_ordering_across_repeated_runs(
        self,
        doc1_passage_supported: EvidencePassage,
        doc2_passage_unsupported: EvidencePassage,
    ) -> None:
        """Reversed input sequence produces strictly identical conflict ordering and attributes."""
        passages_forward = [doc1_passage_supported, doc2_passage_unsupported]
        passages_reverse = [doc2_passage_unsupported, doc1_passage_supported]

        report_fwd = detect_cross_document_contradictions(passages_forward)
        report_rev = detect_cross_document_contradictions(passages_reverse)

        assert report_fwd.status == report_rev.status
        assert report_fwd.total_pairs_evaluated == report_rev.total_pairs_evaluated
        assert len(report_fwd.conflicts) == len(report_rev.conflicts)

        c_fwd = report_fwd.conflicts[0]
        c_rev = report_rev.conflicts[0]
        assert c_fwd.passage_a.doc_id == c_rev.passage_a.doc_id
        assert c_fwd.passage_b.doc_id == c_rev.passage_b.doc_id
        assert c_fwd.reason_code == c_rev.reason_code


# ==============================================================================
# Multi-Type Evidence & Extensibility Tests
# ==============================================================================


class TestMultiTypeEvidenceAndExtensibility:
    """Tests compatibility with DocumentChunk, RetrievalResult, and HybridRetrievalResult."""

    def test_cross_document_with_document_chunks(self) -> None:
        """DocumentChunk objects are accepted and compared across distinct doc_ids."""
        chunk1 = DocumentChunk(
            text="Service mesh sidecars are enabled by default.",
            metadata=ChunkMetadata(
                chunk_id="chunk_c1",
                doc_id="doc_mesh_prod",
                filename="mesh_prod.txt",
                doc_type="txt",
                page_number=None,
                chunk_index=0,
                char_count=45,
                word_count=7,
            ),
        )
        chunk2 = DocumentChunk(
            text="Service mesh sidecars are disabled by default.",
            metadata=ChunkMetadata(
                chunk_id="chunk_c2",
                doc_id="doc_mesh_dev",
                filename="mesh_dev.txt",
                doc_type="txt",
                page_number=None,
                chunk_index=0,
                char_count=46,
                word_count=7,
            ),
        )
        report = detect_cross_document_contradictions([chunk1, chunk2])

        assert report.has_contradictions is True
        conflict = report.conflicts[0]
        assert conflict.passage_a.doc_id == "doc_mesh_dev"  # canonically sorted
        assert conflict.passage_b.doc_id == "doc_mesh_prod"
        assert conflict.reason_code == ContradictionReasonCode.POLARITY_CONFLICT

    def test_abstract_detector_cannot_be_instantiated(self) -> None:
        """BaseContradictionDetector cannot be instantiated directly without implementations."""
        with pytest.raises(TypeError):
            BaseContradictionDetector()  # type: ignore

    def test_custom_detector_injection(
        self, doc1_passage_supported: EvidencePassage, doc2_passage_unsupported: EvidencePassage
    ) -> None:
        """Custom BaseContradictionDetector can be injected into helper function."""

        class MockNeverContradictingDetector(BaseContradictionDetector):
            def compare_passages(self, pa: Any, pb: Any) -> CrossDocumentConflict:
                return CrossDocumentConflict(
                    passage_a=pa,
                    passage_b=pb,
                    status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                    reason_code=ContradictionReasonCode.COMPATIBLE_STATEMENTS,
                    explanation="Mock detector never flags conflicts.",
                )

            def detect_contradictions(self, evidence: Any) -> ContradictionReport:
                return ContradictionReport(
                    conflicts=(),
                    inconclusive=(),
                    total_pairs_evaluated=1,
                    status=ContradictionStatus.NO_CONTRADICTION_DETECTED,
                    summary="Custom mock detector ran.",
                )

        custom = MockNeverContradictingDetector()
        report = detect_cross_document_contradictions(
            [doc1_passage_supported, doc2_passage_unsupported],
            detector=custom,
        )
        assert report.status == ContradictionStatus.NO_CONTRADICTION_DETECTED
        assert "Custom mock detector ran." in report.summary

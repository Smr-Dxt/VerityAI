"""
Tests for Phase 4 Chunk 2: Deterministic Citation Validation.
Verifies grounded evidence citation validation, PDF/TXT page checking, error codes,
duplicate detection, cross-request isolation, and privacy preservation.
"""

import pytest

from verityai.citation_validator import (
    CitationErrorCode,
    CitationValidationReport,
    CitationValidationResult,
    CitationValidator,
    validate_citation,
    validate_citations,
)
from verityai.chunking import ChunkMetadata, DocumentChunk
from verityai.evidence import CitationReference, EvidencePassage
from verityai.retrieval_hybrid import HybridRetrievalResult
from verityai.retrieval_semantic import RetrievalResult


# ==============================================================================
# Synthetic Test Fixtures
# ==============================================================================


@pytest.fixture
def synthetic_pdf_evidence() -> EvidencePassage:
    """Fixture providing synthetic PDF evidence with a 1-indexed page number."""
    return EvidencePassage(
        text="Synthetic PDF content: superconducting qubits maintain quantum coherence.",
        doc_id="synth_doc_pdf",
        filename="quantum_review.pdf",
        chunk_id="synth_doc_pdf_p3_c0",
        page_number=3,
        score=0.92,
        rank=1,
        doc_type="pdf",
    )


@pytest.fixture
def synthetic_txt_evidence() -> EvidencePassage:
    """Fixture providing synthetic TXT evidence without a page number."""
    return EvidencePassage(
        text="Synthetic TXT content: raw calibration parameters updated on 2026-10-01.",
        doc_id="synth_doc_txt",
        filename="calibration_notes.txt",
        chunk_id="synth_doc_txt_c0",
        page_number=None,
        score=0.81,
        rank=2,
        doc_type="txt",
    )


@pytest.fixture
def synthetic_evidence_pool(
    synthetic_pdf_evidence: EvidencePassage,
    synthetic_txt_evidence: EvidencePassage,
) -> list[EvidencePassage]:
    """Fixture providing a pool containing both PDF and TXT evidence."""
    return [synthetic_pdf_evidence, synthetic_txt_evidence]


# ==============================================================================
# Citation Validation Tests
# ==============================================================================


class TestCitationValidator:
    """Test suite covering all deterministic citation validation requirements."""

    def test_valid_pdf_citation_with_correct_page(
        self,
        synthetic_pdf_evidence: EvidencePassage,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies a valid PDF citation with matching doc_id, chunk_id, and page number."""
        citation = CitationReference(
            citation_id="[1]",
            doc_id="synth_doc_pdf",
            filename="quantum_review.pdf",
            chunk_id="synth_doc_pdf_p3_c0",
            page_number=3,
            quote="superconducting qubits maintain quantum coherence",
        )
        validator = CitationValidator(evidence_pool=synthetic_evidence_pool)
        result = validator.validate_citation(citation)

        assert result.is_valid is True
        assert result.error_code is None
        assert result.reason is None
        assert result.matched_evidence is not None
        assert result.matched_evidence.chunk_id == synthetic_pdf_evidence.chunk_id
        assert result.matched_evidence.page_number == 3

    def test_valid_txt_citation_without_page(
        self,
        synthetic_txt_evidence: EvidencePassage,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies a valid TXT citation without an invented page number."""
        citation = CitationReference(
            citation_id="[2]",
            doc_id="synth_doc_txt",
            filename="calibration_notes.txt",
            chunk_id="synth_doc_txt_c0",
            page_number=None,
            quote="raw calibration parameters",
        )
        result = validate_citation(citation, synthetic_evidence_pool)

        assert result.is_valid is True
        assert result.error_code is None
        assert result.reason is None
        assert result.matched_evidence is not None
        assert result.matched_evidence.doc_id == "synth_doc_txt"
        assert result.matched_evidence.page_number is None

    def test_unknown_document_id_rejected(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies citation referencing a non-existent document ID is rejected."""
        citation = CitationReference(
            citation_id="[bad-doc]",
            doc_id="completely_fabricated_doc_999",
            filename="fabricated.pdf",
            chunk_id="fabricated_p1_c0",
            page_number=1,
        )
        result = validate_citation(citation, synthetic_evidence_pool)

        assert result.is_valid is False
        assert result.error_code == CitationErrorCode.UNKNOWN_DOCUMENT
        assert "completely_fabricated_doc_999" in str(result.reason)
        assert result.matched_evidence is None

    def test_unknown_chunk_id_in_valid_document_rejected(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies citation with valid doc_id but fabricated chunk_id is rejected."""
        citation = CitationReference(
            citation_id="[bad-chunk]",
            doc_id="synth_doc_pdf",
            filename="quantum_review.pdf",
            chunk_id="synth_doc_pdf_p99_c99",  # Non-existent chunk
            page_number=3,
        )
        result = validate_citation(citation, synthetic_evidence_pool)

        assert result.is_valid is False
        assert result.error_code == CitationErrorCode.UNKNOWN_CHUNK
        assert "synth_doc_pdf_p99_c99" in str(result.reason)
        assert result.matched_evidence is None

    def test_incorrect_pdf_page_number_rejected(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies citation with wrong page number for a PDF source is rejected."""
        citation = CitationReference(
            citation_id="[wrong-page]",
            doc_id="synth_doc_pdf",
            filename="quantum_review.pdf",
            chunk_id="synth_doc_pdf_p3_c0",
            page_number=9,  # Genuine source is page 3
        )
        result = validate_citation(citation, synthetic_evidence_pool)

        assert result.is_valid is False
        assert result.error_code == CitationErrorCode.PAGE_NUMBER_MISMATCH
        assert "Cited page number 9 does not match source page 3" in str(result.reason)

    def test_missing_page_number_for_pdf_rejected(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies PDF citation without a page number is rejected."""
        citation = CitationReference(
            citation_id="[no-page-pdf]",
            doc_id="synth_doc_pdf",
            filename="quantum_review.pdf",
            chunk_id="synth_doc_pdf_p3_c0",
            page_number=None,  # Source is a PDF on page 3
        )
        result = validate_citation(citation, synthetic_evidence_pool)

        assert result.is_valid is False
        assert result.error_code == CitationErrorCode.MISSING_PAGE_NUMBER
        assert "missing page number" in str(result.reason)

    def test_unexpected_page_number_for_txt_rejected(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies plain text citation specifying an invented page number is rejected."""
        citation = CitationReference(
            citation_id="[invented-page-txt]",
            doc_id="synth_doc_txt",
            filename="calibration_notes.txt",
            chunk_id="synth_doc_txt_c0",
            page_number=4,  # TXT has no pages
        )
        result = validate_citation(citation, synthetic_evidence_pool)

        assert result.is_valid is False
        assert result.error_code == CitationErrorCode.UNEXPECTED_PAGE_NUMBER
        assert "plain text" in str(result.reason)

    def test_filename_mismatch_rejected(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies citation with matching doc_id/chunk_id but mismatched filename is rejected."""
        citation = CitationReference(
            citation_id="[wrong-file]",
            doc_id="synth_doc_pdf",
            filename="entirely_different_name.pdf",
            chunk_id="synth_doc_pdf_p3_c0",
            page_number=3,
        )
        result = validate_citation(citation, synthetic_evidence_pool)

        assert result.is_valid is False
        assert result.error_code == CitationErrorCode.FILENAME_MISMATCH
        assert "does not match source evidence filename" in str(result.reason)

    def test_empty_evidence_collection_safely_handled(self) -> None:
        """Verifies empty evidence pool safely returns invalid result without crashing."""
        citation = CitationReference(
            citation_id="[1]",
            doc_id="some_doc",
            filename="some_file.pdf",
            chunk_id="some_doc_p1_c0",
            page_number=1,
        )
        validator = CitationValidator(evidence_pool=[])
        result = validator.validate_citation(citation)

        assert result.is_valid is False
        assert result.error_code == CitationErrorCode.EMPTY_EVIDENCE
        assert "empty" in str(result.reason).lower()

    def test_empty_citations_and_empty_evidence_batch(self) -> None:
        """Verifies batch validation on empty citations returns empty report cleanly."""
        report = validate_citations([], [])
        assert report.total_citations == 0
        assert report.valid_count == 0
        assert report.invalid_count == 0
        assert report.is_all_valid is False
        assert "No citations" in report.summary()

    def test_duplicate_citations_handling(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies duplicate citations are detected and handled consistently."""
        cite1 = CitationReference(
            citation_id="[1]",
            doc_id="synth_doc_pdf",
            filename="quantum_review.pdf",
            chunk_id="synth_doc_pdf_p3_c0",
            page_number=3,
        )
        # Exact duplicate
        cite2 = CitationReference(
            citation_id="[1]",
            doc_id="synth_doc_pdf",
            filename="quantum_review.pdf",
            chunk_id="synth_doc_pdf_p3_c0",
            page_number=3,
        )

        # Default: allow_duplicates=False -> second citation marked invalid as duplicate
        report_strict = validate_citations([cite1, cite2], synthetic_evidence_pool, allow_duplicates=False)
        assert report_strict.total_citations == 2
        assert report_strict.valid_count == 1
        assert report_strict.invalid_count == 1
        assert report_strict.duplicate_count == 1
        assert report_strict.results[1].is_duplicate is True
        assert report_strict.results[1].error_code == CitationErrorCode.DUPLICATE_CITATION

        # Option: allow_duplicates=True -> second citation is flagged as duplicate but remains valid
        report_permissive = validate_citations([cite1, cite2], synthetic_evidence_pool, allow_duplicates=True)
        assert report_permissive.total_citations == 2
        assert report_permissive.valid_count == 2
        assert report_permissive.invalid_count == 0
        assert report_permissive.duplicate_count == 1
        assert report_permissive.results[1].is_duplicate is True
        assert report_permissive.results[1].is_valid is True

    def test_evidence_from_different_request_isolated(self) -> None:
        """Verifies evidence retrieved in Request A cannot validate citations from Request B."""
        pool_request_a = [
            EvidencePassage(
                text="Request A evidence about robotics kinematics.",
                doc_id="doc_request_a",
                filename="robotics.pdf",
                chunk_id="doc_request_a_p1_c0",
                page_number=1,
            )
        ]
        pool_request_b = [
            EvidencePassage(
                text="Request B evidence about renewable batteries.",
                doc_id="doc_request_b",
                filename="batteries.pdf",
                chunk_id="doc_request_b_p4_c0",
                page_number=4,
            )
        ]

        citation_for_b = CitationReference(
            citation_id="[B-1]",
            doc_id="doc_request_b",
            filename="batteries.pdf",
            chunk_id="doc_request_b_p4_c0",
            page_number=4,
        )

        # Validating against Pool B succeeds
        assert validate_citation(citation_for_b, pool_request_b).is_valid is True

        # Validating citation for B against Pool A must fail with UNKNOWN_DOCUMENT
        res_cross = validate_citation(citation_for_b, pool_request_a)
        assert res_cross.is_valid is False
        assert res_cross.error_code == CitationErrorCode.UNKNOWN_DOCUMENT

    def test_malformed_inputs_rejected(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies non-CitationReference or None inputs are gracefully rejected."""
        validator = CitationValidator(evidence_pool=synthetic_evidence_pool)

        # Pass None
        res_none = validator.validate_citation(None)
        assert res_none.is_valid is False
        assert res_none.error_code == CitationErrorCode.MALFORMED_CITATION

        # Pass raw string
        res_str = validator.validate_citation("Just a string citation")
        assert res_str.is_valid is False
        assert res_str.error_code == CitationErrorCode.MALFORMED_CITATION

        # Pass raw dict
        res_dict = validator.validate_citation({"doc_id": "doc_1"})
        assert res_dict.is_valid is False
        assert res_dict.error_code == CitationErrorCode.MALFORMED_CITATION

    def test_support_for_document_chunk_and_retrieval_results_in_pool(self) -> None:
        """Verifies CitationValidator seamlessly accepts DocumentChunk and RetrievalResult objects."""
        raw_chunk = DocumentChunk(
            text="Raw chunk passage from Phase 3 ingestion.",
            metadata=ChunkMetadata(
                chunk_id="chunk_doc_p2_c0",
                doc_id="chunk_doc",
                filename="phase3_paper.pdf",
                doc_type="pdf",
                page_number=2,
                chunk_index=0,
                char_count=42,
                word_count=7,
            ),
        )
        retrieval_res = RetrievalResult(
            chunk=raw_chunk,
            score=0.91,
            rank=1,
        )
        hybrid_res = HybridRetrievalResult(
            chunk=raw_chunk,
            rrf_score=0.033,
            rank=1,
        )

        # Validate with DocumentChunk in pool
        val_chunk = CitationValidator(evidence_pool=[raw_chunk])
        cite = CitationReference(
            citation_id="1",
            doc_id="chunk_doc",
            filename="phase3_paper.pdf",
            chunk_id="chunk_doc_p2_c0",
            page_number=2,
        )
        assert val_chunk.validate_citation(cite).is_valid is True

        # Validate with RetrievalResult in pool
        val_retrieval = CitationValidator(evidence_pool=[retrieval_res])
        assert val_retrieval.validate_citation(cite).is_valid is True

        # Validate with HybridRetrievalResult in pool
        val_hybrid = CitationValidator(evidence_pool=[hybrid_res])
        assert val_hybrid.validate_citation(cite).is_valid is True

    def test_unsupported_evidence_type_raises_type_error(self) -> None:
        """Verifies loading an unsupported object into evidence pool raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported evidence type"):
            CitationValidator(evidence_pool=["invalid object string"])  # type: ignore

    def test_privacy_preservation_in_error_messages(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies error reasons and reports never leak secret/private document text."""
        # Evidence passage contains sensitive content
        secret_evidence = EvidencePassage(
            text="TOP_SECRET_PASSPHRASE_ALPHA_OMEGA_9999",
            doc_id="classified_doc",
            filename="classified.pdf",
            chunk_id="classified_doc_p1_c0",
            page_number=1,
        )
        validator = CitationValidator(evidence_pool=[secret_evidence])

        # Citation with wrong page number
        cite_wrong_page = CitationReference(
            citation_id="[1]",
            doc_id="classified_doc",
            filename="classified.pdf",
            chunk_id="classified_doc_p1_c0",
            page_number=2,
        )
        res = validator.validate_citation(cite_wrong_page)

        # Assert secret text is absent from reason
        assert "TOP_SECRET" not in str(res.reason)
        assert "PASSPHRASE" not in str(res.reason)

    def test_batch_validation_report_aggregation_and_serialization(
        self,
        synthetic_evidence_pool: list[EvidencePassage],
    ) -> None:
        """Verifies batch validation report metrics, lists, and to_dict serialization."""
        valid_cite = CitationReference(
            citation_id="[1]",
            doc_id="synth_doc_pdf",
            filename="quantum_review.pdf",
            chunk_id="synth_doc_pdf_p3_c0",
            page_number=3,
        )
        invalid_cite = CitationReference(
            citation_id="[2]",
            doc_id="non_existent_doc",
            filename="fake.pdf",
            chunk_id="fake_p1_c0",
            page_number=1,
        )

        report = validate_citations([valid_cite, invalid_cite], synthetic_evidence_pool)

        assert report.total_citations == 2
        assert report.valid_count == 1
        assert report.invalid_count == 1
        assert report.is_all_valid is False
        assert len(report.valid_citations) == 1
        assert len(report.invalid_citations) == 1
        assert "Citation validation failed: 1/2 valid" in report.summary()

        # Serialization
        data = report.to_dict()
        assert data["total_citations"] == 2
        assert data["valid_count"] == 1
        assert data["invalid_count"] == 1
        assert len(data["results"]) == 2
        assert data["results"][0]["is_valid"] is True
        assert data["results"][1]["is_valid"] is False

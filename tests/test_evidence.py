"""
Tests for Phase 4 Typed Evidence, Citation, and Verification Models.
Verifies model creation, validations, edge cases, and TXT/PDF provenance preservation.
"""

from dataclasses import FrozenInstanceError
import pytest

from verityai.chunking import ChunkMetadata, DocumentChunk
from verityai.evidence import (
    Citation,
    CitationReference,
    EvidencePassage,
    VerificationOutcome,
    VerificationResult,
    VerificationStatus,
)
from verityai.retrieval_hybrid import HybridRetrievalResult
from verityai.retrieval_semantic import RetrievalResult


# ==============================================================================
# EvidencePassage Tests
# ==============================================================================


class TestEvidencePassage:
    """Test suite for EvidencePassage model."""

    def test_valid_pdf_evidence(self) -> None:
        """Verifies creating an EvidencePassage with a valid PDF page number."""
        passage = EvidencePassage(
            text="Quantum computing utilizes qubits for superposition.",
            doc_id="doc_123",
            filename="quantum_paper.pdf",
            chunk_id="doc_123_p3_c0",
            page_number=3,
            score=0.912,
            rank=1,
            doc_type="pdf",
        )
        assert passage.text == "Quantum computing utilizes qubits for superposition."
        assert passage.doc_id == "doc_123"
        assert passage.filename == "quantum_paper.pdf"
        assert passage.chunk_id == "doc_123_p3_c0"
        assert passage.page_number == 3
        assert passage.score == 0.912
        assert passage.rank == 1
        assert passage.doc_type == "pdf"

        # Verify serialization
        data = passage.to_dict()
        assert data["page_number"] == 3
        assert data["chunk_id"] == "doc_123_p3_c0"
        assert data["rank"] == 1

    def test_valid_txt_evidence_without_page_number(self) -> None:
        """Verifies creating an EvidencePassage without a page number for plain text."""
        passage = EvidencePassage(
            text="Raw lab measurements recorded at ambient room temperature.",
            doc_id="doc_456",
            filename="lab_notes.txt",
            chunk_id="doc_456_c0",
            page_number=None,
            score=0.78,
            rank=2,
            doc_type="txt",
        )
        assert passage.page_number is None
        assert passage.filename == "lab_notes.txt"
        assert passage.to_dict()["page_number"] is None

    def test_missing_required_fields_raises(self) -> None:
        """Verifies missing required positional arguments raises TypeError."""
        with pytest.raises(TypeError):
            EvidencePassage()  # type: ignore

        with pytest.raises(TypeError):
            EvidencePassage(text="Valid text")  # type: ignore

    @pytest.mark.parametrize("field_name", ["text", "doc_id", "filename", "chunk_id"])
    @pytest.mark.parametrize("bad_val", ["", "   ", "\n\t"])
    def test_empty_required_strings_raise_value_error(self, field_name: str, bad_val: str) -> None:
        """Verifies empty or whitespace-only strings raise ValueError."""
        kwargs = {
            "text": "Valid text",
            "doc_id": "doc_1",
            "filename": "file.pdf",
            "chunk_id": "doc_1_c0",
        }
        kwargs[field_name] = bad_val
        with pytest.raises(ValueError, match=f"{field_name}|Evidence text"):
            EvidencePassage(**kwargs)

    @pytest.mark.parametrize("bad_page", [0, -1, -99])
    def test_invalid_page_numbers_raise_value_error(self, bad_page: int) -> None:
        """Verifies page numbers <= 0 raise ValueError."""
        with pytest.raises(ValueError, match="page_number must be a positive integer"):
            EvidencePassage(
                text="Some text",
                doc_id="doc_1",
                filename="file.pdf",
                chunk_id="doc_1_c0",
                page_number=bad_page,
            )

    @pytest.mark.parametrize("bad_type_page", ["1", 3.14, True, [1]])
    def test_invalid_page_number_types_raise_type_error(self, bad_type_page: object) -> None:
        """Verifies non-integer page numbers raise TypeError."""
        with pytest.raises(TypeError, match="page_number must be an integer"):
            EvidencePassage(
                text="Some text",
                doc_id="doc_1",
                filename="file.pdf",
                chunk_id="doc_1_c0",
                page_number=bad_type_page,  # type: ignore
            )

    @pytest.mark.parametrize("bad_rank", [0, -1])
    def test_invalid_rank_raises_value_error(self, bad_rank: int) -> None:
        """Verifies rank <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="rank must be a positive integer"):
            EvidencePassage(
                text="Some text",
                doc_id="doc_1",
                filename="file.pdf",
                chunk_id="doc_1_c0",
                rank=bad_rank,
            )

    def test_invalid_score_type_raises_type_error(self) -> None:
        """Verifies non-numeric score raises TypeError."""
        with pytest.raises(TypeError, match="score must be a numeric"):
            EvidencePassage(
                text="Some text",
                doc_id="doc_1",
                filename="file.pdf",
                chunk_id="doc_1_c0",
                score="high",  # type: ignore
            )

    def test_evidence_passage_immutability(self) -> None:
        """Verifies EvidencePassage cannot be mutated after creation."""
        passage = EvidencePassage(
            text="Initial text",
            doc_id="doc_1",
            filename="file.txt",
            chunk_id="doc_1_c0",
        )
        with pytest.raises(FrozenInstanceError):
            passage.text = "Changed text"  # type: ignore

    def test_from_chunk_pdf(self) -> None:
        """Verifies constructing from a PDF DocumentChunk preserves 1-indexed page."""
        chunk = DocumentChunk(
            text="Architecture specifications for distributed consensus.",
            metadata=ChunkMetadata(
                chunk_id="doc_789_p4_c1",
                doc_id="doc_789",
                filename="system_design.pdf",
                doc_type="pdf",
                page_number=4,
                chunk_index=1,
                char_count=55,
                word_count=6,
            ),
        )
        evidence = EvidencePassage.from_chunk(chunk, score=0.88, rank=1)
        assert evidence.text == chunk.text
        assert evidence.doc_id == "doc_789"
        assert evidence.filename == "system_design.pdf"
        assert evidence.chunk_id == "doc_789_p4_c1"
        assert evidence.page_number == 4
        assert evidence.score == 0.88
        assert evidence.rank == 1
        assert evidence.doc_type == "pdf"

    def test_from_chunk_txt_normalizes_page_to_none(self) -> None:
        """Verifies constructing from a TXT DocumentChunk normalizes page_number to None."""
        chunk = DocumentChunk(
            text="Simple README notes without page geometry.",
            metadata=ChunkMetadata(
                chunk_id="doc_999_c0",
                doc_id="doc_999",
                filename="instructions.txt",
                doc_type="txt",
                page_number=1,
                chunk_index=0,
                char_count=42,
                word_count=6,
            ),
        )
        evidence = EvidencePassage.from_chunk(chunk, score=0.65, rank=3)
        assert evidence.page_number is None
        assert evidence.filename == "instructions.txt"
        assert evidence.doc_type == "txt"

    def test_from_retrieval_result(self) -> None:
        """Verifies constructing from a semantic RetrievalResult."""
        chunk = DocumentChunk(
            text="Neural network weights quantization guide.",
            metadata=ChunkMetadata(
                chunk_id="doc_111_p2_c0",
                doc_id="doc_111",
                filename="ml_guide.pdf",
                doc_type="pdf",
                page_number=2,
                chunk_index=0,
                char_count=42,
                word_count=5,
            ),
        )
        result = RetrievalResult(chunk=chunk, score=0.95, rank=1)
        evidence = EvidencePassage.from_retrieval_result(result)
        assert evidence.chunk_id == "doc_111_p2_c0"
        assert evidence.score == 0.95
        assert evidence.rank == 1
        assert evidence.page_number == 2

    def test_from_hybrid_result(self) -> None:
        """Verifies constructing from a HybridRetrievalResult."""
        chunk = DocumentChunk(
            text="Hybrid fusion combining lexical and semantic vectors.",
            metadata=ChunkMetadata(
                chunk_id="doc_222_p5_c2",
                doc_id="doc_222",
                filename="fusion_paper.pdf",
                doc_type="pdf",
                page_number=5,
                chunk_index=2,
                char_count=53,
                word_count=7,
            ),
        )
        hybrid_res = HybridRetrievalResult(
            chunk=chunk,
            rrf_score=0.032,
            rank=1,
            semantic_rank=2,
            keyword_rank=1,
        )
        evidence = EvidencePassage.from_hybrid_result(hybrid_res)
        assert evidence.chunk_id == "doc_222_p5_c2"
        assert evidence.score == 0.032
        assert evidence.rank == 1
        assert evidence.page_number == 5


# ==============================================================================
# CitationReference Tests
# ==============================================================================


class TestCitationReference:
    """Test suite for CitationReference model."""

    def test_valid_pdf_citation(self) -> None:
        """Verifies creating a citation reference for a PDF source."""
        cite = CitationReference(
            citation_id="1",
            doc_id="doc_123",
            filename="climate_report.pdf",
            chunk_id="doc_123_p12_c1",
            page_number=12,
            quote="Global surface temperature increased by 1.1 degrees Celsius.",
            rank=1,
            score=0.94,
        )
        assert cite.citation_id == "1"
        assert cite.filename == "climate_report.pdf"
        assert cite.page_number == 12
        assert cite.format_inline() == "[climate_report.pdf, p. 12]"
        assert "p. 12" in cite.format_reference()
        assert "Global surface temperature" in cite.format_reference()

    def test_valid_txt_citation_without_page(self) -> None:
        """Verifies citation for TXT documents renders cleanly without page numbers."""
        cite = CitationReference(
            citation_id="2",
            doc_id="doc_456",
            filename="meeting_notes.txt",
            chunk_id="doc_456_c0",
            page_number=None,
            quote="Project milestone agreed for end of Q4.",
        )
        assert cite.page_number is None
        assert cite.format_inline() == "[meeting_notes.txt]"
        assert "p." not in cite.format_reference()
        assert cite.format_reference() == '[2] meeting_notes.txt: "Project milestone agreed for end of Q4."'

    def test_citation_alias_equivalence(self) -> None:
        """Verifies Citation is an identical alias for CitationReference."""
        assert Citation is CitationReference

    def test_from_evidence_factory(self) -> None:
        """Verifies building a CitationReference from an EvidencePassage."""
        evidence = EvidencePassage(
            text="Decentralized consensus requires fault tolerance.",
            doc_id="doc_abc",
            filename="consensus.pdf",
            chunk_id="doc_abc_p7_c0",
            page_number=7,
            score=0.87,
            rank=2,
            doc_type="pdf",
        )
        cite = CitationReference.from_evidence(evidence, citation_id="ref-1", quote="fault tolerance")
        assert cite.citation_id == "ref-1"
        assert cite.filename == "consensus.pdf"
        assert cite.page_number == 7
        assert cite.quote == "fault tolerance"
        assert cite.rank == 2
        assert cite.score == 0.87

    @pytest.mark.parametrize("field_name", ["citation_id", "doc_id", "filename", "chunk_id"])
    def test_empty_citation_strings_raise_value_error(self, field_name: str) -> None:
        """Verifies empty strings for required citation fields raise ValueError."""
        kwargs = {
            "citation_id": "1",
            "doc_id": "doc_1",
            "filename": "doc.pdf",
            "chunk_id": "doc_1_c0",
        }
        kwargs[field_name] = ""
        with pytest.raises(ValueError):
            CitationReference(**kwargs)

    def test_invalid_citation_page_number_raises(self) -> None:
        """Verifies page numbers <= 0 raise ValueError."""
        with pytest.raises(ValueError, match="page_number must be a positive integer"):
            CitationReference(
                citation_id="1",
                doc_id="doc_1",
                filename="doc.pdf",
                chunk_id="doc_1_c0",
                page_number=0,
            )

    def test_citation_immutability(self) -> None:
        """Verifies CitationReference is immutable."""
        cite = CitationReference(
            citation_id="1",
            doc_id="doc_1",
            filename="doc.pdf",
            chunk_id="doc_1_c0",
        )
        with pytest.raises(FrozenInstanceError):
            cite.filename = "new_doc.pdf"  # type: ignore


# ==============================================================================
# VerificationOutcome Tests
# ==============================================================================


class TestVerificationOutcome:
    """Test suite for VerificationOutcome model."""

    @pytest.mark.parametrize(
        "status, is_ver, is_contra, is_unver",
        [
            (VerificationStatus.SUPPORTED, True, False, False),
            (VerificationStatus.CONTRADICTED, False, True, False),
            (VerificationStatus.PARTIALLY_SUPPORTED, False, False, False),
            (VerificationStatus.UNVERIFIED, False, False, True),
            (VerificationStatus.INSUFFICIENT_EVIDENCE, False, False, True),
            (VerificationStatus.UNVERIFIABLE, False, False, True),
        ],
    )
    def test_verification_statuses_and_properties(
        self,
        status: VerificationStatus,
        is_ver: bool,
        is_contra: bool,
        is_unver: bool,
    ) -> None:
        """Verifies all verification status values and property booleans."""
        outcome = VerificationOutcome(
            claim="The server cluster runs on Ubuntu Linux.",
            status=status,
            confidence=0.95,
            explanation="Cross-referenced against infrastructure documentation.",
        )
        assert outcome.status == status
        assert outcome.is_verified is is_ver
        assert outcome.is_contradicted is is_contra
        assert outcome.is_unverified is is_unver

    def test_insufficient_evidence_and_unverifiable_specific_properties(self) -> None:
        """Verifies is_insufficient_evidence and is_unverifiable properties."""
        outcome_insuff = VerificationOutcome(
            claim="Factual claim.",
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        )
        assert outcome_insuff.is_insufficient_evidence is True
        assert outcome_insuff.is_unverifiable is False
        assert outcome_insuff.is_unverified is True

        outcome_unver = VerificationOutcome(
            claim="",
            status=VerificationStatus.UNVERIFIABLE,
        )
        assert outcome_unver.is_unverifiable is True
        assert outcome_unver.is_insufficient_evidence is False
        assert outcome_unver.is_unverified is True

    def test_verification_result_alias(self) -> None:
        """Verifies VerificationResult is an alias for VerificationOutcome."""
        assert VerificationResult is VerificationOutcome

    def test_string_status_auto_conversion(self) -> None:
        """Verifies case-insensitive string status inputs are converted to Enum."""
        outcome1 = VerificationOutcome(
            claim="Valid claim.",
            status="supported",  # type: ignore
        )
        assert outcome1.status == VerificationStatus.SUPPORTED

        outcome2 = VerificationOutcome(
            claim="Another claim.",
            status="CONTRADICTED",  # type: ignore
        )
        assert outcome2.status == VerificationStatus.CONTRADICTED

    def test_invalid_status_raises_value_error(self) -> None:
        """Verifies unrecognized status strings raise ValueError."""
        with pytest.raises(ValueError, match="Invalid status"):
            VerificationOutcome(
                claim="Some claim.",
                status="plausible",  # type: ignore
            )

    def test_invalid_status_type_raises_type_error(self) -> None:
        """Verifies non-string and non-enum status raises TypeError."""
        with pytest.raises(TypeError, match="status must be a VerificationStatus or str"):
            VerificationOutcome(
                claim="Some claim.",
                status=123,  # type: ignore
            )

    @pytest.mark.parametrize("bad_claim", ["", "   ", "\n"])
    def test_empty_claim_raises_value_error(self, bad_claim: str) -> None:
        """Verifies empty or whitespace claim raises ValueError."""
        with pytest.raises(ValueError, match="claim must be a non-empty string"):
            VerificationOutcome(claim=bad_claim, status=VerificationStatus.UNVERIFIED)

    @pytest.mark.parametrize("valid_conf", [0.0, 0.5, 1.0])
    def test_valid_confidence_scores(self, valid_conf: float) -> None:
        """Verifies confidence between 0.0 and 1.0 is accepted."""
        outcome = VerificationOutcome(
            claim="Reliable claim.",
            status=VerificationStatus.SUPPORTED,
            confidence=valid_conf,
        )
        assert outcome.confidence == valid_conf

    @pytest.mark.parametrize("invalid_conf", [-0.01, 1.01, 5.0])
    def test_invalid_confidence_raises_value_error(self, invalid_conf: float) -> None:
        """Verifies out-of-bounds confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            VerificationOutcome(
                claim="Claim with bad confidence.",
                status=VerificationStatus.SUPPORTED,
                confidence=invalid_conf,
            )

    def test_invalid_confidence_type_raises_type_error(self) -> None:
        """Verifies non-numeric confidence raises TypeError."""
        with pytest.raises(TypeError, match="confidence must be a float or int"):
            VerificationOutcome(
                claim="Claim.",
                status=VerificationStatus.SUPPORTED,
                confidence="high",  # type: ignore
            )

    def test_outcome_with_evidence_and_citations(self) -> None:
        """Verifies full outcome lifecycle with linked evidence and citations."""
        evidence = EvidencePassage(
            text="The speed of light in vacuum is approximately 299,792 km/s.",
            doc_id="physics_doc",
            filename="physics_constants.pdf",
            chunk_id="physics_doc_p1_c0",
            page_number=1,
            score=0.99,
            rank=1,
        )
        cite = CitationReference.from_evidence(evidence, citation_id="[1]")

        outcome = VerificationOutcome(
            claim="Light travels at roughly 300,000 km/s in a vacuum.",
            status=VerificationStatus.SUPPORTED,
            evidence=[evidence],
            citations=[cite],
            confidence=0.99,
            explanation="Supported by physical constant documentation.",
        )

        assert outcome.is_verified is True
        assert len(outcome.evidence) == 1
        assert len(outcome.citations) == 1
        assert outcome.evidence[0].filename == "physics_constants.pdf"
        assert outcome.citations[0].citation_id == "[1]"

        # Verify to_dict serialization
        data = outcome.to_dict()
        assert data["is_verified"] is True
        assert data["status"] == "supported"
        assert len(data["evidence"]) == 1
        assert len(data["citations"]) == 1

    def test_invalid_evidence_type_raises_type_error(self) -> None:
        """Verifies passing non-EvidencePassage items in evidence raises TypeError."""
        with pytest.raises(TypeError, match="Each item in evidence must be an EvidencePassage"):
            VerificationOutcome(
                claim="Some claim.",
                status=VerificationStatus.SUPPORTED,
                evidence=["not an evidence passage"],  # type: ignore
            )

    def test_invalid_citation_type_raises_type_error(self) -> None:
        """Verifies passing non-CitationReference items in citations raises TypeError."""
        with pytest.raises(TypeError, match="Each item in citations must be a CitationReference"):
            VerificationOutcome(
                claim="Some claim.",
                status=VerificationStatus.SUPPORTED,
                citations=["not a citation"],  # type: ignore
            )

    def test_verification_outcome_immutability(self) -> None:
        """Verifies VerificationOutcome cannot be mutated after creation."""
        outcome = VerificationOutcome(
            claim="Immutable claim.",
            status=VerificationStatus.SUPPORTED,
        )
        with pytest.raises(FrozenInstanceError):
            outcome.claim = "Mutated claim."  # type: ignore

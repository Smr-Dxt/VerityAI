"""
VerityAI - Research Workbench Application
Evidence-based, offline-first research assistant.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure the 'src' directory is on sys.path for direct imports
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from verityai.config import get_config
from verityai.ingestion import DocumentResult, safe_ingest_document
from verityai.chunking import chunk_document, DocumentChunk
from verityai.embeddings import BaseEmbedder, LocalEmbedder, MockEmbedder
from verityai.retrieval_semantic import FAISSRetriever
from verityai.retrieval_hybrid import BM25Retriever, HybridRetriever, HybridRetrievalResult
from typing import Optional


@st.cache_resource
def get_app_embedder() -> BaseEmbedder:
    """Provides singleton embedding instance (MockEmbedder in test mode, LocalEmbedder in production)."""
    if os.getenv("VERITY_TEST_MODE") == "1" or os.getenv("VERITY_USE_MOCK_EMBEDDER") == "1":
        return MockEmbedder(dimension=64)
    return LocalEmbedder()


def get_or_create_hybrid_retriever(active_doc: DocumentResult) -> Optional[HybridRetriever]:
    """Retrieves or builds the in-memory hybrid retriever for the active document."""
    doc_key = f"{active_doc.filename}:{active_doc.total_pages}:{active_doc.char_count}"
    if (
        st.session_state.get("hybrid_retriever") is not None
        and st.session_state.get("retriever_doc_key") == doc_key
    ):
        return st.session_state["hybrid_retriever"]

    chunks = chunk_document(active_doc)
    if not chunks:
        return None

    embedder = get_app_embedder()
    sem_retriever = FAISSRetriever(embedder=embedder)
    bm25_retriever = BM25Retriever()
    hybrid = HybridRetriever(
        semantic_retriever=sem_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
    )
    hybrid.index_chunks(chunks)

    st.session_state["hybrid_retriever"] = hybrid
    st.session_state["active_chunks"] = chunks
    st.session_state["retriever_doc_key"] = doc_key
    return hybrid


def inject_custom_styles() -> None:
    """Injects restrained, professional CSS for a research-software aesthetic.

    Emphasizes solid surfaces, clear typography, deliberate contrast, and
    restrained accents. Avoids generic AI dashboard tropes such as glassmorphism,
    translucent cards, neon glows, and excessive gradients.
    """
    st.markdown(
        """
        <style>
        /* Base page layout */
        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #090D16;
            border-right: 1px solid #1E293B;
        }

        /* Brand title & typography */
        .verity-brand-title {
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #F8FAFC;
            margin: 0;
            line-height: 1.2;
        }
        .verity-tagline {
            font-size: 0.95rem;
            color: #94A3B8;
            margin-top: 0.35rem;
            margin-bottom: 1.5rem;
        }

        /* Solid, high-contrast card surfaces */
        .verity-card {
            background-color: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .verity-card-header {
            font-size: 1rem;
            font-weight: 600;
            color: #F1F5F9;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .verity-card-body {
            font-size: 0.9rem;
            color: #94A3B8;
            line-height: 1.6;
        }

        /* Empty state workspace container */
        .empty-workspace-box {
            background-color: #0B1120;
            border: 1px dashed #334155;
            border-radius: 6px;
            padding: 2.75rem 2rem;
            text-align: center;
            margin: 1.25rem 0 1.5rem 0;
        }
        .empty-workspace-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #E2E8F0;
            margin-bottom: 0.4rem;
        }
        .empty-workspace-desc {
            font-size: 0.92rem;
            color: #94A3B8;
            max-width: 620px;
            margin: 0 auto 1.25rem auto;
            line-height: 1.6;
        }

        /* Restrained status badges */
        .verity-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.65rem;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            background-color: #1E293B;
            color: #94A3B8;
            border: 1px solid #334155;
        }
        .verity-badge-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #10B981;
        }

        /* Technical spec row styling */
        .spec-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 0;
            border-bottom: 1px solid #1E293B;
            font-size: 0.88rem;
        }
        .spec-row:last-child {
            border-bottom: none;
        }
        .spec-label {
            color: #94A3B8;
            font-weight: 500;
        }
        .spec-value {
            color: #F8FAFC;
            font-family: monospace;
            font-weight: 600;
        }

        /* Subtle section headers */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #F1F5F9;
            margin: 1.25rem 0 0.75rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config) -> str:
    """Renders the product-oriented navigation sidebar and system indicator."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding-bottom: 0.85rem; border-bottom: 1px solid #1E293B; margin-bottom: 1.25rem;">
                <h2 style="margin: 0; font-size: 1.4rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">VerityAI</h2>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.2rem;">Evidence-Based Research</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("NAVIGATION")
        nav_choice = st.radio(
            "Navigation Menu",
            options=[
                "Research Workspace",
                "Documents",
                "Verification",
                "Model Evaluation",
                "Settings",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown(
            """
            <div style="padding: 0.5rem 0;">
                <div style="display: flex; align-items: center; gap: 0.45rem; font-size: 0.8rem; color: #CBD5E1; font-weight: 500;">
                    <span class="verity-badge-dot"></span>
                    <span>Local Engine Ready</span>
                </div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.25rem;">
                    Offline-First Architecture
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"VerityAI v{config.version}")

    return nav_choice


def render_research_workspace(config) -> None:
    """Renders the primary research workspace view."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1E293B;">
            <h1 class="verity-brand-title">{config.app_name}</h1>
            <p class="verity-tagline">{config.tagline}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Document Staging / Corpus Status
    st.markdown('<div class="section-header">Document Corpus</div>', unsafe_allow_html=True)

    active_doc = st.session_state.get("active_document")
    if active_doc is not None and active_doc.status == "success":
        st.markdown(
            f"""
            <div class="verity-card">
                <div class="verity-card-header">
                    <span>Active Ingested Document</span>
                    <span class="verity-badge"><span class="verity-badge-dot"></span> Ingested</span>
                </div>
                <div class="spec-row"><span class="spec-label">Document:</span><span class="spec-value">{active_doc.filename}</span></div>
                <div class="spec-row"><span class="spec-label">Type:</span><span class="spec-value">{active_doc.doc_type.upper()} ({active_doc.total_pages} page{'s' if active_doc.total_pages != 1 else ''})</span></div>
                <div class="spec-row"><span class="spec-label">Characters:</span><span class="spec-value">{active_doc.char_count:,}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="empty-workspace-box">
                <div class="empty-workspace-title">No Documents Ingested</div>
                <div class="empty-workspace-desc">
                    Your research repository is currently empty. Ingest technical papers or plain-text
                    documentation in the Documents tab to inspect text and verify page metadata.
                </div>
                <span class="verity-badge">
                    <span class="verity-badge-dot"></span>
                    Local Ingestion Engine Awaiting Files
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Evidence Query Area
    st.markdown('<div class="section-header">Evidence Retrieval</div>', unsafe_allow_html=True)

    if active_doc is not None and active_doc.status == "success":
        st.caption(
            "Query the active document using hybrid retrieval (FAISS semantic vectors + BM25 keyword matching fused via RRF)."
        )

        query = st.text_input(
            "Enter a research query across your document:",
            placeholder="e.g., What are the documented tradeoffs between quantized attention heads?",
            key="evidence_query_input",
            help="Queries are executed offline against in-memory FAISS and BM25 indices.",
        )

        col_btn, col_k = st.columns([4, 1])
        with col_btn:
            search_submitted = st.button("Retrieve Evidence", key="btn_search_evidence", type="primary")
        with col_k:
            top_k = st.selectbox("Top Passages:", options=[3, 5, 10], index=0, key="select_top_k")

        # Execute search if query provided and button clicked or enter pressed
        if query.strip():
            with st.spinner("Executing hybrid retrieval..."):
                retriever = get_or_create_hybrid_retriever(active_doc)

            if retriever is None or retriever.num_indexed == 0:
                st.warning("No searchable text chunks could be extracted from this document.")
            else:
                results = retriever.retrieve(query.strip(), top_k=top_k)

                st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                if results:
                    st.markdown(
                        f'<div class="section-header">Retrieved Evidence Passages ({len(results)} matches)</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Passages ranked by Reciprocal Rank Fusion (RRF). Scores represent positional consensus "
                        "between BM25 lexical and FAISS semantic search, not a probability."
                    )

                    for res in results:
                        page_label = (
                            f"Page {res.chunk.metadata.page_number}"
                            if res.chunk.metadata.page_number is not None
                            else "Text Document"
                        )
                        sem_rank_str = f"#{res.semantic_rank}" if res.semantic_rank is not None else "—"
                        kw_rank_str = f"#{res.keyword_rank}" if res.keyword_rank is not None else "—"

                        st.markdown(
                            f"""
                            <div class="verity-card">
                                <div class="verity-card-header">
                                    <span>Rank #{res.rank} — {res.chunk.metadata.filename}</span>
                                    <span class="verity-badge"><span class="verity-badge-dot"></span> {page_label}</span>
                                </div>
                                <div class="spec-row">
                                    <span class="spec-label">Consensus Score (RRF):</span>
                                    <span class="spec-value">{res.rrf_score:.4f}</span>
                                </div>
                                <div class="spec-row">
                                    <span class="spec-label">Channel Ranks:</span>
                                    <span class="spec-value">Semantic {sem_rank_str} | Keyword {kw_rank_str}</span>
                                </div>
                                <div class="spec-row">
                                    <span class="spec-label">Chunk ID:</span>
                                    <span class="spec-value">{res.chunk.metadata.chunk_id}</span>
                                </div>
                                <div style="margin-top: 0.85rem; padding: 0.85rem 1rem; background-color: #0B1120; border-radius: 4px; border: 1px solid #1E293B; font-size: 0.9rem; line-height: 1.6; color: #E2E8F0; white-space: pre-wrap;">{res.chunk.text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info(f"No relevant passages found matching '{query.strip()}'. Try different keywords or broader phrasing.")
    else:
        st.text_input(
            "Enter a research question across your documents:",
            placeholder="e.g., What are the documented tradeoffs between quantized attention heads?",
            disabled=True,
            help="Upload a technical document in the Documents tab to enable evidence retrieval.",
        )
        st.caption("Upload a technical paper or plain-text notes in the Documents tab to enable evidence retrieval.")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Core Research Architecture Cards
    st.markdown('<div class="section-header">Research Capabilities</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="verity-card">
                <div class="verity-card-header">
                    <span>Citation Grounding</span>
                </div>
                <div class="verity-card-body">
                    Every synthesized finding is strictly grounded in exact document passages with verifiable source citations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="verity-card">
                <div class="verity-card-header">
                    <span>Contradiction Detection</span>
                </div>
                <div class="verity-card-body">
                    Cross-references multiple evidence sources to surface conflicting claims, methodological discrepancies, and consensus.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="verity-card">
                <div class="verity-card-header">
                    <span>Local-Only Privacy</span>
                </div>
                <div class="verity-card-body">
                    Engineered for offline, zero-telemetry execution. Your documents, queries, and embeddings never leave your machine.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_documents_view(config) -> None:
    """Renders the document management and local ingestion workspace."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1E293B;">
            <h1 class="verity-brand-title">Documents</h1>
            <p class="verity-tagline">Locally ingest, validate, and inspect research documents (.txt, .pdf).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_mb = config.max_file_size_bytes // (1024 * 1024)

    # Document Upload Control
    st.markdown('<div class="section-header">Document Ingestion</div>', unsafe_allow_html=True)
    st.caption(f"Supported formats: Plain Text (.txt), PDF (.pdf) | File limit: {max_mb} MB, up to {config.max_pdf_pages} pages")

    uploaded_file = st.file_uploader(
        "Upload a technical paper or text file for local ingestion:",
        type=["txt", "pdf"],
        key="doc_uploader",
        help=f"Uploads are processed purely in-memory up to {max_mb} MB.",
    )

    # Handle Upload Processing
    if uploaded_file is not None:
        last_name = st.session_state.get("active_doc_filename")
        if last_name != uploaded_file.name:
            file_bytes = uploaded_file.getvalue()
            doc_result = safe_ingest_document(
                file_content=file_bytes,
                filename=uploaded_file.name,
                max_size_bytes=config.max_file_size_bytes,
                max_pages=config.max_pdf_pages,
            )
            st.session_state["active_document"] = doc_result
            st.session_state["active_doc_filename"] = uploaded_file.name

    # Render Current Ingestion State
    active_doc: DocumentResult | None = st.session_state.get("active_document")

    if active_doc is not None:
        if active_doc.status == "error":
            st.error(f"Ingestion Failure: {active_doc.error_message}")
            if st.button("Clear and Try Another File", key="btn_clear_error"):
                st.session_state["active_document"] = None
                st.session_state["active_doc_filename"] = None
                st.rerun()
        else:
            st.success(f"Successfully extracted content from '{active_doc.filename}' locally.")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(
                    f"""
                    <div class="verity-card">
                        <div class="verity-card-header">
                            <span>Document Summary</span>
                            <span class="verity-badge"><span class="verity-badge-dot"></span> Ingested</span>
                        </div>
                        <div class="spec-row"><span class="spec-label">Filename:</span><span class="spec-value">{active_doc.filename}</span></div>
                        <div class="spec-row"><span class="spec-label">Format:</span><span class="spec-value">{active_doc.doc_type.upper()}</span></div>
                        <div class="spec-row"><span class="spec-label">Pages:</span><span class="spec-value">{active_doc.total_pages}</span></div>
                        <div class="spec-row"><span class="spec-label">Total Characters:</span><span class="spec-value">{active_doc.char_count:,}</span></div>
                        <div class="spec-row"><span class="spec-label">Storage:</span><span class="spec-value">In-Memory (Offline)</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    """
                    <div class="verity-card">
                        <div class="verity-card-header"><span>Pipeline Status</span></div>
                        <div class="verity-card-body">
                            Document text is chunked and indexed into in-memory FAISS dense vector and BM25 lexical stores.
                            Evidence retrieval is active in the Research Workspace.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Reset / Ingest New Document", key="btn_reset_doc"):
                    st.session_state["active_document"] = None
                    st.session_state["active_doc_filename"] = None
                    st.session_state["hybrid_retriever"] = None
                    st.session_state["active_chunks"] = None
                    st.session_state["retriever_doc_key"] = None
                    st.rerun()

            # Extracted Text Inspector
            st.markdown('<div class="section-header">Document Content Inspector</div>', unsafe_allow_html=True)

            if active_doc.doc_type == "pdf" and active_doc.total_pages > 1:
                page_labels = [f"Page {p.page_number} ({p.char_count:,} chars)" for p in active_doc.pages]
                page_idx = st.selectbox(
                    "Select page to inspect:",
                    options=list(range(len(active_doc.pages))),
                    format_func=lambda i: page_labels[i],
                    key="pdf_page_select",
                )
                chosen_page = active_doc.pages[page_idx]
                st.caption(f"Page {chosen_page.page_number} of {active_doc.total_pages} — {chosen_page.char_count:,} characters")
                st.text_area(
                    label=f"Page {chosen_page.page_number} Content",
                    value=chosen_page.text,
                    height=320,
                    disabled=True,
                    label_visibility="collapsed",
                )
            else:
                st.caption(f"Full Extracted Text — {active_doc.char_count:,} characters")
                st.text_area(
                    label="Extracted Text Content",
                    value=active_doc.extracted_text,
                    height=320,
                    disabled=True,
                    label_visibility="collapsed",
                )
    else:
        st.markdown(
            """
            <div class="empty-workspace-box">
                <div class="empty-workspace-title">No Document Loaded</div>
                <div class="empty-workspace-desc">
                    Upload a plain-text (.txt) research note or PDF (.pdf) technical paper above to extract text,
                    inspect page boundaries, and verify document integrity.
                </div>
                <span class="verity-badge">Ingestion Engine Ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">Supported Document Formats</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="verity-card">
            <div class="verity-card-body">
                Phase 2 supports <strong>PDF (.pdf)</strong> documents with full page-boundary preservation and
                <strong>Plain Text (.txt)</strong> documents. Scanned image-only PDFs and DOCX files are planned
                for future phases. All text extraction is executed locally on-device without cloud transmission.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verification_view() -> None:
    """Renders the citation and verification view."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1E293B;">
            <h1 class="verity-brand-title">Verification Matrix</h1>
            <p class="verity-tagline">Audit citation provenance, confidence scores, and cross-document contradictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="empty-workspace-box">
            <div class="empty-workspace-title">No Active Verification Graph</div>
            <div class="empty-workspace-desc">
                The verification matrix activates during active research queries to evaluate whether retrieved
                passages directly entail synthesized answers.
            </div>
            <span class="verity-badge">Standby: Awaiting Evidence Queries</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_eval_view() -> None:
    """Renders the model evaluation and benchmark view."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1E293B;">
            <h1 class="verity-brand-title">Model Evaluation</h1>
            <p class="verity-tagline">Benchmark local model inference, citation fidelity, and latency on your hardware.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="empty-workspace-box">
            <div class="empty-workspace-title">Model Evaluation Standby</div>
            <div class="empty-workspace-desc">
                Benchmark and compare local open-source models (such as quantized 3B–7B parameter models)
                for research retrieval and citation precision without cloud APIs.
            </div>
            <span class="verity-badge">Compute Profile: Local GPU Acceleration</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_settings_view(config) -> None:
    """Renders system configuration, hardware target profile, and diagnostics."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1E293B;">
            <h1 class="verity-brand-title">Settings & System Status</h1>
            <p class="verity-tagline">Runtime configuration, hardware target profile, and system diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Application & Environment</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="verity-card">
                <div class="spec-row"><span class="spec-label">Application:</span><span class="spec-value">{config.app_name}</span></div>
                <div class="spec-row"><span class="spec-label">Version:</span><span class="spec-value">v{config.version}</span></div>
                <div class="spec-row"><span class="spec-label">Environment:</span><span class="spec-value">{config.environment}</span></div>
                <div class="spec-row"><span class="spec-label">Server Address:</span><span class="spec-value">{config.host}:{config.port}</span></div>
                <div class="spec-row"><span class="spec-label">Architecture:</span><span class="spec-value">Offline-First</span></div>
                <div class="spec-row"><span class="spec-label">Log Level:</span><span class="spec-value">{config.log_level}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-header">Target Hardware Profile</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="verity-card">
                <div class="spec-row"><span class="spec-label">System RAM Target:</span><span class="spec-value">16 GB (Local)</span></div>
                <div class="spec-row"><span class="spec-label">Target GPU:</span><span class="spec-value">NVIDIA RTX 3050 (4 GB VRAM)</span></div>
                <div class="spec-row"><span class="spec-label">Target Precision:</span><span class="spec-value">4-bit Quantization (GGUF)</span></div>
                <div class="spec-row"><span class="spec-label">Telemetry:</span><span class="spec-value">Zero External Requests</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown('<div class="section-header">System Diagnostics</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="verity-card">
                <div class="spec-row"><span class="spec-label">Python Runtime:</span><span class="spec-value">{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}</span></div>
                <div class="spec-row"><span class="spec-label">Data Directory:</span><span class="spec-value">./data</span></div>
                <div class="spec-row"><span class="spec-label">Models Directory:</span><span class="spec-value">./models</span></div>
                <div class="spec-row"><span class="spec-label">Test Suite:</span><span class="spec-value">Configured (pytest)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-header">Test Suite Execution</div>', unsafe_allow_html=True)
        st.info(
            "VerityAI includes an automated test harness verifying configuration logic and "
            "safe environment variable handling. Run tests from terminal: `python -m pytest -q`"
        )


def main() -> None:
    """Main application routine."""
    st.set_page_config(
        page_title="VerityAI — Evidence-Based Research",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_styles()
    config = get_config()

    selected_view = render_sidebar(config)

    if selected_view == "Research Workspace":
        render_research_workspace(config)
    elif selected_view == "Documents":
        render_documents_view(config)
    elif selected_view == "Verification":
        render_verification_view()
    elif selected_view == "Model Evaluation":
        render_model_eval_view()
    elif selected_view == "Settings":
        render_settings_view(config)


if __name__ == "__main__":
    main()

"""
VerityAI - Research Workbench Application
Evidence-based, offline-first research assistant.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure the 'src' directory is on sys.path for direct imports
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from verityai.config import get_config


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

    st.markdown(
        """
        <div class="empty-workspace-box">
            <div class="empty-workspace-title">No Documents Ingested</div>
            <div class="empty-workspace-desc">
                Your research repository is currently empty. Ingest technical papers, reports, or
                documentation to build a private evidence repository for verified, citation-backed answers.
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
    st.markdown('<div class="section-header">Evidence Query</div>', unsafe_allow_html=True)

    st.text_input(
        "Enter a research question across your documents:",
        placeholder="e.g., What are the documented tradeoffs between quantized attention heads?",
        disabled=True,
        help="Query execution requires at least one ingested document in the corpus.",
    )
    st.caption("Query execution is enabled once documents have been added to the local corpus.")

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


def render_documents_view() -> None:
    """Renders the document management view."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #1E293B;">
            <h1 class="verity-brand-title">Documents</h1>
            <p class="verity-tagline">Manage local research papers, technical documentation, and evidence sources.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="empty-workspace-box">
            <div class="empty-workspace-title">Document Repository Empty</div>
            <div class="empty-workspace-desc">
                No research files have been added to the local catalog. Document parsing tools for local
                PDF, DOCX, and text files will be configured here.
            </div>
            <span class="verity-badge">Local Storage: ./data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">Supported Document Formats</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="verity-card">
            <div class="verity-card-body">
                The local ingestion pipeline is designed for <strong>PDF</strong> research papers, <strong>DOCX</strong> reports,
                and <strong>Markdown / Plain Text</strong> files. Extracted text will be indexed into deterministic local chunks
                for granular citation attribution.
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
        render_documents_view()
    elif selected_view == "Verification":
        render_verification_view()
    elif selected_view == "Model Evaluation":
        render_model_eval_view()
    elif selected_view == "Settings":
        render_settings_view(config)


if __name__ == "__main__":
    main()

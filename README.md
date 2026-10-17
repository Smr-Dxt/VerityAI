# VerityAI — Offline-First Agentic Research Assistant

VerityAI is an offline-first research assistant designed to process technical documents, retrieve verified evidence, ground answers in strict citations, detect cross-document contradictions, and evaluate local open-source language models.

The project is built specifically to operate locally and cost-free on consumer hardware (tested for systems with **16 GB RAM** and an **NVIDIA RTX 3050 4GB VRAM GPU**).

---

## 1. Project Status: Phase 3 — Chunking, Embeddings & Hybrid Retrieval

> **Current Phase:** Phase 3 (Source-Aware Chunking, Local Embeddings, and Hybrid Retrieval)  
> **Status:** Operational (101 automated tests passing)  
> **Active Focus:** Source-aware text chunking with word boundary snapping and strict 1-indexed PDF page preservation; dense local embedding generation via Sentence Transformers; FAISS vector indexing; BM25 lexical keyword matching; Reciprocal Rank Fusion (RRF); and interactive Streamlit evidence retrieval.

### What is Available Now (Phase 3)
- **Source-Aware Chunking (`src/verityai/chunking.py`):**
  - Character-based chunking (~1,200 chars / ~200–250 words) with word boundary snapping to prevent sliced tokens.
  - Configurable sliding window overlap (~200 chars / ~35–45 words).
  - Strict page attribution: PDF pages are chunked independently without cross-page mixing; every chunk records its exact 1-indexed `page_number`.
  - Deterministic chunk IDs (e.g. `doc_a1b2_p2_c0`).
- **Local Text Embeddings (`src/verityai/embeddings.py`):**
  - Powered by `sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0 license, ~80 MB file size, 384-dimensional dense vectors).
  - Lazy loading: Postpones model initialization until first retrieval call to keep app launch instant.
  - Hardware adaptive: Auto-detects NVIDIA CUDA GPU acceleration when available, with clean CPU fallback.
  - L2-normalized embeddings for fast cosine similarity / inner product search.
  - Zero telemetry: Documents and queries are processed entirely on-device without cloud transmission.
- **Dense Vector Search (`src/verityai/retrieval_semantic.py`):**
  - In-memory `faiss.IndexFlatIP` (Exact Inner Product) vector index.
  - Deterministic 1-to-1 mapping preserving chunk metadata across searches.
- **Lexical Keyword Search & Reciprocal Rank Fusion (`src/verityai/retrieval_hybrid.py`):**
  - BM25 lexical retriever with Lucene-standard non-negative IDF formulation.
  - Reciprocal Rank Fusion (RRF) combining dense semantic and sparse keyword rankings purely by position:
    $$\text{RRF Score}(d) = \sum_{m} \frac{w_m}{k + \text{rank}_m(d)}$$
  - Deduplication by stable chunk ID with consensus score boosting.
- **Interactive Streamlit Workbench (`app.py`):**
  - Active Evidence Retrieval query interface in Research Workspace.
  - Passage cards displaying rank, filename, PDF page number, RRF score, semantic rank, and keyword rank.
  - Explicit empty-result messaging.
- **Automated Test Harness (`tests/`):**
  - 101 automated pytest tests across 7 test suites covering ingestion, chunking, embeddings, semantic retrieval, hybrid search, evaluation metrics, and headless Streamlit `AppTest` workflows.

### Benchmark Evaluation Quality Results
Evaluated on a standardized 6-document technical corpus with 5 representative research queries:

| Metric | Result | Meaning |
|---|:---:|---|
| **Mean Recall@3** | **80.00%** | Relevant target passage retrieved in the top 3 results 80% of the time. |
| **Mean Recall@1** | **40.00%** | The #1 top result is the exact target answer 40% of the time. |
| **Mean Precision@1** | **40.00%** | Top recommendation is directly relevant to the query. |
| **Mean Reciprocal Rank (MRR)** | **0.6000** | On average, relevant evidence ranks between 1st and 2nd position. |
| **Average Query Latency** | **0.433 ms** | Sub-millisecond search speed (<0.5 ms per query) running locally in RAM. |
| **Provenance Integrity** | **100.00%** | 100% of retrieved results strictly preserved their filename, doc_id, and 1-indexed page. |

### What is Planned for Later Phases
- **Phase 4 (Agentic Verification & Citations):** Multi-hop retrieval, citation grounding, claim verification, and automated contradiction checks.
- **Phase 5 (Local Model Evaluation):** Quantized open-source LLM benchmarking (GGUF / Ollama) on RTX 3050 4GB.
- **Future Document Enhancements:** Local OCR for scanned documents and DOCX parser.

---

## 2. Prerequisites

- **Operating System:** Windows 10/11 (PowerShell)
- **Python Version:** Python 3.10 to 3.13 (64-bit recommended; tested on Python 3.13.2)
- **Hardware Profile:** 8 GB+ RAM (16 GB recommended), optional NVIDIA GPU for CUDA acceleration.
- **Internet Access:** Only required during initial package installation via `pip` and initial model download (~80 MB). Once downloaded, VerityAI operates 100% offline.

---

## 3. Quick Start (Windows PowerShell)

Follow these exact commands in Windows PowerShell to set up and run VerityAI:

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd c:\Desktop\VerityAI
```

### Step 2: Create the Local Virtual Environment (`.venv`)
```powershell
py -m venv .venv
```
*(If `py` is not recognized, use `python -m venv .venv`)*

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

> **PowerShell ExecutionPolicy Note:**  
> If PowerShell blocks script activation with an error like `cannot be loaded because running scripts is disabled on this system`, run the following in your PowerShell session to temporarily allow it:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```
> Alternatively, you can run commands directly using the virtual environment's Python executable without activating:
> ```powershell
> .\.venv\Scripts\python.exe -m pip install -r requirements.txt
> ```

### Step 4: Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run the Automated Test Suite
```powershell
python -m pytest -q
```
*(Runs all 101 offline tests in ~30 seconds).*

### Step 6: Launch the Streamlit Research Workbench
```powershell
python -m streamlit run app.py
```
Streamlit will automatically open your default browser at `http://localhost:8501`.

---

## 4. Repository Structure

```text
VerityAI/
├── app.py                            # Streamlit research workbench UI & Evidence Retrieval workspace
├── requirements.txt                  # Phase 3 dependencies (Streamlit, Pytest, PyMuPDF, Sentence-Transformers, FAISS, BM25)
├── README.md                         # Project documentation and getting started guide
├── .gitignore                        # Ignores .venv, cache, local data, models, and temp files
├── .env.example                      # Environment variable template with safe defaults
├── pytest.ini                        # Pytest configuration mapping src directory
├── src/
│   └── verityai/
│       ├── __init__.py               # Package exports, version (0.3.0), and phase metadata
│       ├── config.py                 # Immutable dataclass configuration module with limits
│       ├── ingestion.py              # In-memory TXT/PDF extraction, PageContent, and DocumentResult
│       ├── chunking.py               # Source-aware chunker preserving word and 1-indexed page boundaries
│       ├── embeddings.py             # Local sentence embeddings (all-MiniLM-L6-v2, MockEmbedder)
│       ├── retrieval_semantic.py     # In-memory FAISS dense vector retriever
│       └── retrieval_hybrid.py       # BM25 lexical search + Reciprocal Rank Fusion (RRF)
└── tests/
    ├── test_config.py                # Automated tests verifying configuration logic
    ├── test_ingestion_txt.py         # Unit tests for UTF-8, BOM, Unicode, and TXT edge cases
    ├── test_ingestion_pdf.py         # Unit tests for synthetic PDFs, pages, and encryption
    ├── test_ingestion_validation.py  # Edge case and format validation tests
    ├── test_chunking.py              # Unit tests for source-aware text chunking and page provenance
    ├── test_embeddings.py            # Unit tests for MockEmbedder and LocalEmbedder
    ├── test_retrieval_semantic.py    # Unit tests for FAISS dense vector search
    ├── test_retrieval_hybrid.py      # Unit tests for BM25 and Reciprocal Rank Fusion
    ├── test_retrieval_evaluation.py  # Benchmark retrieval evaluation (Recall, MRR, latency)
    └── test_ui_integration.py        # Streamlit AppTest headless integration tests
```

---

## 5. Technical Design Details

### Embedding Model: `all-MiniLM-L6-v2`
- **License:** Apache 2.0 (open source, permissive commercial/research license).
- **Download:** Downloaded automatically on first query execution via Hugging Face Hub (~80 MB).
- **Cache Location:** Stored in the local cache (`~/.cache/huggingface/hub` or configured `models_dir`), excluded from Git via `.gitignore`.
- **Offline Operation:** Once downloaded, model execution is 100% offline. Zero network calls are made during search.
- **Hardware Optimization:** Uses ~120 MB RAM during inference. Automatically engages CUDA on your NVIDIA GPU (tested on RTX 3050 4GB) or runs with fast matrix operations on CPU.

### Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
Raw BM25 scores (unbounded positive numbers) and cosine similarity scores ([-1.0, 1.0]) cannot be combined directly without unprincipled scaling. VerityAI merges channels using rank-based Reciprocal Rank Fusion with a smoothing constant of $k=60$. Chunks identified by both dense semantic and sparse lexical channels receive a boosted score, prioritizing high-consensus evidence.

---

## 6. Known Scope Limitations
- **No LLM Answer Generation:** Phase 3 is strictly an evidence retrieval engine. Answer synthesis, claim extraction, and citation verification will be implemented in Phase 4.
- **Character Approximation:** Chunking uses character windows with word-boundary snapping rather than subword tokenizers to remain lightweight during ingestion.
- **Scanned Documents:** Phase 3 processes digital PDFs with extractable text layers. Scanned/image-only PDFs require OCR, which is scheduled for future development.

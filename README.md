# VerityAI — Offline-First Agentic Research Assistant

VerityAI is an offline-first research assistant designed to process technical documents, retrieve verified evidence, ground answers in strict citations, detect cross-document contradictions, and evaluate local open-source language models.

The project is built specifically to operate locally and cost-free on consumer hardware (tested for systems with **16 GB RAM** and an **NVIDIA RTX 3050 4GB VRAM GPU**).

---

## 1. Project Status: Phase 2 — Local Document Ingestion

> **Current Phase:** Phase 2 (Local Document Ingestion)  
> **Status:** Operational  
> **Active Focus:** Offline document extraction for plain-text (.txt) and PDF (.pdf) files, strict 1-indexed page preservation for future citations, format and file-size guardrails, and interactive Streamlit workbench ingestion.

### What is Available Now (Phase 2)
- **Modular Project Structure:** Standard `src/` layout separating configuration, ingestion pipelines, test suites, and application UI.
- **Safe Configuration System:** Dataclass-based immutable settings (`src/verityai/config.py`) with environment variable overrides and configurable ingestion ceilings (`max_file_size_bytes`, `max_pdf_pages`).
- **Validated Plain-Text (.txt) Extraction:** UTF-8 decoding, UTF-8 BOM auto-stripping, empty/whitespace rejection, line-break normalization, and untrusted filename sanitization.
- **Page-Aware PDF (.pdf) Extraction:** In-memory byte streaming via PyMuPDF (no disk writes), 1-indexed page boundaries (`PageContent(page_number=...)`) for granular citation attribution, encrypted/password detection, and scanned/image-only PDF detection with clear OCR notices.
- **Unified Ingestion API:** `ingest_document()` dispatcher and `safe_ingest_document()` boundary returning typed `DocumentResult` objects without uncaught crashes.
- **Streamlit Ingestion Workspace:** Interactive file upload, document metadata summary card, page-by-page PDF text inspector, text preview containers, actionable error alerts, and live research corpus status.
- **Automated Test Harness:** 56 passing pytest tests covering unit extraction, limits, encoding, corrupted files, and headless Streamlit `AppTest` execution.

### What is Planned for Later Phases
- **Phase 3 (Offline Embeddings & Indexing):** Local vector indexing and hybrid search (BM25 + FAISS) sized for 16GB RAM.
- **Phase 4 (Agentic Verification & Citations):** Multi-hop retrieval, citation grounding, and automated contradiction checks.
- **Phase 5 (Local Model Evaluation):** Quantized open-source LLM benchmarking (GGUF / Ollama) on RTX 3050 4GB.
- **Future Document Enhancements:** Local OCR for scanned documents and DOCX parser.

---

## 2. Prerequisites

- **Operating System:** Windows 10/11 (PowerShell)
- **Python Version:** Python 3.10 to 3.13 (64-bit recommended; tested on Python 3.13)
- **Hardware Profile:** 8 GB+ RAM (16 GB recommended), optional NVIDIA GPU for later phases.
- **Internet Access:** Only required during initial package installation via `pip`. Once packages are installed, Phase 1 runs completely offline.

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

### Step 6: Launch the Streamlit Research Workbench
```powershell
python -m streamlit run app.py
```
Streamlit will automatically open your default browser at `http://localhost:8501`.

---

## 4. Repository Structure

```text
VerityAI/
├── app.py                            # Streamlit research workbench UI & document workspace
├── requirements.txt                  # Phase 2 dependencies (Streamlit, Pytest, PyMuPDF)
├── README.md                         # Project documentation and getting started guide
├── .gitignore                        # Ignores .venv, cache, local data, models, and temp files
├── .env.example                      # Environment variable template with safe defaults
├── pytest.ini                        # Pytest configuration mapping src directory
├── src/
│   └── verityai/
│       ├── __init__.py               # Package exports, version (0.2.0), and phase metadata
│       ├── config.py                 # Immutable dataclass configuration module with limits
│       └── ingestion.py              # In-memory TXT/PDF extraction, PageContent, and DocumentResult
└── tests/
    ├── test_config.py                # Automated tests verifying configuration logic
    ├── test_ingestion_txt.py         # Unit tests for UTF-8, BOM, Unicode, and TXT edge cases
    ├── test_ingestion_pdf.py         # Unit tests for synthetic PDFs, pages, and encryption
    ├── test_ingestion_validation.py  # Edge case and format validation tests
    └── test_ui_integration.py        # Streamlit AppTest headless integration tests
```

### File Responsibilities
- `app.py`: Streamlit entrypoint; renders research workspace, interactive document uploader, metadata summary cards, and page-by-page text inspector.
- `src/verityai/config.py`: Encapsulates application settings and configurable limits (`max_file_size_bytes=10MB`, `max_pdf_pages=500`).
- `src/verityai/ingestion.py`: Local document ingestion engine with `PageContent`, `DocumentResult`, `extract_txt`, `extract_pdf`, and `safe_ingest_document`.
- `tests/`: 56 automated pytest tests verifying configuration, extraction, limits, and headless UI workflows.
- `.gitignore`: Prevents heavy AI model checkpoints, private research documents, and virtual environments from accidentally being committed.

---

## 5. Troubleshooting

### Issue 1: Script Execution Disabled in PowerShell
- **Symptom:** `.\.venv\Scripts\Activate.ps1 : File ... cannot be loaded because running scripts is disabled on this system.`
- **Fix:** Enable script execution for your current PowerShell session only:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```
  Or directly invoke the virtual environment's Python binary:
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run app.py
  ```

### Issue 2: ModuleNotFoundError: No module named 'verityai'
- **Symptom:** When running tests or the app, Python cannot locate the `verityai` package.
- **Fix:** Ensure you run commands from the project root (`c:\Desktop\VerityAI`). The test suite and `app.py` resolve the root directory automatically. You can also install the package in editable mode if desired:
  ```powershell
  pip install -e .
  ```
  Or set `PYTHONPATH` in PowerShell:
  ```powershell
  $env:PYTHONPATH = "src"
  ```

### Issue 3: Streamlit Port Already in Use (Port 8501)
- **Symptom:** `Port 8501 is already in use.`
- **Fix:** Specify a different port using the `--server.port` flag:
  ```powershell
  python -m streamlit run app.py --server.port 8502
  ```

---

## 6. Development Principles
1. **Offline-First:** All processing must function without mandatory cloud API connections or telemetry.
2. **Deterministic Citations:** Answers must reference exact document excerpts, never hallucinated claims.
3. **Hardware Realism:** Architected to run comfortably within 16 GB RAM and 4 GB VRAM budgets.
4. **Transparent Evolution:** Every phase is fully tested and verified before moving forward.

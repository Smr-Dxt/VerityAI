# VerityAI — Offline-First Agentic Research Assistant

VerityAI is an offline-first research assistant designed to process technical documents, retrieve verified evidence, ground answers in strict citations, detect cross-document contradictions, and evaluate local open-source language models.

The project is built specifically to operate locally and cost-free on consumer hardware (tested for systems with **16 GB RAM** and an **NVIDIA RTX 3050 4GB VRAM GPU**).

---

## 1. Project Status: Phase 1 — Project Foundation

> **Current Phase:** Phase 1 (Project Foundation)  
> **Status:** Operational  
> **Active Focus:** Clean repository structure, immutable configuration management, automated test harness, and Streamlit research workbench UI.

### What is Available Now (Phase 1)
- **Modular Project Structure:** Standard `src/` layout separating configuration, tests, and application logic.
- **Safe Configuration System:** Dataclass-based immutable settings (`src/verityai/config.py`) with environment variable fallback.
- **Automated Test Harness:** Fast, zero-dependency pytest test suite verifying configuration logic.
- **Streamlit Research Workbench:** Polished, high-contrast user interface displaying system diagnostics and an honest placeholder research workspace.
- **Windows-Optimized Environment:** Git and `.gitignore` setup specifically tuned for Windows Python development.

### What is Planned for Later Phases
- **Phase 2 (Document Ingestion):** Local parsing of PDF, DOCX, and text documents with metadata extraction.
- **Phase 3 (Offline Embeddings & Indexing):** Local vector indexing and hybrid search (BM25 + FAISS) sized for 16GB RAM.
- **Phase 4 (Agentic Verification & Citations):** Multi-hop retrieval, citation grounding, and automated contradiction checks.
- **Phase 5 (Local Model Evaluation):** Quantized open-source LLM benchmarking (GGUF / Ollama) on RTX 3050 4GB.

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
├── app.py                  # Streamlit application entrypoint & workbench UI
├── requirements.txt        # Minimal Phase 1 dependencies (Streamlit, Pytest)
├── README.md               # Project documentation and getting started guide
├── .gitignore              # Ignores .venv, cache, local data, models, and temp files
├── .env.example            # Environment variable template with safe defaults
├── pytest.ini              # Pytest configuration mapping src directory
├── src/
│   └── verityai/
│       ├── __init__.py     # Package exports and version metadata
│       └── config.py       # Immutable dataclass configuration module
└── tests/
    └── test_config.py      # Automated pytest tests verifying configuration logic
```

### File Responsibilities
- `app.py`: Launches the interactive user interface, injects high-contrast research styling, and renders system diagnostics and placeholder empty states.
- `src/verityai/config.py`: Encapsulates application settings in an immutable `AppConfig` dataclass and parses environment overrides safely.
- `tests/test_config.py`: Verifies configuration defaults, environment variable overrides, immutability, and boolean helpers.
- `.gitignore`: Prevents heavy AI model checkpoints (`.gguf`, `.safetensors`), private research documents (`.pdf`, `.docx`), and virtual environments from accidentally being committed.

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

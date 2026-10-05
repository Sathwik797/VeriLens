# 🔍 VeriLens AI: Self-Verifying LLM Data Analyst

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-91%2F91%20passing-brightgreen.svg)]()
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-orange.svg)](https://deepmind.google/technologies/gemini/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206-orange.svg)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Autonomous exploratory data analysis with deterministic ground-truth self-verification.**

Traditional LLM data assistants suffer from a fundamental flaw: **hallucination**. They frequently invent metrics, aggregate data incorrectly, or sound convincingly confident while misquoting numerical results. 

**VeriLens AI** solves this with an architectural separation of concerns:
- **LLMs are used strictly for reasoning and natural-language narrative.**
- **Deterministic code engines (Pandas SafeExecutor) perform all mathematical computation.**
- **Deterministic verification algorithms verify every claim against the ground truth before presenting it to the user.**

Every answer is paired with an explainable **Trust Score (0–100)**, claim-by-claim verification breakdown badges (✅ *Verified*, ❌ *Mismatch*, ⚠️ *Inconclusive*), and the raw ground-truth evidence table.

---

## 🏛️ System Architecture

```text
                                USER QUESTION + CSV DATASET
                                             │
                                             ▼
                                     ┌───────────────┐
                                     │PlannerService │  (Gemini LLM)
                                     └───────┬───────┘
                                             │ Structured AnalysisPlan
                                             ▼
                                     ┌───────────────┐
                                     │ SafeExecutor  │  (Sandboxed Pandas Engine)
                                     └───────┬───────┘
                                             │ Ground-Truth Evidence
                                             ▼
                                     ┌───────────────┐
                                     │NarratorService│  (Gemini LLM)
                                     └───────┬───────┘
                                             │ Natural-Language Explanation
                                             ▼
                                  ┌────────────────────┐
                                  │ClaimExtractorService│ (Gemini Structured Extraction)
                                  └──────────┬─────────┘
                                             │ Structured Analytical Claims
                                             ▼
                                  ┌────────────────────┐
                                  │VerificationService │ (Pure Deterministic Verification)
                                  └──────────┬─────────┘
                                             │ VerificationResults
                                             ▼
                                  ┌────────────────────┐
                                  │ TrustScoreService  │ (Deterministic 0-100 Trust Score)
                                  └──────────┬─────────┘
                                             │
                                             ▼
                        GRADIO UI: Answer + Trust Score + Badges + Evidence
```

---

## ✨ Key Features

### 1. Deterministic Self-Verification Loop
- **Zero Hallucination Tolerance:** Every natural language sentence is broken down into structured `AnalyticalClaim` objects (`subject`, `metric`, `value`, `comparison`, `group_by`).
- **Mathematical Ground-Truth Check:** Claims are tested with strict tolerance matching, ranking checks (`highest`, `lowest`, `greater_than`, `less_than`, `equal`), and schema normalization.
- **Explainable Trust Score:** A deterministic scoring algorithm (100 for verified, 50 for inconclusive, 0 for mismatch) provides an unambiguous confidence metric with actionable rationales.

### 2. Dataset-Agnostic Automated EDA
- **Dynamic Heuristic Profiling:** Works out of the box with unseen CSV datasets (tested with Superstore, Walmart Sales, and custom schemas).
- **Intelligent Measure Detection:** Discounts ID, code, phone, and sequential integer columns; boosts continuous floating-point business metrics (`Sales`, `Profit`, `Weekly_Sales`, `Temperature`, `CPI`, etc.).
- **Smart Grouping & Temporal Analysis:** Automatically selects high-quality categorical groupings and detects date/time dimensions for trend series.
- **4 Interactive Visualizations:** Renders distribution histograms, categorical aggregations, and chronological trends using Plotly.

### 3. Sandboxed SafeExecutor
- Executes structured `AnalysisPlan` specifications (`groupby`, `filter`, `aggregate`, `sort`, `top_k`) directly with Pandas.
- Prevents arbitrary code injection and syntax vulnerabilities.

### 4. Interactive Web Interface
- Clean, responsive Gradio dashboard.
- Live dataset profiling cards (Rows, Columns, Duplicates, Memory Usage).
- Side-by-side EDA charts and interactive verification inspection table.

---

## 📁 Repository Structure

```text
VeriLens-AI/
├── app/
│   ├── eda/
│   │   ├── loader.py                 # Robust CSV loading and encoding detection
│   │   ├── profiler.py               # Memory, schema, and statistical profiling
│   │   └── visualizer.py             # Dataset-agnostic Plotly visualization engine
│   ├── executor/
│   │   └── safe_executor.py          # Sandboxed Pandas analytical execution engine
│   ├── formatters/
│   │   ├── dashboard_formatter.py    # HTML dataset profile cards
│   │   └── verification_formatter.py # Trust score cards and verification badge formatters
│   ├── models/
│   │   ├── analysis_plan.py          # Structured Pydantic execution plan schemas
│   │   ├── dataset_profile.py        # Dataset summary profile schema
│   │   ├── trust_score.py            # Trust score result and status models
│   │   └── verification.py           # Claim and VerificationResult models
│   └── services/
│       ├── analysis_service.py       # Complete orchestrator connecting all 7 pipeline stages
│       ├── claim_extractor_service.py# LLM-powered structured claim extraction
│       ├── eda_service.py            # High-level EDA analysis pipeline
│       ├── narrator_service.py       # Grounded LLM narrative generator
│       ├── planner_service.py        # Analytical reasoning and plan generation
│       ├── trust_score_service.py    # Deterministic trust score calculation
│       └── verification_service.py   # Pure deterministic claim verification algorithms
├── datasets/
│   ├── superstore/                   # Sample Superstore dataset
│   └── walmart/                      # Walmart weekly sales dataset
├── docs/                             # Architecture diagrams and technical specs
├── tests/                            # Comprehensive unittest suite (91/91 passing)
│   ├── test_analysis_service.py
│   ├── test_claim_extractor_service.py
│   ├── test_eda_visualizer.py
│   ├── test_narrator_service.py
│   ├── test_planner_service.py
│   ├── test_safe_executor.py
│   ├── test_trust_score_service.py
│   ├── test_ui.py
│   ├── test_verification_models.py
│   └── test_verification_service.py
├── ui/
│   └── app.py                        # Gradio 6 dashboard and event wiring
├── main.py                           # Application entrypoint
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template for environment configuration
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Sathwik797/VeriLens.git
cd VeriLens
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Gemini API Key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 5. Launch the Application
```bash
python main.py
```
Open your browser at `http://127.0.0.1:7860/`.

---

## 🧪 Running Tests

The test suite contains 91 unit and integration tests covering the complete pipeline:

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test suites
python -m unittest tests/test_eda_visualizer.py -v
python -m unittest tests/test_verification_service.py -v
python -m unittest tests/test_analysis_service.py -v
python -m unittest tests/test_trust_score_service.py -v
```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).

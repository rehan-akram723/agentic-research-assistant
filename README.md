# 🔬 Multi-Agent Research Assistant

> An autonomous multi-agent AI system that accepts a natural language research question, searches academic sources (arXiv, Semantic Scholar), summarises relevant papers, verifies citations, and generates a structured Markdown research report — end-to-end, automatically.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-blueviolet.svg)](https://langchain-ai.github.io/langgraph/)
[![React](https://img.shields.io/badge/React-18%2B-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ What It Does

1. **Plans** — An LLM decomposes your research question into 3–5 targeted academic search queries.
2. **Retrieves** — Papers are fetched from arXiv and Semantic Scholar; results are deduplicated and stored in a FAISS vector index.
3. **Summarises** — Each paper is independently summarised by the LLM (problem, method, results, limitations).
4. **Verifies** — Factual claims are cross-checked against the source text using FAISS similarity search and LLM judgement.
5. **Reports** — A structured Markdown report is synthesised from all summaries and verified citations.

---

## 🏗️ Architecture

The system follows **Clean Architecture** with four concentric layers. Each layer depends only on layers below it — no agent imports from the API layer and no route handler imports from the agent layer.

```
┌──────────────────────────────────────┐
│  Interface Layer                     │
│  React UI  ·  FastAPI REST API       │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│  Application Layer                   │
│  ResearchService  ·  Pydantic Schemas│
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│  Orchestration Layer                 │
│  LangGraph CompiledGraph             │
└──┬──────┬───────┬──────────┬─────────┘
   │      │       │          │
   ▼      ▼       ▼          ▼
Planner Retrieval Summarizer Citation  Report
 Agent   Agent    Agent     Verifier  Generator
                             Agent     Agent
   │      │
   ▼      ▼
  LLM   arXiv + Semantic Scholar + FAISS
```

See [`docs/architecture.md`](docs/architecture.md) for a full layer-by-layer breakdown including the agent pipeline flow, error taxonomy, and key design decisions.

### Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 0.2+ |
| LLM | OpenRouter (OpenAI-compatible API) |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + Framer Motion |
| Academic Retrieval | arXiv API + Semantic Scholar API (Concurrent) |
| Vector Memory | FAISS + sentence-transformers |
| Config & Validation | Pydantic V2 + pydantic-settings |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai/) API key (free tier available)

### 1. Clone and install

```bash
git clone https://github.com/rehan-akram723/agentic-research-assistant.git
cd agentic-research-assistant
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4.1-nano   # or any OpenRouter model
LOG_LEVEL=INFO
```

### 3. Start the API server

```bash
uvicorn src.api.main:app --reload
```

The Swagger UI is available at **http://localhost:8000/docs**.

### 4. Start the React UI (separate terminal)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 📡 API Reference

### `POST /research/`

Run the full multi-agent research pipeline.

**Request body:**
```json
{
  "question": "What are recent advances in retrieval-augmented generation?",
  "max_results": 8,
  "include_citations": true
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "question": "...",
  "report": "# Research Synthesis\n\n## Overview\n...",
  "citations": [
    {
      "claim": "RAG improves factual accuracy",
      "source_doc_id": "arxiv:2005.11401",
      "source_title": "Retrieval-Augmented Generation for ...",
      "is_verified": true,
      "quote": "..."
    }
  ],
  "queries_used": ["retrieval augmented generation survey", "..."],
  "docs_retrieved": 8,
  "processing_time_seconds": 42.3,
  "error_message": null
}
```

**Error responses:**

| Status | Meaning |
|---|---|
| 422 | Invalid request (question too short, max_results out of range) |
| 503 | No papers retrieved from arXiv / Semantic Scholar |
| 502 | LLM agent pipeline failure |
| 500 | Unexpected server error |

### `GET /health`

Docker liveness probe. Always returns `{"status": "ok", "version": "1.0.0"}`.

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Integration tests only (API layer)
pytest tests/integration/test_api_routes.py -v

# Unit tests only
pytest tests/unit/ -v
```

The integration tests mock `ResearchService` and run entirely without a network connection or API key.

---

## 📊 Evaluation

Run the benchmark evaluation against 10 curated research questions:

```bash
# Start the API first, then:
python scripts/evaluate.py

# Customise
python scripts/evaluate.py --max-results 5 --questions 5 --timeout 180
```

Results are saved to:
- `docs/evaluation.csv` — raw metrics (latency, docs, word count, citation rate)
- `docs/evaluation.md` — formatted Markdown report with summary statistics

---

## 📁 Project Structure

```
agentic-research-assistant/
├── frontend/            # React + Vite application
├── src/
│   ├── agents/          # Planner, Retrieval, Summarizer, CitationVerifier, ReportGenerator
│   ├── api/             # FastAPI routes, middleware, main app factory
│   ├── core/            # Config (Pydantic Settings), logger, exceptions, LLM helper
│   ├── graph/           # LangGraph state, graph builder, conditional edges
│   ├── schemas/         # Pydantic V2 data contracts
│   ├── services/        # ResearchService, EmbeddingService
│   ├── tools/           # arXiv, Semantic Scholar, FAISS tools
│   └── ui/              # Legacy Streamlit app and components
├── tests/
│   ├── integration/     # FastAPI TestClient tests
│   └── unit/            # Agent and tool unit tests
├── docs/
│   ├── architecture.md  # Detailed system design
│   └── evaluation.md    # Benchmark results (generated)
├── scripts/
│   └── evaluate.py      # End-to-end evaluation benchmark
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── PROJECT_MANIFEST.md
```

---

## 🐳 Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# API will be available at http://localhost:8000
```

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | — | OpenRouter API key |
| `OPENROUTER_MODEL` | ❌ | `openai/gpt-4.1-nano` | LLM model identifier |
| `OPENROUTER_BASE_URL` | ❌ | `https://openrouter.ai/api/v1` | API base URL |
| `LOG_LEVEL` | ❌ | `INFO` | Python logging level |
| `FAISS_INDEX_PATH` | ❌ | `./faiss_index` | FAISS index storage path |
| `MAX_RESULTS_CAP` | ❌ | `30` | Hard cap for max_results |

---

## 📝 Example Queries

```
What are recent methods for low-resource NLP fine-tuning?
Summarize advances in small language model compression and quantization.
How does retrieval-augmented generation improve factual accuracy in LLMs?
What are state-of-the-art techniques for graph neural networks on heterogeneous data?
Summarize research on protein structure prediction with machine learning.
```

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.

---

*Built with LangGraph · FastAPI · React · FAISS · OpenRouter*

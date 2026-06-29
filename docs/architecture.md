# Architecture — Multi-Agent Research Assistant

## Overview

The system is a **multi-agent AI pipeline** that accepts a natural language research question, autonomously retrieves and analyses academic papers from public databases, and generates a structured, citation-backed Markdown report. It is built on the principle of **Clean Architecture**: every layer has a single responsibility and depends only on layers below it.

```
┌─────────────────────────────────────────────────┐
│                 Interface Layer                  │
│        Streamlit UI  ·  FastAPI REST API         │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│               Application Layer                  │
│          ResearchService  ·  Schemas             │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│             Orchestration Layer                  │
│       LangGraph CompiledGraph  ·  Edges          │
└───┬──────────┬───────────┬────────────┬─────────┘
    │          │           │            │
    ▼          ▼           ▼            ▼
Planner   Retrieval  Summarizer  Citation   Report
 Agent     Agent      Agent      Verifier   Generator
                                  Agent      Agent
    │          │
    ▼          ▼
 LLM API    arXiv + Semantic Scholar + FAISS
```

---

## Layers in Detail

### 1. Interface Layer

#### FastAPI (`src/api/`)

The REST API is the primary interface between external clients and the research pipeline.

| File | Responsibility |
|---|---|
| `src/api/main.py` | App factory (`create_app`), CORS middleware, lifespan hook |
| `src/api/routes/research.py` | `POST /research/` endpoint; maps domain errors to HTTP status codes |
| `src/api/routes/health.py` | `GET /health` liveness probe for Docker |
| `src/api/middleware/cors.py` | Explicit origin allowlist targeting Streamlit ports |

Error taxonomy enforced at the route layer:

| Exception | HTTP Status | Meaning |
|---|---|---|
| `RetrievalError` | 503 | No papers found from any source |
| `AgentError` | 502 | LLM or agent pipeline failure |
| `ValueError` | 422 | Invalid parameter values |
| Everything else | 500 | Unexpected infrastructure error |

#### Streamlit UI (`src/ui/`)

A multi-page Streamlit application with four modules:

- **`app.py`** — entry point, global CSS, session state management, API calls
- **`components/sidebar.py`** — settings, API status badge, example queries
- **`components/query_input.py`** — validated textarea with character counter
- **`components/report_viewer.py`** — metadata ribbon, Markdown report, citation expander, download button

---

### 2. Application Layer

#### ResearchService (`src/services/research_service.py`)

The sole boundary between the API layer and the graph internals. No route handler imports from `src.agents` or `src.graph` directly.

Responsibilities:
- Reset the FAISS memory singleton before each request (isolation)
- Construct a fully-typed initial `ResearchState`
- Offload the synchronous `CompiledGraph.invoke` call to `asyncio.to_thread` (non-blocking)
- Classify the final state into `"success" | "partial" | "error"`
- Coerce raw citation dicts into validated `Citation` Pydantic models

#### Schemas (`src/schemas/`)

All data contracts are Pydantic V2 models:

| Model | File | Purpose |
|---|---|---|
| `ResearchRequest` | `request.py` | Validated API input (question, max_results, include_citations) |
| `ResearchResponse` | `response.py` | Validated API output (report, citations, metadata) |
| `HealthResponse` | `response.py` | Health check response |
| `Document` | `document.py` | Retrieved academic paper |
| `Citation` | `document.py` | Verified factual claim linked to a source |
| `ReportPayload` | `report.py` | LLM output contract for the report generator |

---

### 3. Orchestration Layer

#### LangGraph Pipeline (`src/graph/`)

The pipeline is implemented as a `CompiledGraph` that executes five agents sequentially, with two conditional routing decisions:

```
START
  │
  ▼
[Planner Agent]
  │  search_queries: List[str]
  ▼
[Retrieval Agent]
  │  retrieved_docs: List[Document]
  │
  ├─── handle_retrieval_failure() ──► END (if no docs)
  │
  ▼
[Summarizer Agent]
  │  summaries: List[str]
  │
  ├─── should_verify_citations()
  │         │
  │    True ▼               False ──────────────┐
  │  [Citation Verifier Agent]                  │
  │    verified_citations: List[Citation]        │
  │         │                                   │
  └─────────▼───────────────────────────────────┘
            │
            ▼
    [Report Generator Agent]
            │  final_report: str
            ▼
           END
```

**State object** (`src/graph/state.py`): A single `ResearchState` TypedDict is the only mutable object. Each agent reads from it and returns a partial dict; the graph merges the partial dicts into the state.

---

### 4. Agent Layer (`src/agents/`)

Each agent is a **pure function** `(ResearchState) -> dict`. No agent stores mutable state between calls.

#### Planner Agent
- Calls the LLM with the research question
- Outputs 3–5 targeted academic search query strings

#### Retrieval Agent
- Calls `search_arxiv` and `search_semantic_scholar` for each query
- Deduplicates results by document ID
- Embeds all documents into the FAISS index via `EmbeddingService`

#### Summarizer Agent
- Calls the LLM once per retrieved document
- Extracts problem, method, results, and limitations

#### Citation Verifier Agent
- Extracts factual claims from each summary via the LLM
- Uses FAISS similarity search to find the most relevant source passage per claim
- LLM judges whether the source supports the claim

#### Report Generator Agent
- Sends a **JSON-Schema-Aware** prompt to the LLM specifying exactly the three required array fields
- Implements a **self-healing repair loop**: if the first response fails Pydantic validation, a repair prompt embedding the error details is sent at `temperature=0.0`
- Falls back to a deterministic `_fallback_payload` if both LLM attempts fail

---

### 5. Tools & Infrastructure (`src/tools/`, `src/services/`)

#### arXiv Tool (`src/tools/arxiv_tool.py`)
- Wraps the `arxiv` Python client
- Maps raw API results to `Document` schema

#### Semantic Scholar Tool (`src/tools/semantic_scholar_tool.py`)
- Proactive `time.sleep(1.1)` before every HTTP request (≤1 RPS)
- Reactive exponential back-off (5 s → 10 s → 15 s) on 429 responses
- Graceful degradation: after 3 exhausted retries, returns `[]` without raising

#### FAISS Memory Tool (`src/tools/faiss_memory_tool.py`)
- Session-scoped singleton reset before each request
- Embeds document abstracts into a flat L2 index
- Supports similarity search for citation verification

#### Embedding Service (`src/services/embedding_service.py`)
- Thread-safe singleton wrapping `sentence-transformers/all-MiniLM-L6-v2`
- Deterministic fallback embedder for CI environments without the model downloaded

---

## Configuration

All settings are loaded via Pydantic Settings from the `.env` file:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | ✅ | — | OpenRouter API key (OpenAI-compatible) |
| `OPENROUTER_MODEL` | ❌ | `openai/gpt-4.1-nano` | Model identifier |
| `OPENROUTER_BASE_URL` | ❌ | `https://openrouter.ai/api/v1` | API base URL |
| `LOG_LEVEL` | ❌ | `INFO` | Logging verbosity |
| `FAISS_INDEX_PATH` | ❌ | `./faiss_index` | FAISS index persistence path |
| `MAX_RESULTS_CAP` | ❌ | `30` | Hard cap for `max_results` parameter |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Plain `CompiledGraph` (no LangGraph StateGraph) | Avoids LangGraph's async event loop requirements; the graph runs synchronously in `asyncio.to_thread` |
| `ReportPayload` in `src/schemas/` | Keeps data contracts in one layer; the agent never defines its own schema |
| Minimal LLM contract (3 keys only) | Smaller surface area = fewer ValidationErrors; display fields are filled deterministically |
| Module-level service singleton | `build_graph()` is called once at startup; every request reuses the compiled pipeline |
| Explicit CORS origin allowlist | Safer than wildcard; targets Streamlit's known dev ports |

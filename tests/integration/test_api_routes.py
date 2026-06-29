"""Integration tests for the FastAPI API routes.

Strategy
--------
We test the full HTTP layer (routing, serialisation, error mapping) while
**mocking the ResearchService** so the tests run in milliseconds without
requiring a live Gemini API key, a populated FAISS index, or an internet
connection to arXiv / Semantic Scholar.

Two complementary mocking approaches are used:

1. ``app.dependency_overrides`` — preferred for FastAPI dependency-injected
   services.  Not applicable here because the service is a module-level
   singleton, not a ``Depends``-injected dependency.

2. ``unittest.mock.patch`` — patches ``_service`` (the singleton) inside the
   ``research`` router module **before** the TestClient sends requests.
   This is safe because ``TestClient`` is synchronous and each test patches
   for its own duration.

Test groups
-----------
* ``TestHealthEndpoint``       — GET /health correctness & schema.
* ``TestResearchStatusEndpoint`` — GET /research/status correctness.
* ``TestRunResearchEndpoint``  — POST /research/ happy path & error cases.
* ``TestInputValidation``      — Pydantic 422 responses for bad inputs.
* ``TestCORSHeaders``          — CORS pre-flight and simple-request headers.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.core.exceptions import AgentError, RetrievalError
from src.schemas.document import Citation
from src.schemas.response import ResearchResponse

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    """Return a fresh FastAPI application instance for the test module.

    Using ``scope="module"`` means the app is assembled once and shared across
    all tests in this file, which mirrors the real application lifecycle.
    """
    return create_app()


@pytest.fixture(scope="module")
def client(app):
    """Return a TestClient bound to the test app.

    ``TestClient`` runs the ASGI app synchronously (via ``requests`` under the
    hood), so there is no need to spin up a real server process.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def valid_request_payload() -> dict[str, Any]:
    """Return a minimal valid POST /research/ payload."""
    return {
        "question": "What are recent advances in transformer-based language models?",
        "max_results": 5,
        "include_citations": True,
    }


@pytest.fixture
def sample_response() -> ResearchResponse:
    """Return a canned ``ResearchResponse`` for mocking ``ResearchService.run``."""
    return ResearchResponse(
        status="success",
        question="What are recent advances in transformer-based language models?",
        report=(
            "## Overview\nTransformer-based models have achieved state-of-the-art "
            "results across NLP tasks.\n\n"
            "## Key Methods & Approaches\nSelf-attention, pre-training, fine-tuning.\n\n"
            "## Findings & Results\nGPT-4 and related models show remarkable capability.\n\n"
            "## Limitations & Gaps\nComputational cost remains high.\n\n"
            "## References\n[1] Attention is All You Need"
        ),
        citations=[
            Citation(
                claim="Transformers achieve state-of-the-art results in NLP.",
                source_doc_id="arxiv:1706.03762",
                source_title="Attention Is All You Need",
                is_verified=True,
                quote="The Transformer follows an encoder-decoder structure.",
            )
        ],
        queries_used=[
            "transformer language model advances",
            "large language model survey 2024",
            "BERT GPT pre-training techniques",
        ],
        docs_retrieved=5,
        processing_time_seconds=12.34,
        error_message=None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_service(mock_run_coro):
    """Return a context manager that replaces ``_service.run`` in the router.

    Args:
        mock_run_coro: An ``AsyncMock`` (or coroutine factory) to use as the
            replacement for ``ResearchService.run``.

    Returns:
        A ``unittest.mock.patch`` context manager targeting the router module's
        internal ``_service`` attribute.
    """
    return patch(
        "src.api.routes.research._service.run",
        new=mock_run_coro,
    )


# ===========================================================================
# Test group 1 — GET /health
# ===========================================================================


class TestHealthEndpoint:
    """Tests for the GET /health liveness probe."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint must always return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

    def test_health_content_type_is_json(self, client: TestClient) -> None:
        """Response must be JSON."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_health_status_field_is_ok(self, client: TestClient) -> None:
        """The ``status`` field must be the string ``'ok'``."""
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_version_field_present(self, client: TestClient) -> None:
        """The ``version`` field must be present and non-empty."""
        body = client.get("/health").json()
        assert "version" in body
        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0

    def test_health_schema_matches_health_response(self, client: TestClient) -> None:
        """Response body must be parseable as ``HealthResponse``."""
        from src.schemas.response import HealthResponse

        body = client.get("/health").json()
        # Will raise ValidationError if schema is violated
        parsed = HealthResponse.model_validate(body)
        assert parsed.status == "ok"

    def test_health_method_not_allowed_post(self, client: TestClient) -> None:
        """POST /health must return 405 Method Not Allowed."""
        response = client.post("/health", json={})
        assert response.status_code == 405


# ===========================================================================
# Test group 2 — GET /research/status
# ===========================================================================


class TestResearchStatusEndpoint:
    """Tests for the GET /research/status lightweight status check."""

    def test_status_returns_200(self, client: TestClient) -> None:
        """Status endpoint must return HTTP 200."""
        response = client.get("/research/status")
        assert response.status_code == 200

    def test_status_body_has_status_key(self, client: TestClient) -> None:
        """Response body must contain a ``status`` field."""
        body = client.get("/research/status").json()
        assert "status" in body
        assert body["status"] == "running"

    def test_status_body_has_service_key(self, client: TestClient) -> None:
        """Response body must confirm the service is initialised."""
        body = client.get("/research/status").json()
        assert "service" in body
        assert body["service"] == "initialised"


# ===========================================================================
# Test group 3 — POST /research/ happy path
# ===========================================================================


class TestRunResearchEndpoint:
    """Tests for POST /research/ covering the happy path and error cases."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_valid_request_returns_200(
        self,
        client: TestClient,
        valid_request_payload: dict,
        sample_response: ResearchResponse,
    ) -> None:
        """A well-formed request must return HTTP 200."""
        mock_run = AsyncMock(return_value=sample_response)
        with _patch_service(mock_run):
            response = client.post("/research/", json=valid_request_payload)
        assert response.status_code == 200

    def test_valid_request_returns_research_response_schema(
        self,
        client: TestClient,
        valid_request_payload: dict,
        sample_response: ResearchResponse,
    ) -> None:
        """Response body must be parseable as ``ResearchResponse``."""
        mock_run = AsyncMock(return_value=sample_response)
        with _patch_service(mock_run):
            body = client.post("/research/", json=valid_request_payload).json()
        # Will raise if schema is violated
        parsed = ResearchResponse.model_validate(body)
        assert parsed.status == "success"

    def test_valid_request_response_contains_report(
        self,
        client: TestClient,
        valid_request_payload: dict,
        sample_response: ResearchResponse,
    ) -> None:
        """The ``report`` field in the response must be non-empty."""
        mock_run = AsyncMock(return_value=sample_response)
        with _patch_service(mock_run):
            body = client.post("/research/", json=valid_request_payload).json()
        assert "report" in body
        assert len(body["report"]) > 0

    def test_valid_request_response_contains_citations(
        self,
        client: TestClient,
        valid_request_payload: dict,
        sample_response: ResearchResponse,
    ) -> None:
        """The ``citations`` list must be present and correctly structured."""
        mock_run = AsyncMock(return_value=sample_response)
        with _patch_service(mock_run):
            body = client.post("/research/", json=valid_request_payload).json()
        assert "citations" in body
        assert isinstance(body["citations"], list)
        if body["citations"]:
            citation = body["citations"][0]
            assert "claim" in citation
            assert "source_doc_id" in citation
            assert "is_verified" in citation

    def test_valid_request_response_contains_metadata(
        self,
        client: TestClient,
        valid_request_payload: dict,
        sample_response: ResearchResponse,
    ) -> None:
        """Metadata fields (docs_retrieved, processing_time_seconds, etc.) must be present."""
        mock_run = AsyncMock(return_value=sample_response)
        with _patch_service(mock_run):
            body = client.post("/research/", json=valid_request_payload).json()

        assert "docs_retrieved" in body
        assert isinstance(body["docs_retrieved"], int)
        assert body["docs_retrieved"] >= 0

        assert "processing_time_seconds" in body
        assert isinstance(body["processing_time_seconds"], float)
        assert body["processing_time_seconds"] >= 0.0

        assert "queries_used" in body
        assert isinstance(body["queries_used"], list)

    def test_service_run_called_with_correct_args(
        self,
        client: TestClient,
        valid_request_payload: dict,
        sample_response: ResearchResponse,
    ) -> None:
        """``ResearchService.run`` must be called with the exact request fields."""
        mock_run = AsyncMock(return_value=sample_response)
        with _patch_service(mock_run):
            client.post("/research/", json=valid_request_payload)

        mock_run.assert_awaited_once_with(
            question=valid_request_payload["question"],
            max_results=valid_request_payload["max_results"],
            include_citations=valid_request_payload["include_citations"],
        )

    def test_request_with_citations_disabled(
        self,
        client: TestClient,
        sample_response: ResearchResponse,
    ) -> None:
        """``include_citations=False`` must be forwarded to the service correctly."""
        payload = {
            "question": "What are the main applications of reinforcement learning?",
            "max_results": 10,
            "include_citations": False,
        }
        # Adjust sample response to reflect no citations
        no_citation_response = sample_response.model_copy(
            update={"citations": [], "include_citations": False}
        )
        mock_run = AsyncMock(return_value=no_citation_response)
        with _patch_service(mock_run):
            response = client.post("/research/", json=payload)

        assert response.status_code == 200
        mock_run.assert_awaited_once_with(
            question=payload["question"],
            max_results=payload["max_results"],
            include_citations=False,
        )

    # ------------------------------------------------------------------
    # Error path — domain exceptions → HTTP status codes
    # ------------------------------------------------------------------

    def test_retrieval_error_returns_503(
        self,
        client: TestClient,
        valid_request_payload: dict,
    ) -> None:
        """``RetrievalError`` raised by the service must map to HTTP 503."""
        mock_run = AsyncMock(
            side_effect=RetrievalError("No documents returned from arXiv or Semantic Scholar.")
        )
        with _patch_service(mock_run):
            response = client.post("/research/", json=valid_request_payload)

        assert response.status_code == 503
        detail = response.json().get("detail", "")
        assert "retrieval" in detail.lower() or "document" in detail.lower()

    def test_agent_error_returns_502(
        self,
        client: TestClient,
        valid_request_payload: dict,
    ) -> None:
        """``AgentError`` raised by the service must map to HTTP 502."""
        mock_run = AsyncMock(
            side_effect=AgentError("LLM returned an unparseable response.")
        )
        with _patch_service(mock_run):
            response = client.post("/research/", json=valid_request_payload)

        assert response.status_code == 502
        detail = response.json().get("detail", "")
        assert len(detail) > 0

    def test_value_error_returns_422(
        self,
        client: TestClient,
        valid_request_payload: dict,
    ) -> None:
        """``ValueError`` raised by the service must map to HTTP 422."""
        mock_run = AsyncMock(
            side_effect=ValueError("max_results exceeds allowed cap.")
        )
        with _patch_service(mock_run):
            response = client.post("/research/", json=valid_request_payload)

        assert response.status_code == 422

    def test_unexpected_exception_returns_500(
        self,
        client: TestClient,
        valid_request_payload: dict,
    ) -> None:
        """Any other exception raised by the service must map to HTTP 500."""
        mock_run = AsyncMock(
            side_effect=RuntimeError("Unexpected infrastructure failure.")
        )
        with _patch_service(mock_run):
            response = client.post("/research/", json=valid_request_payload)

        assert response.status_code == 500

    def test_partial_status_response_is_accepted(
        self,
        client: TestClient,
        valid_request_payload: dict,
        sample_response: ResearchResponse,
    ) -> None:
        """A ``'partial'`` status response (report + error) must still return 200."""
        partial_response = sample_response.model_copy(
            update={"status": "partial", "error_message": "citation_verifier_error: LLM quota exceeded"}
        )
        mock_run = AsyncMock(return_value=partial_response)
        with _patch_service(mock_run):
            response = client.post("/research/", json=valid_request_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "partial"
        assert body["error_message"] is not None


# ===========================================================================
# Test group 4 — Input validation (Pydantic 422 responses)
# ===========================================================================


class TestInputValidation:
    """Tests that FastAPI's Pydantic integration rejects invalid inputs."""

    def test_short_question_returns_422(self, client: TestClient) -> None:
        """A question shorter than 10 characters must return HTTP 422."""
        response = client.post(
            "/research/",
            json={"question": "Short?", "max_results": 5, "include_citations": True},
        )
        assert response.status_code == 422
        body = response.json()
        # FastAPI wraps validation errors in a ``detail`` list
        assert "detail" in body

    def test_empty_question_returns_422(self, client: TestClient) -> None:
        """An empty question must return HTTP 422."""
        response = client.post(
            "/research/",
            json={"question": "", "max_results": 5, "include_citations": True},
        )
        assert response.status_code == 422

    def test_question_exceeding_max_length_returns_422(self, client: TestClient) -> None:
        """A question longer than 500 characters must return HTTP 422."""
        long_question = "a" * 501
        response = client.post(
            "/research/",
            json={"question": long_question, "max_results": 5, "include_citations": True},
        )
        assert response.status_code == 422

    def test_max_results_zero_returns_422(self, client: TestClient) -> None:
        """``max_results=0`` is below the minimum of 1 — must return HTTP 422."""
        response = client.post(
            "/research/",
            json={
                "question": "What are recent advances in transformer models?",
                "max_results": 0,
                "include_citations": True,
            },
        )
        assert response.status_code == 422

    def test_max_results_above_cap_returns_422(self, client: TestClient) -> None:
        """``max_results=31`` exceeds the schema maximum of 30 — must return HTTP 422."""
        response = client.post(
            "/research/",
            json={
                "question": "What are recent advances in transformer models?",
                "max_results": 31,
                "include_citations": True,
            },
        )
        assert response.status_code == 422

    def test_missing_question_field_returns_422(self, client: TestClient) -> None:
        """Omitting the required ``question`` field must return HTTP 422."""
        response = client.post(
            "/research/",
            json={"max_results": 5, "include_citations": True},
        )
        assert response.status_code == 422

    def test_missing_optional_fields_use_defaults(
        self,
        client: TestClient,
        sample_response: ResearchResponse,
    ) -> None:
        """Omitting optional fields must use schema defaults and not error."""
        mock_run = AsyncMock(return_value=sample_response)
        with _patch_service(mock_run):
            response = client.post(
                "/research/",
                json={"question": "What are the effects of climate change on biodiversity?"},
            )
        assert response.status_code == 200
        # Confirm defaults were applied
        mock_run.assert_awaited_once_with(
            question="What are the effects of climate change on biodiversity?",
            max_results=10,           # Schema default
            include_citations=True,   # Schema default
        )

    def test_non_json_body_returns_422(self, client: TestClient) -> None:
        """Sending a plain-text body instead of JSON must return HTTP 422."""
        response = client.post(
            "/research/",
            content="this is not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_wrong_type_for_max_results_returns_422(self, client: TestClient) -> None:
        """Passing a string for ``max_results`` must fail Pydantic validation."""
        response = client.post(
            "/research/",
            json={
                "question": "What are recent advances in transformer models?",
                "max_results": "ten",   # Must be int
                "include_citations": True,
            },
        )
        assert response.status_code == 422


# ===========================================================================
# Test group 5 — CORS headers
# ===========================================================================


class TestCORSHeaders:
    """Tests that CORS middleware injects the correct headers."""

    def test_cors_header_present_for_allowed_origin(self, client: TestClient) -> None:
        """A request from an allowed Streamlit origin must receive CORS headers."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:8501"},
        )
        # FastAPI/Starlette sets this header when the origin matches
        assert "access-control-allow-origin" in response.headers

    def test_cors_preflight_options_returns_200(self, client: TestClient) -> None:
        """OPTIONS pre-flight for the research endpoint must return 200."""
        response = client.options(
            "/research/",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        # CORS middleware handles OPTIONS and returns 200 OK
        assert response.status_code == 200

    def test_cors_preflight_exposes_allowed_methods(self, client: TestClient) -> None:
        """OPTIONS pre-flight response must list at least POST and GET."""
        response = client.options(
            "/research/",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        methods_header = response.headers.get("access-control-allow-methods", "")
        # Either explicit methods list or wildcard
        assert "POST" in methods_header or "*" in methods_header

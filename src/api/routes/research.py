"""Research API endpoints.

This module defines two endpoints under the ``/research`` prefix:

POST /research/
    Accepts a ``ResearchRequest`` body, delegates execution to the
    ``ResearchService``, and returns a ``ResearchResponse`` on success.

GET /research/status
    Lightweight endpoint to confirm the research router is registered and
    the service instance was successfully initialised at startup.

Error mapping
-------------
The route layer translates domain exceptions raised by ``ResearchService``
into appropriate HTTP status codes so callers receive actionable responses:

* ``RetrievalError``  → 503 Service Unavailable (upstream sources unreachable)
* ``AgentError``      → 502 Bad Gateway (LLM/agent pipeline failure)
* ``ValueError``      → 422 Unprocessable Entity (bad input slipped through)
* Everything else     → 500 Internal Server Error

Each error response body follows the FastAPI ``HTTPException`` detail format::

    {"detail": "<human-readable message>"}

The service instance is created **once** at module import time (application
startup) rather than per-request to amortise the cost of ``build_graph()``.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.core.exceptions import AgentError, RetrievalError
from src.core.logger import get_logger
from src.schemas.request import ResearchRequest
from src.schemas.response import ResearchResponse
from src.services.research_service import ResearchService

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

# ---------------------------------------------------------------------------
# Application-lifetime singleton — initialised once at import time.
# FastAPI guarantees modules are imported during ``app.include_router`` which
# happens inside ``create_app()`` in ``main.py``, so the graph is fully built
# before the first request arrives.
# ---------------------------------------------------------------------------
_service = ResearchService()


# ---------------------------------------------------------------------------
# POST /research/
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Research Pipeline",
    description=(
        "Accepts a natural language research question, runs the full "
        "multi-agent LangGraph pipeline (Planner → Retrieval → Summarizer → "
        "Citation Verifier → Report Generator), and returns a structured "
        "Markdown research report with citations and metadata."
    ),
    responses={
        200: {"description": "Research completed successfully."},
        422: {"description": "Validation error — question too short/long or invalid parameters."},
        502: {"description": "Agent or LLM pipeline failure."},
        503: {"description": "Document retrieval failed — no papers returned from sources."},
        500: {"description": "Unexpected internal server error."},
    },
)
async def run_research(
    request: ResearchRequest,
    http_request: Request,
) -> ResearchResponse:
    """Execute a full research pipeline for the given question.

    FastAPI will automatically return HTTP 422 with validation details if the
    ``ResearchRequest`` body fails Pydantic validation (e.g. question shorter
    than 10 characters).

    Args:
        request: The validated ``ResearchRequest`` body.
        http_request: The raw Starlette request, used to extract the client IP
            for logging purposes.

    Returns:
        A ``ResearchResponse`` containing the report, citations, and metadata.

    Raises:
        HTTPException 503: If ``RetrievalError`` is raised (no papers found).
        HTTPException 502: If ``AgentError`` is raised (LLM/agent failure).
        HTTPException 422: If ``ValueError`` is raised (bad parameter values).
        HTTPException 500: For any other unexpected exception.
    """
    client_host = getattr(http_request.client, "host", "unknown")
    logger.info(
        "POST /research/ — client=%s question_len=%d max_results=%d include_citations=%s",
        client_host,
        len(request.question),
        request.max_results,
        request.include_citations,
    )

    try:
        response = await _service.run(
            question=request.question,
            max_results=request.max_results,
            include_citations=request.include_citations,
        )
    except RetrievalError as exc:
        logger.warning("RetrievalError: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Document retrieval failed — no academic papers were returned "
                f"for the given question. This may be a transient network issue "
                f"with arXiv or Semantic Scholar. Details: {exc}"
            ),
        ) from exc
    except AgentError as exc:
        logger.error("AgentError: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"The research agent pipeline encountered a failure. "
                f"This is typically caused by an LLM API error or an unexpected "
                f"response format. Details: {exc}"
            ),
        ) from exc
    except ValueError as exc:
        logger.warning("ValueError in research pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid parameter value: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error in run_research: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected internal error occurred. Please try again or "
                "contact support if the problem persists."
            ),
        ) from exc

    logger.info(
        "POST /research/ completed — status=%s docs=%d time=%.2fs",
        response.status,
        response.docs_retrieved,
        response.processing_time_seconds,
    )
    return response


# ---------------------------------------------------------------------------
# GET /research/status
# ---------------------------------------------------------------------------

@router.get(
    "/status",
    summary="Research Service Status",
    description="Returns a brief status payload confirming the research router is alive and the service is initialised.",
    responses={
        200: {
            "description": "Router is registered and service is ready.",
            "content": {
                "application/json": {
                    "example": {"status": "running", "service": "initialised"}
                }
            },
        }
    },
)
async def get_status() -> dict:
    """Return a static status payload indicating the research router is live.

    This endpoint is intentionally lightweight — it performs no I/O and always
    returns HTTP 200 as long as the process is healthy.

    Returns:
        A dict with ``status`` and ``service`` keys.
    """
    return {"status": "running", "service": "initialised"}

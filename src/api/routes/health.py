"""Health-check endpoint for Docker, load balancers, and uptime monitors.

GET /health
-----------
Returns HTTP 200 with a JSON body confirming the service is alive.
This endpoint intentionally performs *no* downstream I/O (no DB ping, no LLM
call) so it stays fast even under load and cannot be blocked by upstream
dependency failures.

The ``HealthResponse`` Pydantic model is re-exported from
``src.schemas.response`` so the OpenAPI schema stays accurate and the response
contract is type-checked by FastAPI at runtime.

Docker HEALTHCHECK example
--------------------------
In your ``Dockerfile`` or ``docker-compose.yml`` use::

    HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
        CMD curl -f http://localhost:8000/health || exit 1
"""
from __future__ import annotations

from fastapi import APIRouter

from src.core.config import settings
from src.schemas.response import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description=(
        "Returns the current liveness status of the API server. "
        "Designed for Docker health checks and uptime monitoring. "
        "Always returns HTTP 200 when the server process is running."
    ),
    responses={
        200: {
            "description": "Service is healthy and accepting requests.",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "version": "1.0.0"}
                }
            },
        }
    },
)
def health_check() -> HealthResponse:
    """Liveness probe — returns status ``'ok'`` and the running app version.

    This handler is intentionally **synchronous** (no ``async``) to avoid any
    event-loop overhead for a purely in-memory operation.

    Returns:
        ``HealthResponse`` with ``status="ok"`` and the configured app version.
    """
    return HealthResponse(status="ok", version=settings.app_version)

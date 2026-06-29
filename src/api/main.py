"""FastAPI application factory for the Multi-Agent Research Assistant API.

Application structure
---------------------
The app is assembled using the **factory pattern** (``create_app``) rather
than a bare module-level ``app = FastAPI(...)`` to make it easy to:

* Inject test overrides via ``app.dependency_overrides``.
* Re-create a clean application instance in each test without importing a
  stale global.
* Swap middleware or router configuration between environments.

A module-level ``app`` singleton is still exposed so ``uvicorn`` can be
pointed at ``src.api.main:app`` without any extra launcher code.

Startup/Shutdown lifecycle
--------------------------
The ``@app.on_event("startup")`` handler logs confirmed readiness **after**
all routers (and therefore the ``ResearchService`` singleton) have been
registered.  This mirrors the pattern used by Gunicorn/uvicorn worker hooks.

OpenAPI documentation
---------------------
* Swagger UI: ``http://localhost:8000/docs``
* ReDoc:       ``http://localhost:8000/redoc``
* OpenAPI JSON: ``http://localhost:8000/openapi.json``
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.middleware.cors import add_cors
from src.api.routes import health as health_router_module
from src.api.routes import research as research_router_module
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan context manager (preferred over deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown events.

    Startup
    -------
    * Logs the application name and version.
    * Confirms all registered routes are reachable.

    Shutdown
    --------
    * Logs a graceful shutdown message.

    Args:
        app: The FastAPI application instance (injected by Starlette).

    Yields:
        Control to the running application until shutdown is triggered.
    """
    # ---- Startup ----
    logger.info(
        "Starting %s v%s — %d routes registered.",
        settings.app_name,
        settings.app_version,
        len(app.routes),
    )
    yield
    # ---- Shutdown ----
    logger.info("%s shutting down gracefully.", settings.app_name)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Assemble and return a fully configured FastAPI application instance.

    Call this function once at module level to obtain the ``app`` object used
    by uvicorn.  Callers may also invoke it inside test fixtures to obtain a
    clean, isolated application instance.

    Returns:
        A ``FastAPI`` instance with CORS middleware and all routers registered.
    """
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "A multi-agent AI system that autonomously researches academic "
            "topics by querying arXiv and Semantic Scholar, summarising papers "
            "with an OpenAI-compatible model, verifying citations with FAISS, and generating a "
            "structured Markdown research report.\n\n"
            "**Phase 4 — API Layer**\n\n"
            "Orchestrated by LangGraph; exposed via FastAPI."
        ),
        contact={
            "name": "Research Agent Project",
            "url": "https://github.com/your-org/agentic-research-assistant",
        },
        license_info={
            "name": "MIT",
        },
        openapi_tags=[
            {
                "name": "research",
                "description": "Endpoints for executing the full multi-agent research pipeline.",
            },
            {
                "name": "health",
                "description": "Liveness and readiness probes for Docker and uptime monitors.",
            },
        ],
        lifespan=lifespan,
    )

    # ---- Middleware (must be added before routers) ----
    add_cors(application)

    # ---- Routers ----
    application.include_router(health_router_module.router)
    application.include_router(research_router_module.router)

    return application


# ---------------------------------------------------------------------------
# Module-level singleton — uvicorn entry point: src.api.main:app
# ---------------------------------------------------------------------------

app: FastAPI = create_app()

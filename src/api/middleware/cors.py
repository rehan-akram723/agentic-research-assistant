"""CORS middleware configuration for the FastAPI application.

The ``add_cors`` factory function is called once at application startup in
``src/api/main.py``.  Keeping CORS configuration in its own module makes it
easy to tighten origins per environment without touching the app entrypoint.

Streamlit UI origins
--------------------
During local development the Streamlit app runs at ``http://localhost:8501``.
In production (Streamlit Community Cloud) the origin will be the deployed URL.
Both are included by default.  For maximum security in a production release,
replace the ``allow_origins`` list with the exact origin(s) of your Streamlit
deployment rather than the wildcard.

Security note
-------------
``allow_credentials=True`` is intentionally **not** set to ``True`` here
because the API uses API-key authentication via request headers, not cookies.
Enabling credentials together with a wildcard ``allow_origins`` would be
rejected by browsers anyway, so we keep the defaults safe.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Allowed origins
# ---------------------------------------------------------------------------
# Add any Streamlit Cloud deployment URL here once the project is deployed.
# Example: "https://your-app.streamlit.app"
_ALLOWED_ORIGINS: list[str] = [
    "http://localhost:8501",       # Streamlit dev server (default port)
    "http://127.0.0.1:8501",       # Alternative localhost notation
    "http://localhost:3000",       # Common React / other frontend dev port
    "http://127.0.0.1:3000",
]

_ALLOWED_METHODS: list[str] = [
    "GET",
    "POST",
    "OPTIONS",
]

_ALLOWED_HEADERS: list[str] = [
    "Content-Type",
    "Accept",
    "Authorization",
    "X-Request-ID",
]


def add_cors(app: FastAPI) -> None:
    """Attach ``CORSMiddleware`` to *app* with project-appropriate settings.

    This function is idempotent — calling it multiple times on the same app
    instance will attach multiple middleware layers, so only call it once
    during application initialisation.

    Args:
        app: The FastAPI application instance to configure.
    """
    app.add_middleware(
        CORSMiddleware,
        # In development we allow all local Streamlit origins; tighten this
        # list before deploying to a public-facing environment.
        allow_origins=_ALLOWED_ORIGINS,
        # Wildcard allowed methods is fine for a research API that has no
        # sensitive mutation endpoints — all writes are idempotent research runs.
        allow_methods=_ALLOWED_METHODS,
        allow_headers=_ALLOWED_HEADERS,
        # Do NOT set allow_credentials=True unless you add cookie-based auth.
        allow_credentials=False,
        # Cache the pre-flight OPTIONS response for 10 minutes to reduce latency.
        max_age=600,
    )

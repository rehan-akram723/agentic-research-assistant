"""Shared OpenAI-compatible LLM helper with metrics instrumentation."""
from __future__ import annotations

from src.core.config import settings
from src.core.exceptions import AgentError
from src.core.metrics import get_current_metrics

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def call_openrouter_completion(prompt: str, *, temperature: float = 0.2) -> str:
    """Send a prompt to the configured model and return the text content.

    Records token counts and estimated cost into the active RunMetrics
    (if one is bound to the current execution context).
    """
    if OpenAI is None:
        raise AgentError("openai is not installed.")
    if not settings.openrouter_api_key:
        raise AgentError("OpenRouter API key is not configured.")

    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    metrics = get_current_metrics()

    try:
        response = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[
                {
                    "role": "system",
                    "content": "You must return exactly one valid JSON object and no extra text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception:
        if metrics:
            metrics.record_llm_call(settings.openrouter_model, 0, 0, success=False)
        raise

    choices = getattr(response, "choices", [])
    if not choices:
        raise AgentError("OpenRouter returned no choices.")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if not content:
        raise AgentError("OpenRouter returned no usable content.")

    # Record usage if the response carried token counts.
    if metrics:
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        metrics.record_llm_call(
            settings.openrouter_model,
            prompt_tokens,
            completion_tokens,
            success=True,
        )

    return str(content)
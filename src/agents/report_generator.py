"""Report generator agent for Phase 3.

The report generator synthesizes the final Markdown report from the
per-document summaries and verified citations produced by earlier pipeline
stages.

Key stability features introduced in this version
--------------------------------------------------
1. **JSON-Schema-Aware prompt** — The LLM is shown the exact JSON structure
   it must return, with explicit type constraints and ``[]`` fallback
   instructions.  Field names in the prompt exactly match the ``ReportPayload``
   Pydantic model.

2. **Self-Healing Repair Loop** — After the first LLM call, if Pydantic
   validation raises ``ValidationError``, a single *repair prompt* is sent that
   includes the original error details.  This gives the model a second chance
   before the agent falls back to the deterministic ``_fallback_payload``.

3. **Strict JSON extraction** — The helper ``_extract_json`` strips Markdown
   code fences and leading/trailing noise before attempting ``json.loads``.
   This handles the common case where the model wraps JSON in ``` blocks.

4. **Deterministic fallback** — If both LLM attempts fail, ``_fallback_payload``
   builds a structurally valid ``ReportPayload`` purely from state values,
   guaranteeing the pipeline always produces a non-empty report.
"""
from __future__ import annotations

import json
import re
from typing import List

from pydantic import ValidationError

from src.core.exceptions import AgentError
from src.core.logger import get_logger
from src.core.llm import call_openrouter_completion
from src.core.metrics import track_latency
from src.graph.state import ResearchState
from src.schemas.document import Citation
from src.schemas.report import ReportPayload

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON Schema-Aware prompt template
# ---------------------------------------------------------------------------

# The prompt uses an *inline JSON Schema* block so the model understands the
# exact structure. Field names in the schema block MUST match ReportPayload
# exactly. The three required LLM fields are: methods_and_scope,
# limitations_and_gaps, next_steps — all must be lists of strings.
_REPORT_SCHEMA_BLOCK = """\
Return ONLY valid JSON. Do NOT include markdown fences, prose, or extra keys.
The JSON object must have exactly these three keys:

{
  "methods_and_scope":    <list of strings — research methods and scope found in the literature. If no data, return []>,
  "limitations_and_gaps": <list of strings — limitations and knowledge gaps identified. If no data, return []>,
  "next_steps":           <list of strings — recommended follow-on actions. If no data, return []>
}

Rules:
- Every value MUST be a JSON array of strings (e.g. ["...", "..."]).
- An empty array [] is a valid value — never use null or omit the key.
- Do not add any key not listed above.
- Do not wrap the JSON in ```json or any markdown.
"""

_REPORT_USER_TEMPLATE = """\
{schema}

Research question: {question}

Document summaries:
{summaries_block}

Verified citations:
{citations_block}
"""

_REPAIR_TEMPLATE = """\
Your previous response failed schema validation with this error:

{error_details}

{schema}

Research question: {question}

Document summaries:
{summaries_block}

Verified citations:
{citations_block}

Fix only the structure. The JSON values must be lists of strings.\
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> str:
    """Strip Markdown code fences and surrounding whitespace from *raw*.

    Handles both ```json ... ``` and ``` ... ``` wrappers. If the text has no
    fences, returns it stripped as-is.

    Args:
        raw: The raw string returned by the LLM.

    Returns:
        A string that should be parseable by ``json.loads``.
    """
    # Remove ```json ... ``` or ``` ... ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return cleaned.strip()


def _build_prompt(question: str, summaries: List[str], citations: List[Citation]) -> str:
    """Construct the primary JSON-Schema-Aware prompt.

    Args:
        question: The research question.
        summaries: Flat list of per-document summary strings.
        citations: Validated Citation objects.

    Returns:
        A formatted prompt string ready for the LLM.
    """
    summaries_block = (
        json.dumps(summaries, ensure_ascii=False, indent=2)
        if summaries
        else "[]"
    )
    citation_lines = [
        f"- claim: {c.claim}\n  source: {c.source_title}\n  verified: {c.is_verified}\n  quote: {c.quote or ''}"
        for c in citations
    ]
    citations_block = "\n".join(citation_lines) if citation_lines else "[]"

    return _REPORT_USER_TEMPLATE.format(
        schema=_REPORT_SCHEMA_BLOCK,
        question=question or "N/A",
        summaries_block=summaries_block,
        citations_block=citations_block,
    )


def _build_repair_prompt(
    error_details: str,
    question: str,
    summaries: List[str],
    citations: List[Citation],
) -> str:
    """Construct the repair prompt that includes the validation error details.

    Args:
        error_details: The string representation of the ValidationError.
        question: The research question.
        summaries: Same summaries passed to the first attempt.
        citations: Same citations passed to the first attempt.

    Returns:
        A formatted repair prompt string.
    """
    summaries_block = (
        json.dumps(summaries, ensure_ascii=False, indent=2)
        if summaries
        else "[]"
    )
    citation_lines = [
        f"- claim: {c.claim}\n  source: {c.source_title}\n  verified: {c.is_verified}\n  quote: {c.quote or ''}"
        for c in citations
    ]
    citations_block = "\n".join(citation_lines) if citation_lines else "[]"

    return _REPAIR_TEMPLATE.format(
        error_details=error_details,
        schema=_REPORT_SCHEMA_BLOCK,
        question=question or "N/A",
        summaries_block=summaries_block,
        citations_block=citations_block,
    )


def _parse_payload(raw: str) -> ReportPayload:
    """Extract JSON from *raw*, parse it, and validate against ``ReportPayload``.

    Args:
        raw: The raw LLM response string (may contain Markdown fences).

    Returns:
        A validated ``ReportPayload`` instance.

    Raises:
        json.JSONDecodeError: If the extracted content is not valid JSON.
        ValidationError: If the JSON does not conform to ``ReportPayload``.
    """
    extracted = _extract_json(raw)
    data = json.loads(extracted)
    if not isinstance(data, dict):
        raise ValueError(f"LLM returned JSON type {type(data).__name__}, expected object (dict).")
    return ReportPayload.model_validate(data)


def _call_model_with_repair(
    question: str,
    summaries: List[str],
    citations: List[Citation],
) -> ReportPayload:
    """Invoke the LLM with the schema-aware prompt and attempt a single repair.

    Flow
    ----
    1. Build and send the primary prompt.
    2. Attempt ``_parse_payload`` on the response.
    3. **If** ``ValidationError`` (or ``json.JSONDecodeError``) is raised:
       a. Log the error details.
       b. Build a repair prompt that includes those error details.
       c. Send the repair prompt to the LLM (second call).
       d. Attempt ``_parse_payload`` on the repair response.
    4. If the repair also fails, propagate the exception so the caller can
       fall back to ``_fallback_payload``.

    Args:
        question: The research question.
        summaries: Per-document summary strings.
        citations: Validated Citation objects.

    Returns:
        A validated ``ReportPayload``.

    Raises:
        Exception: If both the initial call and the repair call fail.
    """
    primary_prompt = _build_prompt(question, summaries, citations)

    # ---- First attempt ----
    raw_response = call_openrouter_completion(primary_prompt, temperature=0.2)
    logger.debug("report_generator: raw LLM response (first attempt): %s", raw_response[:300])

    try:
        return _parse_payload(raw_response)

    except (ValidationError, json.JSONDecodeError, ValueError) as first_error:
        # Capture the full error details to embed in the repair prompt.
        error_details = str(first_error)
        logger.warning(
            "report_generator: first LLM response failed validation — issuing repair prompt. "
            "Error: %s",
            error_details,
        )

        # ---- Repair attempt ----
        repair_prompt = _build_repair_prompt(error_details, question, summaries, citations)
        raw_repair = call_openrouter_completion(repair_prompt, temperature=0.0)
        logger.debug("report_generator: raw LLM response (repair attempt): %s", raw_repair[:300])

        try:
            repaired = _parse_payload(raw_repair)
            logger.info("report_generator: repair attempt succeeded.")
            return repaired
        except (ValidationError, json.JSONDecodeError, ValueError) as repair_error:
            logger.error(
                "report_generator: repair attempt also failed. "
                "Falling back to deterministic payload. Repair error: %s",
                repair_error,
            )
            raise repair_error


def _fallback_payload(
    question: str, summaries: List[str], citations: List[Citation]
) -> ReportPayload:
    """Build a structurally-valid ``ReportPayload`` from state values alone.

    Called when both LLM attempts fail. Never raises — guarantees the pipeline
    produces a non-empty ``ReportPayload``.

    Args:
        question: The research question.
        summaries: Per-document summary strings (may be empty).
        citations: Verified Citation objects (may be empty).

    Returns:
        A fully-populated ``ReportPayload``.
    """
    key_findings: List[str] = []
    for summary in summaries:
        lines = summary.splitlines()
        # Try to find the "Summary:" line; fall back to the first non-empty line.
        summary_line = next(
            (
                line.split(":", 1)[1].strip()
                for line in lines
                if line.lower().startswith("summary:") and ":" in line
            ),
            lines[0].strip() if lines else summary.strip(),
        )
        if summary_line:
            key_findings.append(summary_line[:280])

    methods = [f"Reviewed {len(summaries)} retrieved document summaries."]
    if question:
        methods.append(f"Focused on the research question: {question}")

    if citations:
        unverified = sum(1 for c in citations if not c.is_verified)
        limitations = [
            f"{unverified} of {len(citations)} citation(s) could not be fully verified."
        ]
    else:
        limitations = ["No citations were available for verification."]

    next_steps = [
        "Expand retrieval with more targeted queries for weaker subtopics.",
        "Review any unverified claims before using the report externally.",
    ]

    return ReportPayload(
        title="Research Synthesis",
        executive_summary=(
            f"This report summarises the evidence gathered for: {question}"
            if question
            else "This report summarises the evidence gathered during the research run."
        ),
        key_findings=key_findings[:5],
        methods_and_scope=methods,
        limitations_and_gaps=limitations,
        next_steps=next_steps,
    )


def _format_report(
    payload: ReportPayload,
    question: str,
    summaries: List[str],
    citations: List[Citation],
) -> str:
    """Render a ``ReportPayload`` into a Markdown string.

    Args:
        payload: The validated payload from either the LLM or the fallback.
        question: The research question (appended at the end).
        summaries: Used to generate the Source Summaries section.
        citations: Used to render the References section.

    Returns:
        A Markdown-formatted research report string.
    """
    lines = [
        f"# {payload.title}",
        "",
        "## Executive Summary",
        payload.executive_summary.strip() or "No executive summary was generated.",
        "",
        "## Key Findings",
    ]
    if payload.key_findings:
        lines.extend(f"- {item.strip()}" for item in payload.key_findings if item.strip())
    else:
        lines.append("- No key findings were generated.")

    lines.extend(["", "## Methods & Scope"])
    if payload.methods_and_scope:
        lines.extend(f"- {item.strip()}" for item in payload.methods_and_scope if item.strip())
    else:
        lines.append("- No method summary was generated.")

    lines.extend(["", "## Limitations & Gaps"])
    if payload.limitations_and_gaps:
        lines.extend(f"- {item.strip()}" for item in payload.limitations_and_gaps if item.strip())
    else:
        lines.append("- No limitations were generated.")

    lines.extend(["", "## Next Steps"])
    if payload.next_steps:
        lines.extend(f"- {item.strip()}" for item in payload.next_steps if item.strip())
    else:
        lines.append("- No next steps were generated.")

    lines.extend(["", "## References"])
    if citations:
        for c in citations:
            status = "verified" if c.is_verified else "unverified"
            ref = f"- {c.source_title or 'Unknown source'} ({c.source_doc_id or 'n/a'}) [{status}]"
            if c.quote:
                ref += f": {c.quote[:160]}"
            lines.append(ref)
    else:
        lines.append("- No citations were available.")

    if question:
        lines.extend(["", "## Research Question", question])

    if summaries:
        lines.extend(["", "## Source Summaries"])
        for summary in summaries:
            first = summary.splitlines()[0].strip() if summary.splitlines() else summary.strip()
            if first:
                lines.append(f"- {first}")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

@track_latency("report_generator")
def report_generator_agent(state: ResearchState) -> dict:
    """LangGraph node — generate the final structured Markdown research report.

    Reads ``question``, ``summaries``, and ``verified_citations`` from *state*.
    Writes ``final_report`` and appends to ``step_log``.

    Error handling
    --------------
    * Pydantic ``ValidationError`` from the first LLM call → triggers repair prompt.
    * Repair prompt also fails → falls back to ``_fallback_payload``.
    * ``_fallback_payload`` always succeeds.
    * If the assembled report is still empty (should never happen) → raises
      ``AgentError``.

    Args:
        state: The current ``ResearchState`` dict.

    Returns:
        A dict with keys ``final_report`` (str) and ``step_log`` (List[str]).

    Raises:
        AgentError: If the final report is empty after all fallbacks.
    """
    try:
        question = str(state.get("question", "") or "").strip()
        summaries = [
            str(s).strip()
            for s in (state.get("summaries") or [])
            if str(s).strip()
        ]
        citations: List[Citation] = []
        for item in state.get("verified_citations") or []:
            if isinstance(item, Citation):
                citations.append(item)
            elif isinstance(item, dict):
                citations.append(Citation.model_validate(item))

        # --- LLM call with self-healing repair loop ---
        try:
            payload = _call_model_with_repair(question, summaries, citations)
            # Populate display-side fields from state values (not from LLM).
            if not payload.executive_summary:
                payload.executive_summary = (
                    f"This report summarises the evidence gathered for: {question}"
                    if question
                    else "This report summarises the evidence gathered during the research run."
                )
            if not payload.key_findings:
                payload.key_findings = [
                    s.splitlines()[0].strip()[:280]
                    for s in summaries
                    if s.splitlines()
                ][:5]
            if not payload.title or payload.title == "Research Synthesis":
                payload.title = "Research Synthesis"

        except Exception as exc:
            logger.warning(
                "report_generator_agent: both LLM attempts failed — using deterministic fallback. "
                "Reason: %s",
                exc,
            )
            payload = _fallback_payload(question, summaries, citations)

        final_report = _format_report(payload, question, summaries, citations)
        if not final_report.strip():
            raise AgentError("report_generator_agent produced an empty report.")

        step_log = [
            f"report_generator: assembled report "
            f"({len(summaries)} summaries, {len(citations)} citations)"
        ]
        logger.info("report_generator_agent assembled report successfully.")
        return {"final_report": final_report, "step_log": step_log}

    except AgentError:
        raise
    except Exception as exc:
        raise AgentError(f"Report generator failed unexpectedly: {exc}") from exc

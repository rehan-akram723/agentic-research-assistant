"""Report generator for Phase 3.

The report generator synthesizes the final markdown report from summaries and
verified citations. Gemini is used with a parseable JSON response format when
available, and a deterministic fallback is used otherwise.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.exceptions import AgentError
from src.core.logger import get_logger
from src.graph.state import ResearchState
from src.schemas.document import Citation

logger = get_logger(__name__)

try:  # pragma: no cover - import depends on environment
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None


class ReportPayload(BaseModel):
    title: str = Field(default="Research Synthesis")
    executive_summary: str = Field(default="")
    key_findings: List[str] = Field(default_factory=list)
    methods_and_scope: List[str] = Field(default_factory=list)
    limitations_and_gaps: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


def _build_llm() -> object | None:
    if ChatGoogleGenerativeAI is None or not settings.gemini_api_key:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )


def _fallback_payload(question: str, summaries: List[str], citations: List[Citation]) -> ReportPayload:
    key_findings = []
    for summary in summaries:
        summary_line = next((line.split(":", 1)[1].strip() for line in summary.splitlines() if line.lower().startswith("summary:") and ":" in line), summary.splitlines()[0].strip() if summary.splitlines() else summary.strip())
        if summary_line:
            key_findings.append(summary_line[:280])

    methods = [f"Reviewed {len(summaries)} retrieved document summaries."]
    if question:
        methods.append(f"Focused on the research question: {question}")

    limitations = []
    if citations:
        unverified_count = sum(1 for citation in citations if not citation.is_verified)
        limitations.append(f"{unverified_count} citation(s) were not fully verified against the supplied source material.")
    else:
        limitations.append("No citations were available for verification.")

    next_steps = [
        "Expand retrieval with more targeted queries for weaker subtopics.",
        "Review any unverified claims before using the report externally.",
    ]

    return ReportPayload(
        title="Research Synthesis",
        executive_summary=(
            f"This report summarizes the evidence gathered for: {question}"
            if question
            else "This report summarizes the evidence gathered during the research run."
        ),
        key_findings=key_findings[:5],
        methods_and_scope=methods,
        limitations_and_gaps=limitations,
        next_steps=next_steps,
    )


def _parse_payload(payload: str) -> ReportPayload:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Report payload must be a JSON object")
    return ReportPayload.model_validate(data)


def _call_gemini(question: str, summaries: List[str], citations: List[Citation]) -> ReportPayload:
    if ChatGoogleGenerativeAI is None:
        raise AgentError("langchain-google-genai is not installed.")
    if not settings.gemini_api_key:
        raise AgentError("Gemini API key is not configured.")

    citation_lines = [
        f"- claim: {citation.claim}\n  source: {citation.source_title}\n  verified: {citation.is_verified}\n  quote: {citation.quote or ''}"
        for citation in citations
    ]
    prompt = (
        "You are generating a structured markdown research report. Return JSON only with keys title, "
        "executive_summary, key_findings, methods_and_scope, limitations_and_gaps, and next_steps. "
        "Keep the content concise, evidence-grounded, and directly tied to the supplied summaries and citations.\n\n"
        f"Research question: {question or 'N/A'}\n\n"
        f"Summaries:\n{json.dumps(summaries, ensure_ascii=False, indent=2)}\n\n"
        f"Verified citations:\n{chr(10).join(citation_lines) if citation_lines else '[]'}"
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )
    response = model.invoke(prompt)
    content = getattr(response, "content", response)
    return _parse_payload(str(content))


def _format_report(payload: ReportPayload, question: str, summaries: List[str], citations: List[Citation]) -> str:
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
        for citation in citations:
            status = "verified" if citation.is_verified else "unverified"
            reference = f"- {citation.source_title or 'Unknown source'} ({citation.source_doc_id or 'n/a'}) [{status}]"
            if citation.quote:
                reference += f": {citation.quote[:160]}"
            lines.append(reference)
    else:
        lines.append("- No citations were available.")

    if question:
        lines.extend(["", "## Research Question", question])

    if summaries:
        lines.extend(["", "## Source Summaries"])
        for summary in summaries:
            first_line = summary.splitlines()[0].strip() if summary.splitlines() else summary.strip()
            if first_line:
                lines.append(f"- {first_line}")

    return "\n".join(lines).strip()


def report_generator_agent(state: ResearchState) -> dict:
    try:
        question = str(state.get("question", "") or "").strip()
        summaries = [str(summary).strip() for summary in (state.get("summaries") or []) if str(summary).strip()]
        citations: List[Citation] = []
        for citation in state.get("verified_citations") or []:
            if isinstance(citation, Citation):
                citations.append(citation)
            elif isinstance(citation, dict):
                if hasattr(Citation, "model_validate"):
                    citations.append(Citation.model_validate(citation))  # type: ignore[attr-defined]
                else:
                    citations.append(Citation(**citation))

        try:
            payload = _call_gemini(question, summaries, citations)
        except Exception as exc:
            logger.warning("report_generator_agent falling back to deterministic report: %s", exc)
            payload = _fallback_payload(question, summaries, citations)

        final_report = _format_report(payload, question, summaries, citations)
        if not final_report:
            raise AgentError("report_generator_agent produced an empty report.")

        step_log = [f"report_generator: assembled report ({len(summaries)} summaries, {len(citations)} citations)"]
        logger.info("report_generator_agent assembled report")
        return {"final_report": final_report, "step_log": step_log}
    except AgentError:
        raise
    except Exception as exc:
        raise AgentError(f"Report generator failed: {exc}") from exc

"""Report generator: assembles final markdown report from summaries & citations.

Produces a structured markdown document following the required sections.
"""
from __future__ import annotations

from typing import List
from src.graph.state import ResearchState
from src.schemas.document import Citation


def report_generator_agent(state: ResearchState) -> dict:
    question = state.get("question", "")
    summaries: List[str] = state.get("summaries", []) or []
    citations: List[Citation] = state.get("verified_citations", []) or []

    overview = f"## Overview\n\nQuestion: {question}\n\n"

    methods = "## Key Methods & Approaches\n\n"
    findings = "## Findings & Results\n\n"
    limitations = "## Limitations & Gaps\n\n"

    for s in summaries:
        methods += f"- {s.split('\n')[0]}\n"
        findings += f"- {s[:300]}\n"

    for c in citations:
        limitations += f"- Citation: {c.source_title} — {'verified' if c.is_verified else 'unverified'}\n"

    references = "## References\n\n"
    for c in citations:
        if c.source_doc_id and c.source_title:
            references += f"- {c.source_title} ({c.source_doc_id})\n"

    final = "\n".join([overview, methods, findings, limitations, references])

    step_log = [f"report_generator: assembled report ({len(summaries)} summaries, {len(citations)} citations)"]
    return {"final_report": final, "step_log": step_log}

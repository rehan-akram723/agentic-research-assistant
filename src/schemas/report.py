"""ReportPayload: the canonical Pydantic schema for the report generator's LLM output.

This model lives here (``src/schemas/report.py``) rather than inline in the
agent so that it can be imported by:

* ``src/agents/report_generator.py`` — for validation and formatting.
* Any future test that needs to construct a canned payload without importing
  the full agent module.

Contract alignment
------------------
The JSON structure expected from the LLM is intentionally minimal. Only the
three *structural array fields* are required — the remaining display fields
(title, executive_summary, key_findings) are *filled deterministically* by the
agent after the LLM call succeeds.  Keeping the LLM contract small reduces the
surface area for ValidationError failures.

LLM-returned JSON must look exactly like::

    {
        "methods_and_scope":   ["...", "..."],
        "limitations_and_gaps": ["...", "..."],
        "next_steps":          ["..."]
    }

Display-side fields (title, executive_summary, key_findings) are populated
from ``state`` values after the LLM returns, never by the LLM itself.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class ReportPayload(BaseModel):
    """Validated output payload produced by the report generator LLM call.

    Only ``methods_and_scope``, ``limitations_and_gaps``, and ``next_steps``
    are sourced from the LLM.  All other fields are populated by the agent
    after the fact and are therefore *not* required in the LLM JSON response.

    Attributes:
        methods_and_scope:   Bullet-level descriptions of the research methods
                             and scope of the reviewed literature.
        limitations_and_gaps: Known shortcomings or open questions identified
                              from the literature.
        next_steps:          Actionable recommendations for follow-on research.
        title:               Report title — set by the agent, not the LLM.
        executive_summary:   Brief narrative summary — set by the agent.
        key_findings:        Top findings extracted from summaries — set by
                             the agent from the ``summaries`` state field.
    """

    # ------------------------------------------------------------------ #
    # LLM-sourced fields (must be present in the JSON the model returns)  #
    # ------------------------------------------------------------------ #

    methods_and_scope: List[str] = Field(
        default_factory=list,
        description=(
            "List of strings describing the research methods and scope. "
            "Return [] if no data is available."
        ),
    )
    limitations_and_gaps: List[str] = Field(
        default_factory=list,
        description=(
            "List of strings describing limitations and knowledge gaps. "
            "Return [] if no data is available."
        ),
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description=(
            "List of strings describing recommended next steps. "
            "Return [] if no data is available."
        ),
    )

    executive_summary: str = Field(
        default="",
        description="A deeply synthesized, specific executive summary of the findings. Must not be generic.",
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="List of synthesized, highly specific insights derived across the papers.",
    )

    # ------------------------------------------------------------------ #
    # Agent-populated display fields (not expected from the LLM)          #
    # ------------------------------------------------------------------ #

    title: str = Field(default="Research Synthesis")

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    @field_validator("methods_and_scope", "limitations_and_gaps", "next_steps", "key_findings", mode="before")
    @classmethod
    def coerce_to_list_of_strings(cls, value: object) -> List[str]:
        """Accept a list, a bare string, or None — always produce List[str].

        The LLM occasionally returns a single string instead of a list, or
        ``null`` instead of an empty list.  This validator normalises those
        edge cases before Pydantic does its type check, preventing spurious
        ValidationErrors.

        Args:
            value: The raw value from the LLM JSON.

        Returns:
            A list of non-empty stripped strings.
        """
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        # Anything else (dict, int, …) — coerce to string and wrap
        return [str(value).strip()] if str(value).strip() else []

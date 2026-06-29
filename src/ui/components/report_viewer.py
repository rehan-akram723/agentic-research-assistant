"""Report viewer component."""
from typing import Any
import streamlit as st

def render_report(response: dict[str, Any]) -> None:
    status = response.get("status", "unknown")
    report_md = response.get("report", "")
    citations = response.get("citations", [])
    queries = response.get("queries_used", [])
    time_sec = response.get("processing_time_seconds", 0.0)
    
    if status == "error":
        st.error(f"Execution Fault: {response.get('error_message')}")
        return

    mx: dict = response.get("metrics") or {}

    # Metadata ribbon — pulls from both top-level response and metrics snapshot
    tokens = mx.get("total_prompt_tokens", 0) + mx.get("total_completion_tokens", 0)
    cost = mx.get("estimated_cost_usd", 0.0)
    dupes = mx.get("duplicates_removed", 0)

    st.markdown(
        f"""
        <div style="display: flex; flex-wrap: wrap; gap: 3rem; border-bottom: 1px solid #1a1a1a; padding-bottom: 1.5rem; margin-bottom: 2.5rem;">
            <div>
                <div style="font-size: 0.65rem; color: #555555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;">Status</div>
                <div style="color: #d4af37; font-size: 0.9rem; letter-spacing: 1px;">{status.upper()}</div>
            </div>
            <div>
                <div style="font-size: 0.65rem; color: #555555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;">Latency</div>
                <div style="color: #e5e5e5; font-size: 0.9rem; letter-spacing: 1px;">{time_sec:.2f}s</div>
            </div>
            <div>
                <div style="font-size: 0.65rem; color: #555555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;">Sources</div>
                <div style="color: #e5e5e5; font-size: 0.9rem; letter-spacing: 1px;">{response.get('docs_retrieved', 0)}</div>
            </div>
            <div>
                <div style="font-size: 0.65rem; color: #555555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;">Tokens</div>
                <div style="color: #e5e5e5; font-size: 0.9rem; letter-spacing: 1px;">{tokens:,}</div>
            </div>
            <div>
                <div style="font-size: 0.65rem; color: #555555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;">Est. Cost</div>
                <div style="color: #e5e5e5; font-size: 0.9rem; letter-spacing: 1px;">${cost:.4f}</div>
            </div>
            <div>
                <div style="font-size: 0.65rem; color: #555555; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 4px;">Duplicates</div>
                <div style="color: #e5e5e5; font-size: 0.9rem; letter-spacing: 1px;">{dupes}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Document body
    st.markdown(
        '<div style="background: #0a0a0a; border: 1px solid #1a1a1a; padding: 3rem; margin-bottom: 2rem;">',
        unsafe_allow_html=True
    )
    st.markdown(report_md)
    st.markdown("</div>", unsafe_allow_html=True)

    # Citations
    if citations:
        with st.expander("VERIFICATION MATRIX", expanded=False):
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            for i, c in enumerate(citations, 1):
                is_ver = c.get("is_verified", False)
                border = "#d4af37" if is_ver else "#8c1515"
                st.markdown(
                    f"""
                    <div style="border-left: 1px solid {border}; padding-left: 1.5rem; margin-bottom: 2rem;">
                        <div style="color: #e5e5e5; font-size: 0.85rem; margin-bottom: 0.5rem;">
                            <span style="color: {border}; margin-right: 8px;">[{i}]</span> {c.get("source_title", "Unknown")}
                        </div>
                        <div style="color: #888888; font-size: 0.8rem; line-height: 1.5;">Claim: {c.get("claim", "")}</div>
                        <div style="color: #444444; font-size: 0.7rem; margin-top: 0.5rem; font-family: monospace;">ID: {c.get("source_doc_id", "N/A")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    if queries:
        with st.expander("EXECUTION TRACE", expanded=False):
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            for q in queries:
                st.markdown(f"<div style='color: #888888; font-family: monospace; font-size: 0.8rem; margin-bottom: 0.5rem;'>{q}</div>", unsafe_allow_html=True)

    # Telemetry: per-agent latencies from metrics snapshot
    agent_latencies: dict = mx.get("agent_latencies", {})
    if agent_latencies or mx.get("embedding_time_seconds"):
        with st.expander("TELEMETRY", expanded=False):
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
            rows = []
            for agent, data in agent_latencies.items():
                rows.append(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-family:monospace; font-size:0.78rem; color:#888888; "
                    f"border-bottom:1px solid #111; padding:6px 0;'>"
                    f"<span style='color:#e5e5e5;'>{agent}</span>"
                    f"<span>{data.get('calls', 0)} calls &nbsp;|&nbsp; "
                    f"avg {data.get('avg', 0):.3f}s</span></div>"
                )
            embed_s = mx.get("embedding_time_seconds", 0.0)
            if embed_s:
                rows.append(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-family:monospace; font-size:0.78rem; color:#888888; "
                    f"border-bottom:1px solid #111; padding:6px 0;'>"
                    f"<span style='color:#e5e5e5;'>embedding</span>"
                    f"<span>total {embed_s:.3f}s</span></div>"
                )
            api_fail = mx.get("api_failures", 0)
            api_fail_color = "#8c1515" if api_fail else "#888"
            rows.append(
                f"<div style='display:flex; justify-content:space-between; "
                f"font-family:monospace; font-size:0.78rem; color:#888888; padding:6px 0;'>"
                f"<span style='color:#e5e5e5;'>api_failures</span>"
                f"<span style='color:{api_fail_color};'>{api_fail}</span></div>"
            )
            st.markdown("".join(rows), unsafe_allow_html=True)

    if report_md:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="EXPORT DOCUMENT",
            data=report_md.encode("utf-8"),
            file_name="research_synthesis.md",
            mime="text/markdown",
            use_container_width=True
        )

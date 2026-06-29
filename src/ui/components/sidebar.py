"""Sidebar component."""
import streamlit as st
import requests

_API_HEALTH_URL = "http://127.0.0.1:8000/health"

_EXAMPLE_QUERIES = [
    "Low-resource NLP fine-tuning methodologies",
    "Small language model compression advances",
    "Retrieval-augmented generation factual accuracy",
    "Graph neural networks on heterogeneous data",
]

def _check_api_health() -> bool:
    try:
        resp = requests.get(_API_HEALTH_URL, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 1rem 0 2rem 0;">
                <h2 style="margin:0; font-size:0.9rem; font-weight:600; color:#d4af37; letter-spacing: 2px; text-transform: uppercase;">
                    Control Panel
                </h2>
                <div style="height: 1px; width: 40px; background-color: #8c1515; margin-top: 12px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        api_ok = _check_api_health()
        status_col = "#d4af37" if api_ok else "#8c1515"
        status_lbl = "SYSTEM ONLINE" if api_ok else "SYSTEM OFFLINE"
        
        st.markdown(
            f"""
            <div style="margin-bottom: 3rem; display: flex; align-items: center; gap: 12px;">
                <div style="width: 6px; height: 6px; background-color: {status_col}; border-radius: 50%;"></div>
                <span style="font-size: 0.7rem; color: #555555; letter-spacing: 1px;">{status_lbl}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='font-size:0.75rem; color:#888888; letter-spacing:1.5px; margin-bottom:1rem; text-transform: uppercase;'>Parameters</div>", unsafe_allow_html=True)
        max_results = st.slider("Depth", 3, 20, 8, label_visibility="collapsed")
        
        include_citations = st.toggle("Verify Sources", value=True)

        st.markdown("<div style='font-size:0.75rem; color:#888888; letter-spacing:1.5px; margin-top:3rem; margin-bottom:1rem; text-transform: uppercase;'>Presets</div>", unsafe_allow_html=True)
        for q in _EXAMPLE_QUERIES:
            if st.button(q, key=hash(q), use_container_width=True):
                st.session_state["prefill_question"] = q

        st.markdown("<div style='font-size:0.75rem; color:#888888; letter-spacing:1.5px; margin-top:3rem; margin-bottom:1rem; text-transform: uppercase;'>Telemetry</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Ops", st.session_state.get("total_queries", 0))
        t = st.session_state.get("last_processing_time")
        c2.metric("Latency", f"{t:.1f}s" if t else "—")

    return {"max_results": max_results, "include_citations": include_citations}

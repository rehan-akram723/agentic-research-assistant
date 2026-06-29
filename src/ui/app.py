"""Main Streamlit app entry point."""
import time
import requests
import streamlit as st

from src.ui.components.sidebar import render_sidebar
from src.ui.components.query_input import render_query_input
from src.ui.components.report_viewer import render_report

st.set_page_config(
    page_title="Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #050505; color: #e5e5e5; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1000px; }
    
    .stButton > button[kind="primary"] {
        background-color: #000000;
        color: #d4af37;
        border: 1px solid #d4af37;
        border-radius: 2px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #d4af37;
        color: #000000;
        border-color: #d4af37;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }
    
    textarea {
        background-color: #0a0a0a !important;
        color: #e5e5e5 !important;
        border: 1px solid #222222 !important;
        border-radius: 2px !important;
        font-family: monospace !important;
    }
    textarea:focus {
        border-color: #8c1515 !important;
        box-shadow: none !important;
    }
    
    h1, h2, h3, h4, h5 { 
        color: #ffffff !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important; 
    }
    
    [data-testid="stMetricLabel"] { color: #555555 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { color: #d4af37 !important; }
    
    .streamlit-expanderHeader {
        background-color: #050505 !important;
        border-radius: 2px !important;
        color: #888888 !important;
        border: 1px solid #1a1a1a !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
    }
    
    hr { border-color: #1a1a1a !important; }
    
    .stDownloadButton > button {
        background-color: #050505;
        color: #888888;
        border: 1px solid #222222;
        border-radius: 2px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
    }
    .stDownloadButton > button:hover {
        border-color: #d4af37;
        color: #d4af37;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_API_BASE = "http://127.0.0.1:8000"
_RESEARCH_ENDPOINT = f"{_API_BASE}/research/"

if "total_queries" not in st.session_state:
    st.session_state["total_queries"] = 0
if "last_processing_time" not in st.session_state:
    st.session_state["last_processing_time"] = None
if "last_response" not in st.session_state:
    st.session_state["last_response"] = None

sidebar_settings = render_sidebar()

st.markdown(
    """
    <div style="margin-bottom: 3rem;">
        <h1 style="font-size:2.2rem; font-weight:300; letter-spacing: -0.5px; margin-bottom: 0;">
            Research <span style="color: #8c1515;">Assistant</span>
        </h1>
        <p style="color:#555555; font-size:0.85rem; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 2px;">
            Autonomous Synthesis Engine
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

query_data = render_query_input()
question: str = query_data["question"]
submitted: bool = query_data["submitted"]

if submitted and question:
    payload = {
        "question": question,
        "max_results": sidebar_settings["max_results"],
        "include_citations": sidebar_settings["include_citations"],
    }

    with st.status("Initializing sequence...", expanded=True) as status_box:
        st.write("Constructing queries...")
        st.write("Executing retrieval...")
        st.write("Processing literature...")
        if sidebar_settings["include_citations"]:
            st.write("Verifying sources...")
        st.write("Compiling synthesis...")

        wall_start = time.perf_counter()
        try:
            resp = requests.post(_RESEARCH_ENDPOINT, json=payload, timeout=300)
        except requests.exceptions.ConnectionError:
            status_box.update(label="System Offline", state="error")
            st.error("Connection refused. Verify backend services.")
            st.stop()
        except requests.exceptions.Timeout:
            status_box.update(label="Request Timeout", state="error")
            st.error("Operation timed out. Reduce requested scope.")
            st.stop()
        except Exception as exc:
            status_box.update(label="System Error", state="error")
            st.error(f"Unexpected fault: {exc}")
            st.stop()

        wall_elapsed = time.perf_counter() - wall_start

        if resp.status_code == 200:
            status_box.update(label="Sequence Complete", state="complete")
            response_data = resp.json()
            st.session_state["total_queries"] += 1
            st.session_state["last_processing_time"] = response_data.get("processing_time_seconds", wall_elapsed)
            st.session_state["last_response"] = response_data

        elif resp.status_code == 422:
            status_box.update(label="Validation Fault", state="error")
            st.error(f"Invalid parameters: {resp.json().get('detail')}")
            st.stop()
        elif resp.status_code == 503:
            status_box.update(label="Retrieval Fault", state="error")
            st.warning("No documents retrieved for the given parameters.")
            st.stop()
        else:
            status_box.update(label="System Fault", state="error")
            st.error(f"HTTP {resp.status_code}")
            st.stop()

if st.session_state.get("last_response"):
    st.divider()
    render_report(st.session_state["last_response"])
elif not submitted:
    st.markdown(
        """
        <div style="padding: 4rem 2rem; border: 1px solid #1a1a1a; background: #0a0a0a; margin-top: 2rem; text-align: center;">
            <div style="color: #d4af37; font-size: 1.5rem; margin-bottom: 1rem;">❖</div>
            <h3 style="color:#ffffff; font-weight:300; margin-bottom: 1rem; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">Awaiting Directive</h3>
            <p style="font-size:0.85rem; color:#555555; margin:0; line-height: 1.5;">
                Enter parameters above to initiate the retrieval and synthesis pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

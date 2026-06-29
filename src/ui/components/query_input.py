"""Query input component."""
import streamlit as st

def render_query_input() -> dict:
    prefill = st.session_state.pop("prefill_question", "")

    st.markdown(
        """
        <div style="font-size:0.8rem; color:#d4af37; letter-spacing:1.5px; font-weight:600; margin-bottom: 0.5rem; text-transform: uppercase;">
            Directive
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Directive",
        value=prefill,
        height=100,
        placeholder="Input research parameters...",
        label_visibility="collapsed"
    )

    char_count = len(question.strip())
    
    st.markdown(
        f"""
        <div style="text-align: right; font-size: 0.7rem; color: #555555; margin-top: -12px; margin-bottom: 1rem; letter-spacing: 1px;">
            {char_count} CHARS
        </div>
        """,
        unsafe_allow_html=True,
    )

    submitted = st.button("EXECUTE", type="primary", disabled=char_count < 10)

    return {
        "question": question.strip(),
        "submitted": submitted,
    }

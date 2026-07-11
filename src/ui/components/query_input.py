"""Query input component."""
import streamlit as st

def render_query_input() -> dict:
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = ""
        
    if "prefill_question" in st.session_state:
        st.session_state["query_text"] = st.session_state.pop("prefill_question")

    st.markdown(
        """
        <div style="font-size:0.8rem; color:#d4af37; letter-spacing:1.5px; font-weight:600; margin-bottom: 0.5rem; text-transform: uppercase;">
            Directive
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form(key="research_form", clear_on_submit=False):
        question = st.text_area(
            "Directive",
            key="query_text",
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

        # Form submit buttons don't support `disabled` based on real-time text input in Streamlit,
        # so we handle validation after submission.
        submitted = st.form_submit_button("EXECUTE", type="primary")

    if submitted and len(question.strip()) < 10:
        st.warning("Please enter a directive of at least 10 characters.")
        submitted = False

    return {
        "question": question.strip(),
        "submitted": submitted,
    }

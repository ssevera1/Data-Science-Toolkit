"""Reusable AI interpretation component for all analysis pages.

Provides render_ai_interpretation() (per-page results) and
render_data_plan() (home page after file upload).  Handles both
API mode (inline Gemini calls) and clipboard fallback mode.
"""

import streamlit as st
from core.gemini import is_api_available, query_gemini, open_gemini_in_browser
from utils.prompt_builder import (
    build_brief_prompt,
    build_deep_dive_prompt,
    build_data_plan_prompt,
    build_clipboard_text,
)


def render_ai_interpretation(
    entry_type: str,
    result: dict,
    variables: dict,
    alpha: float = 0.05,
    page_key: str = "",
) -> dict:
    """Render the AI interpretation section after analysis results.

    Call this ONCE per page, after the results/assumptions/charts tabs
    and before the PDF export section.

    Parameters
    ----------
    entry_type : str
        Same as the PDF entry_type (e.g. "independent_ttest").
    result : dict
        The stats result dict from the computation.
    variables : dict
        Variable names used (e.g. {"dependent_variable": "score"}).
    alpha : float
        Significance level.
    page_key : str
        Unique prefix for widget keys (e.g. "ind").

    Returns
    -------
    dict with keys "brief" (str|None) and "deep_dive" (str|None).
    """
    ai_texts: dict = {"brief": None, "deep_dive": None}

    # Deterministic cache key based on result fingerprint
    _fp = f"{page_key}_{result.get('p', '')}_{result.get('statistic', result.get('t', result.get('F', '')))}"
    brief_ck = f"_gemini_{page_key}_brief_{hash(_fp)}"
    deep_ck = f"_gemini_{page_key}_deep_{hash(_fp)}"

    st.markdown("---")
    mode_label = "Gemini API" if is_api_available() else "Clipboard Mode"
    st.markdown(f"**AI Interpretation** &nbsp; *({mode_label})*")

    if is_api_available():
        # ── API mode: auto-brief ────────────────────────────────────
        if brief_ck not in st.session_state:
            prompt = build_brief_prompt(entry_type, result, variables, alpha)
            text = query_gemini(prompt, max_tokens=250)
            if text:
                st.session_state[brief_ck] = text

        brief = st.session_state.get(brief_ck)
        if brief:
            ai_texts["brief"] = brief
            st.info(brief)
        else:
            st.caption("AI interpretation unavailable.")

        # ── API mode: deep dive (on-demand) ─────────────────────────
        if st.button("Deep Dive Analysis", key=f"{page_key}_deep_dive_btn"):
            prompt = build_deep_dive_prompt(entry_type, result, variables, alpha)
            text = query_gemini(prompt, max_tokens=1200)
            if text:
                st.session_state[deep_ck] = text

        deep = st.session_state.get(deep_ck)
        if deep:
            ai_texts["deep_dive"] = deep
            with st.expander("Detailed AI Analysis", expanded=True):
                st.markdown(deep)

    else:
        # ── Clipboard fallback ──────────────────────────────────────
        st.caption(
            "No Gemini API key configured. Copy the prompt below and "
            "paste it into Gemini for an AI interpretation."
        )
        brief_prompt = build_brief_prompt(entry_type, result, variables, alpha)
        st.code(build_clipboard_text(brief_prompt), language=None)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Open Gemini in Browser", key=f"{page_key}_open_gemini"):
                open_gemini_in_browser()
                st.info("Gemini opened in your browser. Paste the prompt above.")
        with col2:
            if st.button("Get Deep Dive Prompt", key=f"{page_key}_deep_prompt_btn"):
                deep_prompt = build_deep_dive_prompt(entry_type, result, variables, alpha)
                st.code(build_clipboard_text(deep_prompt), language=None)

    return ai_texts


def render_data_plan(
    df_summary: dict,
    page_key: str = "home",
) -> str | None:
    """Render the AI-recommended analysis plan after file upload.

    Parameters
    ----------
    df_summary : dict
        {"columns": [{"name", "dtype", "n_unique", "n_missing", ...}],
         "n_rows": int, "n_cols": int, "target_col": str|None}

    Returns
    -------
    str or None — the recommendation text.
    """
    plan_ck = "_gemini_data_plan"

    st.markdown("---")
    st.subheader("AI-Recommended Analysis Plan")

    if is_api_available():
        if plan_ck not in st.session_state:
            prompt = build_data_plan_prompt(
                columns=df_summary["columns"],
                n_rows=df_summary["n_rows"],
                n_cols=df_summary["n_cols"],
                target_col=df_summary.get("target_col"),
            )
            text = query_gemini(prompt, max_tokens=1500)
            if text:
                st.session_state[plan_ck] = text

        plan = st.session_state.get(plan_ck)
        if plan:
            st.markdown(plan)
            return plan
        else:
            st.caption("AI analysis plan unavailable.")
    else:
        st.caption(
            "No Gemini API key configured. Copy the prompt below and "
            "paste it into Gemini for a recommended analysis plan."
        )
        prompt = build_data_plan_prompt(
            columns=df_summary["columns"],
            n_rows=df_summary["n_rows"],
            n_cols=df_summary["n_cols"],
            target_col=df_summary.get("target_col"),
        )
        st.code(build_clipboard_text(prompt), language=None)
        if st.button("Open Gemini in Browser", key=f"{page_key}_plan_gemini"):
            open_gemini_in_browser()

    return None

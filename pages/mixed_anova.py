"""Mixed ANOVA page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_any_variable, select_nominal_variable
from stats.anova import mixed_anova
from charts.barplot import two_way_bar
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df

_CACHE_KEY = "_result_mix_anova"


def render():
    st.title("Mixed ANOVA")
    st.markdown("Test effects of one within-subjects and one between-subjects factor.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        dv = select_metric_variable("Dependent variable", key="mx_dv")
        within = select_any_variable("Within-subjects factor", key="mx_within")
    with col2:
        between = select_nominal_variable("Between-subjects factor", key="mx_between")
        subject = select_any_variable("Subject ID", key="mx_subject")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="mx_alpha")

    if dv and within and between and subject and st.button("Calculate", type="primary"):
        clean = df[[dv, within, between, subject]].dropna().copy()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        try:
            result = mixed_anova(clean, dv, within, between, subject, alpha=alpha)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return

        st.session_state[_CACHE_KEY] = {
            "inputs": (dv, within, between, subject, alpha),
            "result": result,
            "clean": clean,
        }

    # ── Invalidate cache if inputs changed ─────────────────────────────
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached["inputs"] != (dv, within, between, subject, alpha):
        del st.session_state[_CACHE_KEY]
        cached = None

    if cached:
        result = cached["result"]
        clean = cached["clean"]

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            st.markdown("### ANOVA Table")
            aov = result["anova_table"]
            st.dataframe(aov, width="stretch", hide_index=True)

            for _, row in aov.iterrows():
                source = row.get("Source", "")
                p = row.get("p-unc", None)
                if p is not None and source:
                    sig = "✓ Significant" if p < alpha else "✗ Not significant"
                    st.markdown(f"**{source}:** F = {row.get('F', 0):.4f}, p = {p:.4f} — {sig}")

            st.markdown("### Group Descriptives")
            st.dataframe(result["group_desc"], width="stretch", hide_index=True)

        with tab_chart:
            fig = two_way_bar(clean, dv, within, between)
            st.plotly_chart(fig, width="stretch")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = [_serialize_df(result["anova_table"], "ANOVA Table")]
        if "group_desc" in result:
            _tables.append(_serialize_df(result["group_desc"], "Group Descriptives"))
        _log_entry = build_log_entry(
            entry_type="mixed_anova",
            title=f"Mixed ANOVA: {dv} ~ {within} x {between}",
            result=result,
            tables=_tables,
            variables={"dv": dv, "within": within, "between": between, "subject": subject},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="mx_pdf_chart")
        if _include_chart:
            _fig = two_way_bar(clean, dv, within, between)
            _log_entry["figures"] = [{"label": "Two-Way Bar Chart", "fig_dict": _fig.to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="mx_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="mixed_anova.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
Mixed ANOVA tests the effects of **one within-subjects factor** and **one between-subjects factor** simultaneously on a continuous dependent variable. It combines elements of repeated measures ANOVA and independent-groups ANOVA.

#### When to Use
- You have **repeated measurements** on the same subjects (within-subjects factor) **and** a **grouping variable** that divides subjects into separate groups (between-subjects factor).
- Example: treatment group vs. control group (between) measured at three time points (within).

#### Input
- **Dependent variable (DV)** -- metric (continuous) variable.
- **Within-subjects factor** -- identifies the repeated condition (e.g., time point, measurement occasion).
- **Between-subjects factor** -- nominal variable that separates subjects into independent groups (e.g., treatment vs. control).
- **Subject ID** -- uniquely identifies each participant.

#### Results Tab
- **ANOVA table** showing F-statistic and p-value for:
    - **Within-subjects effect** -- do means change across the repeated conditions?
    - **Between-subjects effect** -- do the groups differ overall?
    - **Interaction** -- does the pattern of change across conditions differ between the groups?
- **Group descriptives** -- N, Mean, and SD broken down by both factors.

#### Charts Tab
- **Two-way bar chart** -- shows means broken down by both the within-subjects and between-subjects factors, helping visualize main effects and interactions.
        """)

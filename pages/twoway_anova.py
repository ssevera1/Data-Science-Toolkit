"""Two-Way ANOVA page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import render_assumption_check
from stats.anova import twoway_anova
from charts.barplot import two_way_bar
from charts.boxplot import grouped_boxplot
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df


def render():
    st.title("Two-Way ANOVA")
    st.markdown("Test the effects of two factors and their interaction on a dependent variable.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2, col3 = st.columns(3)
    with col1:
        dv = select_metric_variable("Dependent variable", key="tw_dv")
    with col2:
        factor1 = select_nominal_variable("Factor 1", key="tw_f1")
    with col3:
        factor2 = select_nominal_variable("Factor 2", key="tw_f2")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="tw_alpha")

    if dv and factor1 and factor2 and st.button("Calculate", type="primary"):
        if factor1 == factor2:
            st.error("Please select two different factors.")
            return

        clean = df[[dv, factor1, factor2]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        if len(clean) < 4:
            st.error("Not enough data for Two-Way ANOVA.")
            return

        result = twoway_anova(clean, dv, factor1, factor2, alpha=alpha)

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            st.markdown("### ANOVA Table")
            aov = result["anova_table"]
            st.dataframe(aov, width="stretch", hide_index=True)

            # Significance summary
            for _, row in aov.iterrows():
                source = row.get("Source", "")
                p = row.get("p-unc", None)
                if p is not None and source:
                    sig = "✓ Significant" if p < alpha else "✗ Not significant"
                    st.markdown(f"**{source}:** F = {row.get('F', 0):.4f}, p = {p:.4f} — {sig}")

            st.markdown("### Group Descriptives")
            st.dataframe(result["group_desc"], width="stretch", hide_index=True)

        with tab_assume:
            homo = result["assumptions"]["homogeneity"]
            render_assumption_check(
                "Levene's Test (Homogeneity of Variances)", homo["statistic"],
                homo["p_value"], homo["passed"], homo["detail"]
            )

        with tab_chart:
            fig = two_way_bar(clean, dv, factor1, factor2)
            st.plotly_chart(fig, width="stretch")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = [_serialize_df(result["anova_table"], "ANOVA Table")]
        if "group_desc" in result:
            _tables.append(_serialize_df(result["group_desc"], "Group Descriptives"))
        _log_entry = build_log_entry(
            entry_type="twoway_anova",
            title=f"Two-Way ANOVA: {dv} ~ {factor1} x {factor2}",
            result=result,
            tables=_tables,
            variables={"dv": dv, "factor_1": factor1, "factor_2": factor2},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="tw_pdf_chart")
        if _include_chart:
            _fig = two_way_bar(clean, dv, factor1, factor2)
            _log_entry["figures"] = [{"label": "Two-Way Bar Chart", "fig_dict": _fig.to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="tw_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="twoway_anova.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
Two-way ANOVA tests the effects of **two categorical factors** and their **interaction** on a continuous dependent variable.

#### When to Use
- One **continuous dependent variable** (DV).
- **Two categorical independent variables** (factors) -- e.g., treatment type and gender.
- You want to assess **main effects** of each factor and whether the two factors **interact**.

#### Results Tab
- **ANOVA table** -- shows the F-statistic and p-value for:
    - **Factor 1 main effect** -- does the DV differ across levels of Factor 1, averaging over Factor 2?
    - **Factor 2 main effect** -- does the DV differ across levels of Factor 2, averaging over Factor 1?
    - **Interaction effect** -- does the effect of one factor depend on the level of the other factor?
- **Group descriptives** -- N, Mean, and SD broken down by both factors.

#### Interpreting the Interaction
- A **significant interaction** means the effect of one factor **changes depending on** the level of the other factor. In this case, interpret main effects with caution -- they may be misleading.
- A **non-significant interaction** means the effects of the two factors are **additive** (independent of each other).

#### Assumptions Tab
- **Levene's test** -- tests **homogeneity of variances** across all factor-level combinations. If significant, the equal-variance assumption is violated.

#### Charts Tab
- **Grouped bar chart** -- shows mean values of the DV broken down by both factors, making it easy to visualize main effects and interactions.
        """)


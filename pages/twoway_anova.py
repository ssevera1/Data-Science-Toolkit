"""Two-Way ANOVA page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import render_assumption_check
from stats.anova import twoway_anova
from charts.barplot import two_way_bar
from charts.boxplot import grouped_boxplot
from core.state import get_df


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
            st.dataframe(aov, use_container_width=True, hide_index=True)

            # Significance summary
            for _, row in aov.iterrows():
                source = row.get("Source", "")
                p = row.get("p-unc", None)
                if p is not None and source:
                    sig = "✓ Significant" if p < alpha else "✗ Not significant"
                    st.markdown(f"**{source}:** F = {row.get('F', 0):.4f}, p = {p:.4f} — {sig}")

            st.markdown("### Group Descriptives")
            st.dataframe(result["group_desc"], use_container_width=True, hide_index=True)

        with tab_assume:
            homo = result["assumptions"]["homogeneity"]
            render_assumption_check(
                "Levene's Test (Homogeneity of Variances)", homo["statistic"],
                homo["p_value"], homo["passed"], homo["detail"]
            )

        with tab_chart:
            fig = two_way_bar(clean, dv, factor1, factor2)
            st.plotly_chart(fig, use_container_width=True)

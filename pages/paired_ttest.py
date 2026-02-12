"""Paired Samples t-Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable
from components.results_display import (
    render_significance_result, render_assumption_check,
    render_effect_size, interpret_cohens_d,
)
from stats.ttest import paired_ttest
from core.validators import validate_two_metrics
from charts.boxplot import paired_boxplot
from charts.qq_plot import qq_plot
from core.state import get_df


def render():
    st.title("Paired Samples t-Test")
    st.markdown("Compare means of two related measurements.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        var1 = select_metric_variable("Variable 1 (Time 1 / Condition A)", key="paired_v1")
    with col2:
        var2 = select_metric_variable("Variable 2 (Time 2 / Condition B)", key="paired_v2")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="paired_alpha")

    if var1 and var2 and st.button("Calculate", type="primary"):
        if var1 == var2:
            st.error("Please select two different variables.")
            return

        valid, msg = validate_two_metrics(var1, var2)
        if not valid:
            st.error(msg)
            return

        result = paired_ttest(df[var1], df[var2], alpha=alpha)

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "t", result["t"], result["p"], result["df"], alpha
            )
            st.markdown("---")

            cols = st.columns(4)
            cols[0].metric("N (pairs)", result["n"])
            cols[1].metric(f"Mean ({var1})", f"{result['mean1']:.4f}")
            cols[2].metric(f"Mean ({var2})", f"{result['mean2']:.4f}")
            cols[3].metric("Mean Diff", f"{result['mean_diff']:.4f}")

            st.markdown(f"**SD of Differences:** {result['sd_diff']:.4f}")
            st.markdown(f"**95% CI for Mean Difference:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

            render_effect_size("Cohen's d", result["cohens_d"], interpret_cohens_d(result["cohens_d"]))

        with tab_assume:
            norm = result["assumptions"]["normality_of_differences"]
            render_assumption_check(
                "Shapiro-Wilk (Normality of Differences)", norm["statistic"],
                norm["p_value"], norm["passed"], norm["detail"]
            )

        with tab_chart:
            c1, c2 = st.columns(2)
            with c1:
                clean = df[[var1, var2]].dropna()
                clean[var1] = pd.to_numeric(clean[var1], errors="coerce")
                clean[var2] = pd.to_numeric(clean[var2], errors="coerce")
                clean = clean.dropna()
                fig = paired_boxplot(clean, var1, var2)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                diff = clean[var1] - clean[var2]
                fig = qq_plot(diff, "Differences")
                st.plotly_chart(fig, use_container_width=True)

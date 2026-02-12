"""Wilcoxon Signed-Rank Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable
from components.results_display import render_significance_result, render_effect_size
from stats.nonparametric import wilcoxon_signed_rank
from core.validators import validate_two_metrics
from charts.boxplot import paired_boxplot
from charts.histogram import histogram_with_normal
from core.state import get_df


def render():
    st.title("Wilcoxon Signed-Rank Test")
    st.markdown("Non-parametric alternative to the paired samples t-test.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        var1 = select_metric_variable("Variable 1", key="wil_v1")
    with col2:
        var2 = select_metric_variable("Variable 2", key="wil_v2")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="wil_alpha")
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"], key="wil_alt")

    if var1 and var2 and st.button("Calculate", type="primary"):
        if var1 == var2:
            st.error("Please select two different variables.")
            return

        valid, msg = validate_two_metrics(var1, var2)
        if not valid:
            st.error(msg)
            return

        result = wilcoxon_signed_rank(df[var1], df[var2], alternative=alternative, alpha=alpha)

        if result.get("W") is None:
            st.error(result.get("detail", "Could not compute test."))
            return

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "W", result["W"], result["p"], alpha=alpha
            )
            st.markdown("---")

            cols = st.columns(4)
            cols[0].metric("N (non-zero diff)", result["n"])
            cols[1].metric(f"Median ({var1})", f"{result['median1']:.4f}")
            cols[2].metric(f"Median ({var2})", f"{result['median2']:.4f}")
            cols[3].metric("Median Diff", f"{result['median_diff']:.4f}")

            render_effect_size("Effect size (r)", result["r_effect"])

        with tab_chart:
            clean = df[[var1, var2]].dropna()
            clean[var1] = pd.to_numeric(clean[var1], errors="coerce")
            clean[var2] = pd.to_numeric(clean[var2], errors="coerce")
            clean = clean.dropna()

            c1, c2 = st.columns(2)
            with c1:
                fig = paired_boxplot(clean, var1, var2)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                diff = clean[var1] - clean[var2]
                fig = histogram_with_normal(diff, "Differences")
                st.plotly_chart(fig, use_container_width=True)

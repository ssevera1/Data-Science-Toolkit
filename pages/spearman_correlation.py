"""Spearman Correlation page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable
from components.results_display import render_significance_result, render_effect_size, interpret_r
from stats.correlation import spearman_correlation
from core.validators import validate_two_metrics
from charts.scatter import correlation_scatter
from core.state import get_df


def render():
    st.title("Spearman Correlation")
    st.markdown("Measure the monotonic relationship between two variables using ranks.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        var1 = select_metric_variable("Variable X", key="spear_x")
    with col2:
        var2 = select_metric_variable("Variable Y", key="spear_y")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="spear_alpha")

    if var1 and var2 and st.button("Calculate", type="primary"):
        if var1 == var2:
            st.error("Please select two different variables.")
            return

        valid, msg = validate_two_metrics(var1, var2)
        if not valid:
            st.error(msg)
            return

        result = spearman_correlation(df[var1], df[var2], alpha=alpha)

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "ρ", result["rho"], result["p"], alpha=alpha
            )
            st.markdown("---")

            cols = st.columns(3)
            cols[0].metric("N", result["n"])
            cols[1].metric("ρ (rho)", f"{result['rho']:.4f}")
            cols[2].metric("p", f"{result['p']:.4f}")

            st.markdown(f"**95% CI for ρ:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

            render_effect_size("Spearman ρ", result["rho"], interpret_r(result["rho"]))

        with tab_chart:
            clean = df[[var1, var2]].dropna()
            clean[var1] = pd.to_numeric(clean[var1], errors="coerce")
            clean[var2] = pd.to_numeric(clean[var2], errors="coerce")
            clean = clean.dropna()

            fig = correlation_scatter(clean, var1, var2, r_value=result["rho"])
            st.plotly_chart(fig, use_container_width=True)

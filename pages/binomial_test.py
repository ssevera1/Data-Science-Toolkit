"""Binomial Test page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components.data_table import render_data_preview
from components.variable_selector import select_any_variable
from components.results_display import render_significance_result
from stats.binomial import binomial_test
from core.state import get_df
from charts.theme import apply_theme, COLORS


def render():
    st.title("Binomial Test")
    st.markdown("Test whether the proportion of a binary variable matches a hypothesized value.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        var = select_any_variable("Binary variable (2 categories)", key="binom_var")
    with col2:
        test_prop = st.number_input(
            "Hypothesized proportion (p₀)",
            min_value=0.01, max_value=0.99, value=0.50, step=0.05,
            key="binom_prop",
        )

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="binom_alpha")

    if var and st.button("Calculate", type="primary"):
        series = df[var].dropna()
        if len(series.unique()) != 2:
            st.error(f"Variable must have exactly 2 categories, found {len(series.unique())}.")
            return

        result = binomial_test(series, test_prop=test_prop, alpha=alpha)

        if "error" in result:
            st.error(result["error"])
            return

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "Observed p", result["observed_prop"],
                result["p"], alpha=alpha
            )
            st.markdown("---")

            cols = st.columns(4)
            cols[0].metric("N", result["n"])
            cols[1].metric(f"'{result['success_label']}'", result["n_success"])
            cols[2].metric(f"'{result['failure_label']}'", result["n_failure"])
            cols[3].metric("Observed Proportion", f"{result['observed_prop']:.4f}")

            st.markdown(f"**Test proportion (p₀):** {result['test_prop']:.4f}")
            st.markdown(f"**95% CI for proportion:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

        with tab_chart:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[str(result["success_label"]), str(result["failure_label"])],
                y=[result["n_success"], result["n_failure"]],
                marker_color=[COLORS[0], COLORS[1]],
                text=[result["n_success"], result["n_failure"]],
                textposition="auto",
            ))
            fig.update_layout(
                title=f"Frequencies of {var}",
                xaxis_title=var,
                yaxis_title="Count",
            )
            st.plotly_chart(apply_theme(fig), use_container_width=True)

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
The binomial test determines whether the **observed proportion** of a binary variable significantly differs from a **hypothesized proportion**.

#### When to Use
- You have one variable with **exactly 2 categories** (e.g., success/failure, heads/tails, yes/no).
- You want to test whether the observed proportion matches a specific expected value (e.g., 50/50, 70/30).

#### Input
- **Binary variable** -- a column with exactly 2 unique categories.
- **Hypothesized proportion (p0)** -- the expected proportion of the first category under the null hypothesis. Default is **0.50** (equal split).

#### Results Tab
- **Observed proportion** -- the actual proportion of the first category in the data.
- **p-value** -- probability of observing a proportion this extreme (or more) if the true proportion equals p0.
- **N** -- total number of observations.
- **Counts** for each category -- how many observations fall into each of the two categories.
- **95% Confidence Interval for the proportion** -- range likely to contain the true population proportion.

#### Charts Tab
- **Bar chart** -- shows the frequency (count) of each category, providing a visual summary of the distribution.
        """)


"""One-Sample t-Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable
from components.results_display import (
    render_significance_result, render_assumption_check,
    render_effect_size, interpret_cohens_d,
)
from stats.ttest import one_sample_ttest
from charts.boxplot import single_boxplot
from charts.histogram import histogram_with_normal
from charts.qq_plot import qq_plot
from core.state import get_df


def render():
    st.title("One-Sample t-Test")
    st.markdown("Test whether a sample mean differs from a known or hypothesized value.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        var = select_metric_variable("Test variable", key="os_var")
    with col2:
        test_value = st.number_input("Test value (μ₀)", value=0.0, key="os_mu")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="os_alpha")

    if var and st.button("Calculate", type="primary"):
        series = pd.to_numeric(df[var], errors="coerce").dropna()
        if len(series) < 2:
            st.error("Need at least 2 observations.")
            return

        result = one_sample_ttest(series, mu=test_value, alpha=alpha)

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "t", result["t"], result["p"], result["df"], alpha
            )
            st.markdown("---")
            cols = st.columns(4)
            cols[0].metric("N", result["n"])
            cols[1].metric("Mean", f"{result['mean']:.4f}")
            cols[2].metric("Mean Difference", f"{result['mean_diff']:.4f}")
            cols[3].metric("Std. Error", f"{result['se']:.4f}")

            st.markdown(f"**95% CI for Mean Difference:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

            render_effect_size("Cohen's d", result["cohens_d"], interpret_cohens_d(result["cohens_d"]))

        with tab_assume:
            norm = result["assumptions"]["normality"]
            render_assumption_check(
                "Shapiro-Wilk (Normality)", norm["statistic"], norm["p_value"],
                norm["passed"], norm["detail"]
            )

        with tab_chart:
            c1, c2 = st.columns(2)
            with c1:
                fig = single_boxplot(series, var, title=f"{var} (μ₀ = {test_value})")
                fig.add_hline(y=test_value, line_dash="dash", line_color="red",
                             annotation_text=f"μ₀ = {test_value}")
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = qq_plot(series, var)
                st.plotly_chart(fig, width="stretch")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
The one-sample t-test determines whether a **sample mean** significantly differs from a known or hypothesized **population mean** (denoted as the test value).

#### When to Use
- You have **one group** of continuous (metric) data.
- You want to compare that group's mean to a **specific reference value** (e.g., a population average, a target, or a benchmark).

#### Input
- **Test variable** -- must be metric (continuous numeric data).
- **Test value (or test mean)** -- the hypothesized population mean to compare against.

#### Results Tab
- **t-statistic** -- measures how many standard errors the sample mean is from the test value.
- **p-value** -- probability of observing a result this extreme if the null hypothesis (mean = test value) is true.
- **Degrees of freedom** -- N - 1, where N is the number of observations.
- **Mean difference** -- sample mean minus the test value.
- **Standard error (SE)** -- SD / sqrt(N); precision of the sample mean estimate.
- **95% Confidence Interval** -- range likely to contain the true mean difference.
- **Cohen's d** -- standardized effect size: **small ~ 0.2**, **medium ~ 0.5**, **large ~ 0.8**.

#### Assumptions Tab
- **Shapiro-Wilk normality test** -- tests whether the data follow a normal distribution. A p-value > 0.05 suggests normality is plausible. The t-test is robust to normality violations when **N > 30** (Central Limit Theorem).

#### Charts Tab
- **Box plot** with a dashed reference line at the test value -- visually compare the sample distribution to the hypothesized mean.
- **Q-Q plot** -- points should follow the diagonal line if the data are approximately normally distributed. Deviations indicate departures from normality.
        """)


"""Pearson Correlation page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable
from components.results_display import (
    render_significance_result, render_assumption_check,
    render_effect_size, interpret_r,
)
from stats.correlation import pearson_correlation
from core.validators import validate_two_metrics
from charts.scatter import correlation_scatter
from charts.qq_plot import qq_plot
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report

_CACHE_KEY = "_result_pearson"


def render():
    st.title("Pearson Correlation")
    st.markdown("Measure the linear relationship between two continuous variables.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        var1 = select_metric_variable("Variable X", key="pear_x")
    with col2:
        var2 = select_metric_variable("Variable Y", key="pear_y")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="pear_alpha")

    if var1 and var2 and st.button("Calculate", type="primary"):
        if var1 == var2:
            st.error("Please select two different variables.")
            return

        valid, msg = validate_two_metrics(var1, var2)
        if not valid:
            st.error(msg)
            return

        result = pearson_correlation(df[var1], df[var2], alpha=alpha)

        clean = df[[var1, var2]].dropna().copy()
        clean[var1] = pd.to_numeric(clean[var1], errors="coerce")
        clean[var2] = pd.to_numeric(clean[var2], errors="coerce")
        clean = clean.dropna()

        st.session_state[_CACHE_KEY] = {
            "inputs": (var1, var2, alpha),
            "result": result,
            "clean": clean,
        }

    # ── Invalidate cache if inputs changed ─────────────────────────────
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached["inputs"] != (var1, var2, alpha):
        del st.session_state[_CACHE_KEY]
        cached = None

    if cached:
        result = cached["result"]
        clean = cached["clean"]

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "r", result["r"], result["p"], result["df"], alpha
            )
            st.markdown("---")

            cols = st.columns(4)
            cols[0].metric("N", result["n"])
            cols[1].metric("r", f"{result['r']:.4f}")
            cols[2].metric("R²", f"{result['r_squared']:.4f}")
            cols[3].metric("p", f"{result['p']:.4f}")

            st.markdown(f"**95% CI for r:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

            render_effect_size("Pearson r", result["r"], interpret_r(result["r"]))

        with tab_assume:
            norm_x = result["assumptions"]["normality_x"]
            render_assumption_check(
                f"Shapiro-Wilk: {var1}", norm_x["statistic"], norm_x["p_value"],
                norm_x["passed"], norm_x["detail"]
            )
            norm_y = result["assumptions"]["normality_y"]
            render_assumption_check(
                f"Shapiro-Wilk: {var2}", norm_y["statistic"], norm_y["p_value"],
                norm_y["passed"], norm_y["detail"]
            )

        with tab_chart:
            fig = correlation_scatter(clean, var1, var2, r_value=result["r"])
            st.plotly_chart(fig, width="stretch")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _log_entry = build_log_entry(
            entry_type="pearson_correlation",
            title=f"Pearson Correlation: {var1} vs {var2}",
            result=result,
            variables={"variable_x": var1, "variable_y": var2},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="pear_pdf_chart")
        if _include_chart:
            _fig = correlation_scatter(clean, var1, var2, r_value=result["r"])
            _log_entry["figures"] = [{"label": "Correlation Scatter", "fig_dict": _fig.to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="pear_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="pearson_correlation.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
Pearson correlation measures the **strength and direction of the linear relationship** between two continuous variables.

#### When to Use
- Both variables are **continuous** (metric).
- The relationship between the variables is expected to be **linear** (a straight-line trend).
- If the relationship is monotonic but not linear, consider Spearman correlation instead.

#### Results Tab
- **r (Pearson correlation coefficient)** -- ranges from **-1** (perfect negative linear relationship) through **0** (no linear relationship) to **+1** (perfect positive linear relationship).
- **R-squared** -- the proportion of variance in one variable explained by the other (r squared).
- **p-value** -- tests the null hypothesis that the true correlation is zero.
- **Degrees of freedom** -- N - 2.
- **95% Confidence Interval for r** -- range likely to contain the true population correlation.

#### Effect Size Interpretation
- |r| < 0.1 -- **negligible**
- |r| 0.1 to 0.3 -- **small**
- |r| 0.3 to 0.5 -- **medium**
- |r| > 0.5 -- **large**

#### Assumptions Tab
- **Shapiro-Wilk normality test** for each variable -- Pearson correlation assumes **bivariate normality**. Individual normality tests provide a practical check. Robust to violations with larger samples.
- **Linearity** -- the relationship should be linear, which can be assessed visually with the scatter plot.

#### Charts Tab
- **Scatter plot with regression line** -- visualizes the linear relationship between the two variables and the direction/strength of the correlation.
        """)

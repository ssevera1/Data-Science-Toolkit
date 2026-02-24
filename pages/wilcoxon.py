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
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report


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
                st.plotly_chart(fig, width="stretch")
            with c2:
                diff = clean[var1] - clean[var2]
                fig = histogram_with_normal(diff, "Differences")
                st.plotly_chart(fig, width="stretch")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _log_entry = build_log_entry(
            entry_type="wilcoxon",
            title=f"Wilcoxon Signed-Rank: {var1} vs {var2}",
            result=result,
            variables={"variable_1": var1, "variable_2": var2},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include charts in PDF", value=True, key="wil_pdf_chart")
        if _include_chart:
            _figures = []
            _bfig = paired_boxplot(clean, var1, var2)
            _figures.append({"label": "Paired Boxplot", "fig_dict": _bfig.to_dict()})
            _diff = clean[var1] - clean[var2]
            _hfig = histogram_with_normal(_diff, "Differences")
            _figures.append({"label": "Histogram of Differences", "fig_dict": _hfig.to_dict()})
            _log_entry["figures"] = _figures
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="wil_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="wilcoxon.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
The Wilcoxon signed-rank test is a **non-parametric alternative to the paired samples t-test**. It compares two related measurements without assuming the differences are normally distributed.

#### When to Use
- **Paired data** (same subjects measured twice) that **violates the normality assumption**.
- Ordinal or heavily skewed difference scores.
- Small sample sizes where normality cannot be reliably assessed.

#### How It Works
The test computes the **differences** between paired observations, ranks the **absolute values** of those differences, and then compares the sum of ranks for positive vs. negative differences.

#### Input
- **Variable 1 and Variable 2** -- two metric variables representing paired observations.
- **Alternative hypothesis** -- two-sided, greater, or less.

#### Results Tab
- **W statistic** -- the smaller of the positive and negative rank sums (or the test statistic variant used).
- **p-value** -- significance of the difference between the two conditions.
- **N (non-zero differences)** -- number of pairs where the difference is not zero (ties at zero are excluded).
- **Medians** for both variables and the **median difference**.
- **Effect size r** -- computed as r = Z / sqrt(N). Interpretation: **small ~ 0.1**, **medium ~ 0.3**, **large ~ 0.5**.

#### Charts Tab
- **Paired box plot** -- shows both variables' distributions side by side.
- **Histogram of differences** -- visualizes the distribution of difference scores with a normal curve overlay.
        """)


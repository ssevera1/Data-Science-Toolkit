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
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report

_CACHE_KEY = "_result_paired_ttest"


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
                fig = paired_boxplot(clean, var1, var2)
                st.plotly_chart(fig, width="stretch")
            with c2:
                diff = clean[var1] - clean[var2]
                fig_qq = qq_plot(diff, "Differences")
                st.plotly_chart(fig_qq, width="stretch")

        # ── AI Interpretation ──────────────────────────────────────────
        from components.ai_advisor import render_ai_interpretation
        ai_texts = render_ai_interpretation(
            entry_type="paired_ttest",
            result=result,
            variables={"variable_1": var1, "variable_2": var2},
            alpha=alpha,
            page_key="paired",
        )

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _log_entry = build_log_entry(
            entry_type="paired_ttest",
            title=f"Paired t-Test: {var1} vs {var2}",
            result=result,
            variables={"variable_1": var1, "variable_2": var2},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        if ai_texts.get("brief"):
            _log_entry["ai_interpretation"] = ai_texts["brief"]
        if ai_texts.get("deep_dive"):
            _log_entry["ai_deep_dive"] = ai_texts["deep_dive"]
        _include_chart = st.checkbox("Include charts in PDF", value=True, key="paired_pdf_chart")
        if _include_chart:
            _fig_box = paired_boxplot(clean, var1, var2)
            _diff = clean[var1] - clean[var2]
            _fig_qq = qq_plot(_diff, "Differences")
            _log_entry["figures"] = [
                {"label": "Paired Box Plot", "fig_dict": _fig_box.to_dict()},
                {"label": "Q-Q Plot (Differences)", "fig_dict": _fig_qq.to_dict()},
            ]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="paired_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="paired_ttest.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
The paired samples t-test compares the **means of two related measurements** taken on the **same subjects** to determine whether they differ significantly.

#### When to Use
- **Pre/post designs** -- the same participants measured before and after an intervention.
- **Matched pairs** -- each participant in one condition is matched with a participant in the other.
- **Repeated measurements** -- the same individuals measured under two different conditions.

#### Input
- **Variable 1 and Variable 2** -- two metric (continuous) variables representing paired observations (e.g., Time 1 and Time 2, Condition A and Condition B).

#### Results Tab
- **t-statistic** and **p-value** -- test whether the mean of the differences is significantly different from zero.
- **Degrees of freedom** -- N - 1, where N is the number of pairs.
- **N (pairs)** -- number of complete pairs used in the analysis.
- **Means** for both variables and the **mean difference** between them.
- **SD of differences** -- standard deviation of the difference scores.
- **95% Confidence Interval** for the mean difference.
- **Cohen's d** -- standardized effect size: **small ~ 0.2**, **medium ~ 0.5**, **large ~ 0.8**.

#### Assumptions Tab
- **Shapiro-Wilk on the differences** -- tests whether the **difference scores** (Variable 1 minus Variable 2) are normally distributed. Robust to violations with N > 30 pairs.

#### Charts Tab
- **Paired box plot** -- shows the distribution of both variables side by side.
- **Q-Q plot of differences** -- assesses normality of the difference scores; points should follow the diagonal if differences are normally distributed.
        """)

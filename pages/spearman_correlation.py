"""Spearman Correlation page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable
from components.results_display import render_significance_result, render_effect_size, interpret_r
from stats.correlation import spearman_correlation
from core.validators import validate_two_metrics
from charts.scatter import correlation_scatter
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report

_CACHE_KEY = "_result_spearman"


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
            fig = correlation_scatter(clean, var1, var2, r_value=result["rho"])
            st.plotly_chart(fig, width="stretch")

        # ── AI Interpretation ──────────────────────────────────────────
        from components.ai_advisor import render_ai_interpretation
        ai_texts = render_ai_interpretation(
            entry_type="spearman_correlation",
            result=result,
            variables={"variable_x": var1, "variable_y": var2},
            alpha=alpha,
            page_key="spear",
        )

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _log_entry = build_log_entry(
            entry_type="spearman_correlation",
            title=f"Spearman Correlation: {var1} vs {var2}",
            result={**result, "r": result.get("rho")},
            variables={"variable_x": var1, "variable_y": var2},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        if ai_texts.get("brief"):
            _log_entry["ai_interpretation"] = ai_texts["brief"]
        if ai_texts.get("deep_dive"):
            _log_entry["ai_deep_dive"] = ai_texts["deep_dive"]
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="spear_pdf_chart")
        if _include_chart:
            _fig = correlation_scatter(clean, var1, var2, r_value=result["rho"])
            _log_entry["figures"] = [{"label": "Correlation Scatter", "fig_dict": _fig.to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="spear_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="spearman_correlation.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
Spearman correlation measures the **strength and direction of the monotonic relationship** between two variables using their **ranks** rather than raw values.

#### When to Use
- One or both variables are **ordinal**.
- The data have **non-normal distributions** or contain **outliers**.
- The relationship is **monotonic** (consistently increasing or decreasing) but **not necessarily linear**.

#### Difference from Pearson
- **Pearson** uses raw values and measures **linear** association. It is sensitive to outliers.
- **Spearman** converts values to **ranks** first, then computes the correlation on ranks. This makes it robust to outliers and non-linear (but monotonic) relationships.

#### Results Tab
- **Rho (Spearman correlation coefficient)** -- ranges from **-1** (perfect negative monotonic relationship) through **0** (no monotonic relationship) to **+1** (perfect positive monotonic relationship).
- **p-value** -- tests the null hypothesis that the true correlation is zero.
- **N** -- number of valid observations.
- **95% Confidence Interval for rho**.

#### Effect Size Interpretation
Same as Pearson r:
- |rho| < 0.1 -- **negligible**
- |rho| 0.1 to 0.3 -- **small**
- |rho| 0.3 to 0.5 -- **medium**
- |rho| > 0.5 -- **large**

#### Charts Tab
- **Scatter plot with trend line** -- visualizes the relationship between the two variables. Because Spearman uses ranks, the trend may appear curved even with a strong correlation.
        """)

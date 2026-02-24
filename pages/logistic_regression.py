"""Logistic Regression page."""

import streamlit as st
import pandas as pd
import numpy as np
from components.data_table import render_data_preview
from components.variable_selector import select_any_variable, select_multiple_variables
from components.results_display import render_significance_result, render_effect_size
from stats.regression import logistic_regression
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df
import plotly.graph_objects as go
from charts.theme import apply_theme, get_chart_colors


def render():
    st.title("Logistic Regression")
    st.markdown("Predict a binary outcome from one or more predictors.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    dv = select_any_variable("Dependent variable (binary)", key="log_dv")
    predictors = select_multiple_variables("Predictor variables", key="log_pred", var_type="Metric")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="log_alpha")

    if dv and predictors and st.button("Calculate", type="primary"):
        if dv in predictors:
            st.error("The dependent variable cannot also be a predictor.")
            return

        result = logistic_regression(df, dv, predictors, alpha=alpha)

        if "error" in result:
            st.error(result["error"])
            return

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                "Model Fit (Likelihood Ratio)", "χ²", result["chi2"],
                result["chi2_p"], alpha=alpha
            )
            st.markdown("---")

            cols = st.columns(4)
            cols[0].metric("Pseudo R² (McFadden)", f"{result['pseudo_r_squared']:.4f}")
            cols[1].metric("Accuracy", f"{result['accuracy']:.1%}")
            cols[2].metric("N", result["n"])
            cols[3].metric("AIC", f"{result['aic']:.1f}")

            st.markdown("### Coefficients")
            st.dataframe(result["coef_table"], width="stretch", hide_index=True)

            st.markdown("*OR = Odds Ratio; CI = Confidence Interval for Odds Ratio*")

        with tab_chart:
            if len(predictors) == 1:
                clean = df[[dv, predictors[0]]].dropna()
                for c in clean.columns:
                    clean[c] = pd.to_numeric(clean[c], errors="coerce")
                clean = clean.dropna()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=clean[predictors[0]], y=clean[dv],
                    mode="markers", name="Data",
                    marker=dict(color=get_chart_colors()[0], size=7, opacity=0.6),
                ))

                # Predicted probabilities
                probs = result["predicted_probs"]
                sort_idx = clean[predictors[0]].argsort()
                fig.add_trace(go.Scatter(
                    x=clean[predictors[0]].iloc[sort_idx],
                    y=probs[sort_idx],
                    mode="lines", name="Predicted probability",
                    line=dict(color=get_chart_colors()[1], width=2),
                ))

                fig.update_layout(
                    title=f"Logistic Regression: {dv} ~ {predictors[0]}",
                    xaxis_title=predictors[0],
                    yaxis_title=f"P({dv} = 1)",
                )
                st.plotly_chart(apply_theme(fig), width="stretch")
            else:
                # Confusion-style summary
                st.markdown("### Classification Summary")
                predicted_class = (result["predicted_probs"] >= 0.5).astype(int)
                cols_used = [dv] + predictors
                clean = df[cols_used].dropna()
                for c in cols_used:
                    clean[c] = pd.to_numeric(clean[c], errors="coerce")
                clean = clean.dropna()
                y = clean[dv].values

                correct = (predicted_class == y).sum()
                total = len(y)
                st.markdown(f"**Correctly classified:** {correct}/{total} ({correct/total:.1%})")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = [_serialize_df(result["coef_table"], "Coefficients")]
        _log_entry = build_log_entry(
            entry_type="logistic_regression",
            title=f"Logistic Regression: {dv} ~ {' + '.join(predictors[:4])}{'...' if len(predictors) > 4 else ''}",
            result=result,
            tables=_tables,
            variables={"dv": dv, "predictors": ", ".join(predictors)},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="log_pdf_chart")
        if _include_chart and len(predictors) == 1:
            _clean = df[[dv, predictors[0]]].dropna()
            for _c in _clean.columns:
                _clean[_c] = pd.to_numeric(_clean[_c], errors="coerce")
            _clean = _clean.dropna()
            _fig = go.Figure()
            _fig.add_trace(go.Scatter(
                x=_clean[predictors[0]], y=_clean[dv],
                mode="markers", name="Data",
                marker=dict(color=get_chart_colors()[0], size=7, opacity=0.6),
            ))
            _probs = result["predicted_probs"]
            _sort_idx = _clean[predictors[0]].argsort()
            _fig.add_trace(go.Scatter(
                x=_clean[predictors[0]].iloc[_sort_idx],
                y=_probs[_sort_idx],
                mode="lines", name="Predicted probability",
                line=dict(color=get_chart_colors()[1], width=2),
            ))
            _fig.update_layout(
                title=f"Logistic Regression: {dv} ~ {predictors[0]}",
                xaxis_title=predictors[0],
                yaxis_title=f"P({dv} = 1)",
            )
            _log_entry["figures"] = [{"label": "Logistic Curve", "fig_dict": apply_theme(_fig).to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="log_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="logistic_regression.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
Logistic regression predicts a **binary outcome** (0 or 1) from one or more predictor variables. Instead of predicting a continuous value, it estimates the **probability** that the outcome equals 1.

#### When to Use
- The **dependent variable is binary** (e.g., pass/fail, yes/no, survived/not survived).
- Predictor variables are **metric** (continuous).

#### Results Tab
- **Likelihood Ratio Chi-squared test** -- tests overall model fit by comparing the model with predictors to a null (intercept-only) model.
- **Pseudo R-squared (McFadden)** -- analogous to R-squared in linear regression but based on log-likelihoods. Values of 0.2-0.4 are considered a good fit.
- **Accuracy** -- percentage of observations correctly classified using a 0.5 probability cutoff.
- **N** -- number of observations used.
- **AIC** -- model fit measure; lower values indicate better fit.
- **Coefficient table** with **Odds Ratios (OR)**:
    - **B (coefficient)** -- the log-odds change for a one-unit increase in the predictor.
    - **OR (Odds Ratio)** -- exponentiated coefficient. **OR > 1** means increased odds of the outcome being 1 per unit increase in the predictor. **OR < 1** means decreased odds. **OR = 1** means no effect.
    - **CI for OR** -- confidence interval for the odds ratio.

#### Charts Tab
- **Logistic curve** (simple regression with one predictor) -- shows the S-shaped probability curve and the observed data points.
- **Classification summary** (multiple regression) -- shows the number of correctly and incorrectly classified observations.
        """)


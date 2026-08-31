"""Multivariate Linear Regression page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_multiple_variables
from components.results_display import (
    render_significance_result, render_assumption_check,
)
from stats.multivariate_regression import multivariate_regression
from charts.regression_plot import multi_regression_actual_vs_predicted
from charts.qq_plot import qq_plot
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df

_CACHE_KEY = "_result_mv_reg"


def render():
    st.title("Multivariate Regression")
    st.markdown(
        "Predict **multiple continuous outcomes** from one or more predictor variables."
    )

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    # ── Variable selection ────────────────────────────────────────────────
    dv_cols = select_multiple_variables(
        "Dependent variables (2 or more metric)", key="mvr_dv", var_type="Metric"
    )
    predictors = select_multiple_variables(
        "Predictor variables", key="mvr_pred", var_type="Metric"
    )

    with st.expander("Options"):
        alpha = st.slider(
            "Significance level (\u03b1)", 0.01, 0.10, 0.05, 0.01, key="mvr_alpha"
        )

    # ── Validation & computation ──────────────────────────────────────────
    if (
        dv_cols
        and len(dv_cols) >= 2
        and predictors
        and st.button("Calculate", type="primary")
    ):
        # Check for overlap
        overlap = set(dv_cols) & set(predictors)
        if overlap:
            st.error(
                f"Variables cannot be both DV and predictor: {', '.join(overlap)}"
            )
            return

        result = multivariate_regression(df, dv_cols, predictors, alpha=alpha)

        if result["n"] < len(predictors) + 2:
            st.error(
                f"Not enough observations ({result['n']}) for "
                f"{len(predictors)} predictor(s). Need at least "
                f"{len(predictors) + 2}."
            )
            return

        st.session_state[_CACHE_KEY] = {
            "inputs": (tuple(dv_cols), tuple(predictors), alpha),
            "result": result,
        }

    # ── Cache invalidation ─────────────────────────────────────────────
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached["inputs"] != (tuple(dv_cols or []), tuple(predictors or []), alpha):
        del st.session_state[_CACHE_KEY]
        cached = None

    if cached:
        result = cached["result"]

        # ── Tabs ──────────────────────────────────────────────────────────
        tab_res, tab_models, tab_assume, tab_chart = st.tabs(
            ["Results", "Individual Models", "Assumptions", "Charts"]
        )

        # ── Results tab ───────────────────────────────────────────────────
        with tab_res:
            mv_tests = result["multivariate_tests"]
            if len(mv_tests) > 0:
                # Show significance from first predictor's Wilks' lambda
                first_wilks = mv_tests[
                    mv_tests["Test"] == "Wilks' lambda"
                ].iloc[0]
                render_significance_result(
                    "Multivariate Test",
                    "Wilks' \u039b F",
                    first_wilks["F"],
                    first_wilks["p"],
                    (int(first_wilks["Num DF"]), int(first_wilks["Den DF"])),
                    alpha,
                )
                st.markdown("---")

                st.markdown("**Multivariate Tests per Predictor**")
                st.dataframe(mv_tests, width="stretch", hide_index=True)
            else:
                st.warning("Could not compute multivariate test statistics.")

            st.metric("N (complete cases)", result["n"])

        # ── Individual Models tab ─────────────────────────────────────────
        with tab_models:
            for m in result["individual_models"]:
                st.markdown(f"### {m['dv']}")

                render_significance_result(
                    f"Model: {m['dv']}",
                    "F",
                    m["f_stat"],
                    m["f_p"],
                    (m["df_model"], m["df_resid"]),
                    alpha,
                )

                cols = st.columns(4)
                cols[0].metric("R\u00b2", f"{m['r_squared']:.4f}")
                cols[1].metric("Adj. R\u00b2", f"{m['adj_r_squared']:.4f}")
                cols[2].metric("AIC", f"{m['aic']:.1f}")
                cols[3].metric("BIC", f"{m['bic']:.1f}")

                st.markdown("**Coefficients**")
                st.dataframe(m["coef_table"], width="stretch", hide_index=True)
                st.markdown("---")

        # ── Assumptions tab ───────────────────────────────────────────────
        with tab_assume:
            st.markdown("**Multivariate Normality of Residuals (Henze-Zirkler)**")
            mvn = result["assumptions"]["multivariate_normality"]
            render_assumption_check(
                "Henze-Zirkler test",
                mvn["statistic"],
                mvn["p_value"],
                mvn["passed"],
                mvn["detail"],
            )

            st.markdown("---")
            st.markdown("**Residual Normality per DV (Shapiro-Wilk)**")
            for dv, sw in result["assumptions"]["residual_normality"].items():
                render_assumption_check(
                    f"Residuals \u2014 {dv}",
                    sw["statistic"],
                    sw["p_value"],
                    sw["passed"],
                    sw["detail"],
                )

        # ── Charts tab ────────────────────────────────────────────────────
        with tab_chart:
            for m in result["individual_models"]:
                st.markdown(f"#### {m['dv']}")
                c1, c2 = st.columns(2)
                with c1:
                    fig = multi_regression_actual_vs_predicted(
                        pd.Series(m["residuals"].values + m["fitted"].values,
                                  name=m["dv"]),
                        pd.Series(m["fitted"], name="Predicted"),
                        title=f"Actual vs Predicted: {m['dv']}",
                    )
                    st.plotly_chart(fig, width="stretch")
                with c2:
                    fig = qq_plot(
                        pd.Series(m["residuals"]),
                        f"Residuals ({m['dv']})",
                    )
                    st.plotly_chart(fig, width="stretch")

        # ── AI Interpretation ──────────────────────────────────────────
        from components.ai_advisor import render_ai_interpretation
        ai_texts = render_ai_interpretation(
            entry_type="multivariate_regression",
            result=result,
            variables={"dvs": ", ".join(dv_cols), "predictors": ", ".join(predictors)},
            alpha=alpha,
            page_key="mvr",
        )

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = []
        if len(result.get("multivariate_tests", [])) > 0:
            _tables.append(_serialize_df(result["multivariate_tests"], "Multivariate Tests"))
        # Per-model performance summary
        _model_summary_rows = []
        for _m in result.get("individual_models", []):
            _model_summary_rows.append({
                "DV": _m["dv"],
                "R²": round(_m["r_squared"], 4),
                "Adj. R²": round(_m["adj_r_squared"], 4),
                "F": round(_m["f_stat"], 4),
                "p (F)": round(_m["f_p"], 6),
                "AIC": round(_m["aic"], 1),
                "BIC": round(_m["bic"], 1),
            })
        if _model_summary_rows:
            _tables.append(_serialize_df(pd.DataFrame(_model_summary_rows), "Model Performance Summary"))
        for _m in result.get("individual_models", []):
            _tables.append(_serialize_df(_m["coef_table"], f"Coefficients: {_m['dv']}"))
        _log_entry = build_log_entry(
            entry_type="multivariate_regression",
            title=f"Multivariate Regression: {', '.join(dv_cols[:3])}{'...' if len(dv_cols) > 3 else ''}",
            result=result,
            tables=_tables,
            variables={"dvs": ", ".join(dv_cols), "predictors": ", ".join(predictors)},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        if ai_texts.get("brief"):
            _log_entry["ai_interpretation"] = ai_texts["brief"]
        if ai_texts.get("deep_dive"):
            _log_entry["ai_deep_dive"] = ai_texts["deep_dive"]
        _include_chart = st.checkbox("Include charts in PDF", value=True, key="mvr_pdf_chart")
        if _include_chart:
            _figures = []
            for _m in result.get("individual_models", [])[:4]:
                _avp = multi_regression_actual_vs_predicted(
                    pd.Series(_m["residuals"].values + _m["fitted"].values, name=_m["dv"]),
                    pd.Series(_m["fitted"], name="Predicted"),
                    title=f"Actual vs Predicted: {_m['dv']}",
                )
                _figures.append({"label": f"Actual vs Predicted: {_m['dv']}", "fig_dict": _avp.to_dict()})
                _qqfig = qq_plot(
                    pd.Series(_m["residuals"]),
                    f"Residuals ({_m['dv']})",
                )
                _figures.append({"label": f"Q-Q Plot: {_m['dv']}", "fig_dict": _qqfig.to_dict()})
            _log_entry["figures"] = _figures
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="mvr_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="multivariate_regression.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
Multivariate linear regression predicts **two or more continuous outcomes** (dependent variables) simultaneously from one or more **predictor variables**.  It is the multivariate extension of standard linear regression.

#### When to Use
- Two or more **continuous dependent variables** (DVs).
- One or more **continuous predictors** (IVs).
- You want to test whether predictors are jointly related to the set of DVs, accounting for correlations among outcomes.

#### Results Tab
- **Multivariate test statistics** per predictor:
    - **Wilks' Lambda** — most commonly reported; values closer to 0 indicate a stronger effect.
    - **Pillai's Trace** — more robust to assumption violations.
    - **Hotelling-Lawley Trace** — powerful when effects are concentrated along one dimension.
    - **Roy's Greatest Root** — most powerful for single-dimension effects, but most sensitive to violations.
- Each row shows the **F-statistic** and **p-value** for the multivariate effect of that predictor.

#### Individual Models Tab
- Separate **OLS regression** results for each DV:
    - **R-squared** and **Adjusted R-squared** — proportion of variance explained.
    - **F-test** — overall model significance for that DV.
    - **Coefficient table** — unstandardized (B) and standardized (Beta) coefficients, standard errors, t-values, p-values, and confidence intervals.
    - **AIC / BIC** — model comparison metrics.

#### Assumptions Tab
- **Multivariate normality of residuals (Henze-Zirkler)** — tests whether the residuals from all DVs jointly follow a multivariate normal distribution.
- **Shapiro-Wilk per DV residuals** — univariate normality check for each outcome's residuals.

#### Charts Tab
- **Actual vs Predicted** plots per DV — points near the diagonal line indicate good fit.
- **Q-Q plots** of residuals per DV — points on the diagonal suggest normally distributed residuals.
        """)

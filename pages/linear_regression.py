"""Linear Regression page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_multiple_variables
from components.results_display import (
    render_significance_result, render_assumption_check, render_effect_size,
)
from stats.regression import linear_regression
from charts.regression_plot import regression_scatter, multi_regression_actual_vs_predicted
from charts.qq_plot import qq_plot
from charts.histogram import histogram_with_normal
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df

_CACHE_KEY = "_result_lin_reg"


def render():
    st.title("Linear Regression")
    st.markdown("Predict a continuous outcome from one or more predictors (OLS).")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    dv = select_metric_variable("Dependent variable (outcome)", key="lr_dv")
    predictors = select_multiple_variables("Predictor variables", key="lr_pred", var_type="Metric")

    with st.expander("Options"):
        alpha = st.slider("Significance level (\u03b1)", 0.01, 0.10, 0.05, 0.01, key="lr_alpha")

    if dv and predictors and st.button("Calculate", type="primary"):
        if dv in predictors:
            st.error("The dependent variable cannot also be a predictor.")
            return

        result = linear_regression(df, dv, predictors, alpha=alpha)

        if "error" in result:
            st.error(result["error"])
            return

        # Extract endog before caching (model object is not serializable)
        _endog = result["model"].model.endog.copy()

        # Remove non-serializable model object from result before caching
        result_for_cache = {k: v for k, v in result.items() if k != "model"}

        st.session_state[_CACHE_KEY] = {
            "inputs": (dv, tuple(predictors), alpha),
            "result": result_for_cache,
            "endog": _endog,
        }

    # ── Cache invalidation ─────────────────────────────────────────────
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached["inputs"] != (dv, tuple(predictors or []), alpha):
        del st.session_state[_CACHE_KEY]
        cached = None

    if cached:
        result = cached["result"]
        _endog = cached["endog"]

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            # Model summary
            render_significance_result(
                "Overall Model", "F", result["f_stat"], result["f_p"],
                (result["df_model"], result["df_resid"]), alpha
            )
            st.markdown("---")

            cols = st.columns(4)
            cols[0].metric("R\u00b2", f"{result['r_squared']:.4f}")
            cols[1].metric("Adj. R\u00b2", f"{result['adj_r_squared']:.4f}")
            cols[2].metric("N", result["n"])
            cols[3].metric("AIC", f"{result['aic']:.1f}")

            st.markdown("### Coefficients")
            st.dataframe(result["coef_table"], width="stretch", hide_index=True)

        with tab_assume:
            norm = result["assumptions"]["residual_normality"]
            render_assumption_check(
                "Shapiro-Wilk (Residual Normality)", norm["statistic"],
                norm["p_value"], norm["passed"], norm["detail"]
            )

        with tab_chart:
            if len(predictors) == 1:
                clean = df[[dv, predictors[0]]].dropna().copy()
                for c in clean.columns:
                    clean[c] = pd.to_numeric(clean[c], errors="coerce")
                clean = clean.dropna()
                fig = regression_scatter(clean, predictors[0], dv)
                st.plotly_chart(fig, width="stretch")
            else:
                fig = multi_regression_actual_vs_predicted(
                    pd.Series(_endog),
                    pd.Series(result["fitted"]),
                )
                st.plotly_chart(fig, width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                fig = qq_plot(pd.Series(result["residuals"]), "Residuals")
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = histogram_with_normal(pd.Series(result["residuals"]), "Residuals")
                st.plotly_chart(fig, width="stretch")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = [_serialize_df(result["coef_table"], "Coefficients")]
        _log_entry = build_log_entry(
            entry_type="linear_regression",
            title=f"Linear Regression: {dv} ~ {' + '.join(predictors[:4])}{'...' if len(predictors) > 4 else ''}",
            result=result,
            tables=_tables,
            variables={"dv": dv, "predictors": ", ".join(predictors)},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include charts in PDF", value=True, key="lr_pdf_chart")
        if _include_chart:
            _figures = []
            if len(predictors) == 1:
                _clean = df[[dv, predictors[0]]].dropna()
                for _c in _clean.columns:
                    _clean[_c] = pd.to_numeric(_clean[_c], errors="coerce")
                _clean = _clean.dropna()
                _sfig = regression_scatter(_clean, predictors[0], dv)
                _figures.append({"label": "Regression Scatter", "fig_dict": _sfig.to_dict()})
            else:
                _avp = multi_regression_actual_vs_predicted(
                    pd.Series(_endog),
                    pd.Series(result["fitted"]),
                )
                _figures.append({"label": "Actual vs Predicted", "fig_dict": _avp.to_dict()})
            _qfig = qq_plot(pd.Series(result["residuals"]), "Residuals")
            _figures.append({"label": "Q-Q Plot (Residuals)", "fig_dict": _qfig.to_dict()})
            _hfig = histogram_with_normal(pd.Series(result["residuals"]), "Residuals")
            _figures.append({"label": "Histogram (Residuals)", "fig_dict": _hfig.to_dict()})
            _log_entry["figures"] = _figures
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="lr_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="linear_regression.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
Linear regression predicts a **continuous outcome** (dependent variable) from one or more **predictor variables** using **Ordinary Least Squares (OLS)** estimation.

#### When to Use
- The **dependent variable** is continuous (metric).
- One or more **predictors** are metric (continuous).
- The relationship between predictors and the outcome is expected to be **linear**.

#### Results Tab
- **Overall model F-test** -- tests whether the model as a whole explains a significant amount of variance (i.e., at least one predictor is useful).
- **R-squared** -- proportion of variance in the DV explained by the predictors (0 to 1).
- **Adjusted R-squared** -- R-squared corrected for the number of predictors; more appropriate for comparing models with different numbers of predictors.
- **N** -- number of observations used.
- **AIC (Akaike Information Criterion)** -- model fit measure; lower values indicate better fit (useful for model comparison).
- **Coefficient table** for each predictor:
    - **B (coefficient)** -- expected change in the DV for a one-unit increase in the predictor, holding all other predictors constant.
    - **SE** -- standard error of the coefficient.
    - **t** and **p** -- test whether the coefficient is significantly different from zero.
    - **95% CI** -- confidence interval for the coefficient.

#### Assumptions Tab
- **Shapiro-Wilk on residuals** -- tests whether the **residuals** (prediction errors) are normally distributed. Violations may affect confidence intervals and p-values but not the coefficient estimates themselves.

#### Charts Tab
- **Scatter with regression line** (simple regression with one predictor) or **Actual vs. Predicted plot** (multiple regression) -- shows how well the model fits the data.
- **Q-Q plot of residuals** -- points should follow the diagonal if residuals are normally distributed.
- **Histogram of residuals** -- visualizes the distribution of residuals with a normal curve overlay.
        """)

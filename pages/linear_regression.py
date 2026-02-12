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
from core.state import get_df


def render():
    st.title("Linear Regression")
    st.markdown("Predict a continuous outcome from one or more predictors (OLS).")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    dv = select_metric_variable("Dependent variable (outcome)", key="lr_dv")
    predictors = select_multiple_variables("Predictor variables", key="lr_pred", var_type="Metric")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="lr_alpha")

    if dv and predictors and st.button("Calculate", type="primary"):
        if dv in predictors:
            st.error("The dependent variable cannot also be a predictor.")
            return

        result = linear_regression(df, dv, predictors, alpha=alpha)

        if "error" in result:
            st.error(result["error"])
            return

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            # Model summary
            render_significance_result(
                "Overall Model", "F", result["f_stat"], result["f_p"],
                (result["df_model"], result["df_resid"]), alpha
            )
            st.markdown("---")

            cols = st.columns(4)
            cols[0].metric("R²", f"{result['r_squared']:.4f}")
            cols[1].metric("Adj. R²", f"{result['adj_r_squared']:.4f}")
            cols[2].metric("N", result["n"])
            cols[3].metric("AIC", f"{result['aic']:.1f}")

            st.markdown("### Coefficients")
            st.dataframe(result["coef_table"], use_container_width=True, hide_index=True)

        with tab_assume:
            norm = result["assumptions"]["residual_normality"]
            render_assumption_check(
                "Shapiro-Wilk (Residual Normality)", norm["statistic"],
                norm["p_value"], norm["passed"], norm["detail"]
            )

        with tab_chart:
            if len(predictors) == 1:
                clean = df[[dv, predictors[0]]].dropna()
                for c in clean.columns:
                    clean[c] = pd.to_numeric(clean[c], errors="coerce")
                clean = clean.dropna()
                fig = regression_scatter(clean, predictors[0], dv)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = multi_regression_actual_vs_predicted(
                    pd.Series(result["model"].model.endog),
                    pd.Series(result["fitted"]),
                )
                st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = qq_plot(pd.Series(result["residuals"]), "Residuals")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = histogram_with_normal(pd.Series(result["residuals"]), "Residuals")
                st.plotly_chart(fig, use_container_width=True)

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


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

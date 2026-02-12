"""Chi-Squared Test page."""

import streamlit as st
import pandas as pd
import plotly.express as px
from components.data_table import render_data_preview
from components.variable_selector import select_nominal_variable
from components.results_display import render_significance_result, render_assumption_check, render_effect_size
from stats.chi_squared import chi_squared_test
from core.state import get_df
from charts.theme import apply_theme, COLORS


def render():
    st.title("Chi-Squared Test")
    st.markdown("Test the association between two categorical variables.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        var1 = select_nominal_variable("Variable 1", key="chi_v1")
    with col2:
        var2 = select_nominal_variable("Variable 2", key="chi_v2")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="chi_alpha")

    if var1 and var2 and st.button("Calculate", type="primary"):
        if var1 == var2:
            st.error("Please select two different variables.")
            return

        result = chi_squared_test(df, var1, var2, alpha=alpha)

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "χ²", result["chi2"], result["p"], result["df"], alpha
            )
            st.markdown("---")

            st.markdown("### Observed Frequencies")
            st.dataframe(result["contingency"], use_container_width=True)

            st.markdown("### Expected Frequencies")
            st.dataframe(
                result["expected"].round(2), use_container_width=True
            )

            cols = st.columns(3)
            cols[0].metric("N", result["n"])
            cols[1].metric("Cramér's V", f"{result['cramers_v']:.4f}")
            cols[2].metric("df", result["df"])

        with tab_assume:
            ef = result["assumptions"]["expected_frequencies"]
            render_assumption_check(
                "Expected Frequencies ≥ 5", None, None,
                ef["passed"], ef["detail"]
            )

        with tab_chart:
            # Stacked bar chart
            contingency = result["contingency"]
            fig = px.bar(
                contingency.T,
                barmode="group",
                color_discrete_sequence=COLORS,
                title=f"Frequencies: {var1} × {var2}",
            )
            fig.update_layout(xaxis_title=var2, yaxis_title="Count")
            st.plotly_chart(apply_theme(fig), use_container_width=True)

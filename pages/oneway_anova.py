"""One-Way ANOVA page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import (
    render_significance_result, render_assumption_check,
    render_effect_size, interpret_eta_squared,
)
from stats.anova import oneway_anova
from core.validators import validate_groups
from charts.boxplot import grouped_boxplot
from charts.barplot import group_means_bar
from core.state import get_df


def render():
    st.title("One-Way ANOVA")
    st.markdown("Compare means across three or more independent groups.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        dv = select_metric_variable("Dependent variable", key="ow_dv")
    with col2:
        group = select_nominal_variable("Factor (grouping variable)", key="ow_group")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="ow_alpha")

    if dv and group and st.button("Calculate", type="primary"):
        valid, msg = validate_groups(dv, group, min_groups=2)
        if not valid:
            st.error(msg)
            return

        clean = df[[dv, group]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        result = oneway_anova(clean, dv, group, alpha=alpha)

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "F", result["F"], result["p"],
                (result["df_between"], result["df_within"]), alpha
            )
            st.markdown("---")

            st.markdown("**ANOVA Table**")
            st.dataframe(result["anova_table"], use_container_width=True, hide_index=True)

            st.markdown("**Group Descriptives**")
            st.dataframe(result["group_desc"], use_container_width=True, hide_index=True)

            render_effect_size("η² (Eta-squared)", result["eta_squared"], interpret_eta_squared(result["eta_squared"]))
            render_effect_size("ω² (Omega-squared)", result["omega_squared"], interpret_eta_squared(result["omega_squared"]))

            if result["posthoc"] is not None:
                st.markdown("### Post-Hoc Tests (Tukey HSD)")
                st.dataframe(result["posthoc"], use_container_width=True, hide_index=True)

        with tab_assume:
            st.markdown("**Normality (Shapiro-Wilk per group)**")
            for gname, norm in result["assumptions"]["normality"].items():
                render_assumption_check(
                    f"Group: {gname}", norm["statistic"], norm["p_value"],
                    norm["passed"], norm["detail"]
                )
            st.markdown("---")
            homo = result["assumptions"]["homogeneity"]
            render_assumption_check(
                "Levene's Test (Homogeneity of Variances)", homo["statistic"],
                homo["p_value"], homo["passed"], homo["detail"]
            )

        with tab_chart:
            c1, c2 = st.columns(2)
            with c1:
                fig = grouped_boxplot(clean, dv, group)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = group_means_bar(clean, dv, group)
                st.plotly_chart(fig, use_container_width=True)

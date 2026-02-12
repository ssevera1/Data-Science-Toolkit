"""Kruskal-Wallis H Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import render_significance_result, render_effect_size
from stats.nonparametric import kruskal_wallis
from core.validators import validate_groups
from charts.boxplot import grouped_boxplot
from core.state import get_df


def render():
    st.title("Kruskal-Wallis H Test")
    st.markdown("Non-parametric alternative to one-way ANOVA.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        dv = select_metric_variable("Test variable", key="kw_dv")
    with col2:
        group = select_nominal_variable("Grouping variable", key="kw_group")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="kw_alpha")

    if dv and group and st.button("Calculate", type="primary"):
        valid, msg = validate_groups(dv, group, min_groups=2)
        if not valid:
            st.error(msg)
            return

        clean = df[[dv, group]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        result = kruskal_wallis(clean, dv, group, alpha=alpha)

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "H", result["H"], result["p"], result["df"], alpha
            )
            st.markdown("---")

            st.markdown("**Group Descriptives**")
            st.dataframe(result["group_desc"], use_container_width=True, hide_index=True)

            render_effect_size("ε² (Epsilon-squared)", result["epsilon_squared"])

            if result["posthoc"] is not None:
                st.markdown("### Post-Hoc Pairwise Comparisons (Bonferroni)")
                st.dataframe(result["posthoc"], use_container_width=True, hide_index=True)

        with tab_chart:
            fig = grouped_boxplot(clean, dv, group)
            st.plotly_chart(fig, use_container_width=True)

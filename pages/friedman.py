"""Friedman Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_any_variable
from components.results_display import render_significance_result, render_effect_size
from stats.nonparametric import friedman_test
from charts.boxplot import grouped_boxplot
from charts.barplot import group_means_bar
from core.state import get_df


def render():
    st.title("Friedman Test")
    st.markdown("Non-parametric alternative to repeated measures ANOVA.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2, col3 = st.columns(3)
    with col1:
        dv = select_metric_variable("Dependent variable", key="fr_dv")
    with col2:
        within = select_any_variable("Within-subjects factor", key="fr_within")
    with col3:
        subject = select_any_variable("Subject ID", key="fr_subject")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="fr_alpha")

    if dv and within and subject and st.button("Calculate", type="primary"):
        clean = df[[dv, within, subject]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        result = friedman_test(clean, dv, within, subject, alpha=alpha)

        if result.get("chi2") is None:
            st.error(result.get("detail", "Could not compute test."))
            return

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "χ²", result["chi2"], result["p"], result["df"], alpha
            )
            st.markdown("---")

            cols = st.columns(3)
            cols[0].metric("N (subjects)", result["n_subjects"])
            cols[1].metric("K (conditions)", result["n_conditions"])
            cols[2].metric("Kendall's W", f"{result['kendalls_w']:.4f}")

            st.markdown("**Group Descriptives**")
            st.dataframe(result["group_desc"], use_container_width=True, hide_index=True)

            if result["posthoc"] is not None:
                st.markdown("### Post-Hoc Pairwise Comparisons (Bonferroni)")
                st.dataframe(result["posthoc"], use_container_width=True, hide_index=True)

        with tab_chart:
            c1, c2 = st.columns(2)
            with c1:
                fig = grouped_boxplot(clean, dv, within)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = group_means_bar(clean, dv, within)
                st.plotly_chart(fig, use_container_width=True)

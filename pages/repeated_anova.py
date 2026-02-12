"""Repeated Measures ANOVA page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_any_variable
from components.results_display import render_assumption_check
from stats.anova import repeated_measures_anova
from charts.boxplot import grouped_boxplot
from charts.barplot import group_means_bar
from core.state import get_df


def render():
    st.title("Repeated Measures ANOVA")
    st.markdown("Compare means across three or more related conditions (within-subjects).")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2, col3 = st.columns(3)
    with col1:
        dv = select_metric_variable("Dependent variable", key="rm_dv")
    with col2:
        within = select_any_variable("Within-subjects factor", key="rm_within")
    with col3:
        subject = select_any_variable("Subject ID", key="rm_subject")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="rm_alpha")

    if dv and within and subject and st.button("Calculate", type="primary"):
        clean = df[[dv, within, subject]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        if len(clean[within].unique()) < 2:
            st.error("Need at least 2 conditions in the within-subjects factor.")
            return

        try:
            result = repeated_measures_anova(clean, dv, within, subject, alpha=alpha)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            st.markdown("### ANOVA Table")
            st.dataframe(result["anova_table"], use_container_width=True, hide_index=True)

            st.markdown("### Group Descriptives")
            st.dataframe(result["group_desc"], use_container_width=True, hide_index=True)

            if result["posthoc"] is not None:
                st.markdown("### Post-Hoc Pairwise Comparisons (Bonferroni)")
                st.dataframe(result["posthoc"], use_container_width=True, hide_index=True)

        with tab_assume:
            spher = result["assumptions"].get("sphericity")
            if spher:
                render_assumption_check(
                    "Mauchly's Test of Sphericity", spher["statistic"],
                    spher["p_value"], spher["passed"], spher["detail"]
                )
            else:
                st.info("Sphericity test not applicable (2 conditions) or could not be computed.")

        with tab_chart:
            c1, c2 = st.columns(2)
            with c1:
                fig = grouped_boxplot(clean, dv, within)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = group_means_bar(clean, dv, within)
                st.plotly_chart(fig, use_container_width=True)

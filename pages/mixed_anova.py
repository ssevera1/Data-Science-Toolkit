"""Mixed ANOVA page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_any_variable, select_nominal_variable
from stats.anova import mixed_anova
from charts.barplot import two_way_bar
from core.state import get_df


def render():
    st.title("Mixed ANOVA")
    st.markdown("Test effects of one within-subjects and one between-subjects factor.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        dv = select_metric_variable("Dependent variable", key="mx_dv")
        within = select_any_variable("Within-subjects factor", key="mx_within")
    with col2:
        between = select_nominal_variable("Between-subjects factor", key="mx_between")
        subject = select_any_variable("Subject ID", key="mx_subject")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="mx_alpha")

    if dv and within and between and subject and st.button("Calculate", type="primary"):
        clean = df[[dv, within, between, subject]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        try:
            result = mixed_anova(clean, dv, within, between, subject, alpha=alpha)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            st.markdown("### ANOVA Table")
            aov = result["anova_table"]
            st.dataframe(aov, use_container_width=True, hide_index=True)

            for _, row in aov.iterrows():
                source = row.get("Source", "")
                p = row.get("p-unc", None)
                if p is not None and source:
                    sig = "✓ Significant" if p < alpha else "✗ Not significant"
                    st.markdown(f"**{source}:** F = {row.get('F', 0):.4f}, p = {p:.4f} — {sig}")

            st.markdown("### Group Descriptives")
            st.dataframe(result["group_desc"], use_container_width=True, hide_index=True)

        with tab_chart:
            fig = two_way_bar(clean, dv, within, between)
            st.plotly_chart(fig, use_container_width=True)

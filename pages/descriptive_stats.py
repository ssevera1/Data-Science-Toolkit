"""Descriptive Statistics page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_multiple_variables
from stats.descriptive import compute_descriptives_table, compute_frequency_table
from charts.histogram import histogram_with_normal
from charts.boxplot import single_boxplot
from core.state import get_df, get_var_type


def render():
    st.title("Descriptive Statistics")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    # Variable selection
    st.subheader("Select Variables")
    selected = select_multiple_variables("Choose variables to analyze", key="desc_vars")

    if not selected:
        st.info("Select one or more variables to compute descriptive statistics.")
        return

    if st.button("Calculate", type="primary"):
        # Separate metric and categorical
        metric_vars = [v for v in selected if get_var_type(v) in ("Metric", "Ordinal")]
        nominal_vars = [v for v in selected if get_var_type(v) == "Nominal"]

        tab_results, tab_charts = st.tabs(["Results", "Charts"])

        with tab_results:
            if metric_vars:
                st.markdown("### Numeric Variables")
                desc_table = compute_descriptives_table(df, metric_vars)
                st.dataframe(desc_table.T, use_container_width=True)

            if nominal_vars:
                st.markdown("### Categorical Variables")
                for var in nominal_vars:
                    st.markdown(f"**{var}**")
                    freq = compute_frequency_table(df[var].dropna())
                    st.dataframe(freq, use_container_width=True, hide_index=True)

        with tab_charts:
            if metric_vars:
                for var in metric_vars:
                    series = pd.to_numeric(df[var], errors="coerce").dropna()
                    if len(series) > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = histogram_with_normal(series, var)
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            fig = single_boxplot(series, var)
                            st.plotly_chart(fig, use_container_width=True)

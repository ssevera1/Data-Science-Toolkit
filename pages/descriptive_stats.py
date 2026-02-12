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
                st.dataframe(desc_table.T, width="stretch")

            if nominal_vars:
                st.markdown("### Categorical Variables")
                for var in nominal_vars:
                    st.markdown(f"**{var}**")
                    freq = compute_frequency_table(df[var].dropna())
                    st.dataframe(freq, width="stretch", hide_index=True)

        with tab_charts:
            if metric_vars:
                for var in metric_vars:
                    series = pd.to_numeric(df[var], errors="coerce").dropna()
                    if len(series) > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = histogram_with_normal(series, var)
                            st.plotly_chart(fig, width="stretch")
                        with col2:
                            fig = single_boxplot(series, var)
                            st.plotly_chart(fig, width="stretch")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Overview
Select one or more variables to compute descriptive statistics. The statistics and charts produced depend on the variable type assigned on the Data Input page.

#### Numeric Variables (Metric / Ordinal)
A summary table is generated with the following statistics for each selected variable:

- **N** -- number of non-missing observations
- **Mean** -- arithmetic average
- **Standard Deviation (SD)** -- measure of spread around the mean
- **Min / Max** -- smallest and largest observed values
- **Q1 (25th percentile)** -- value below which 25% of observations fall
- **Median (Q2, 50th percentile)** -- middle value of the sorted data
- **Q3 (75th percentile)** -- value below which 75% of observations fall
- **Skewness** -- measure of distribution asymmetry (0 = symmetric; positive = right-skewed; negative = left-skewed)
- **Kurtosis** -- measure of tail heaviness relative to a normal distribution (0 = normal-like tails)
- **Standard Error (SE)** -- standard deviation of the sampling distribution of the mean (SD / sqrt(N))

#### Categorical Variables (Nominal)
A **frequency table** is displayed for each nominal variable, showing:

- **Count** -- number of observations in each category
- **Percentage** -- proportion of total observations per category

#### Charts
- **Histogram with normal curve overlay** -- shows the shape of the distribution; the curve helps assess whether the data approximate a normal distribution.
- **Box plot** -- displays the median (center line), interquartile range (box), and potential outliers (points beyond the whiskers).
        """)


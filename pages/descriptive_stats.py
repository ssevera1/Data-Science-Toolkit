"""Descriptive Statistics page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_multiple_variables
from stats.descriptive import compute_descriptives_table, compute_frequency_table
from charts.histogram import histogram_with_normal
from charts.boxplot import single_boxplot
from core.state import get_df, get_var_type, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df

_CACHE_KEY = "_result_descriptive"


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

        # Compute descriptive tables
        desc_table = None
        if metric_vars:
            desc_table = compute_descriptives_table(df, metric_vars)

        freq_tables = {}
        if nominal_vars:
            for var in nominal_vars:
                freq_tables[var] = compute_frequency_table(df[var].dropna())

        # Store in session state cache
        st.session_state[_CACHE_KEY] = {
            "inputs": (tuple(selected),),
            "metric_vars": metric_vars,
            "nominal_vars": nominal_vars,
            "desc_table": desc_table,
            "freq_tables": freq_tables,
        }

    # ── Invalidate cache if inputs changed ──────────────────────────
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached["inputs"] != (tuple(selected),):
        del st.session_state[_CACHE_KEY]
        cached = None

    if cached:
        metric_vars = cached["metric_vars"]
        nominal_vars = cached["nominal_vars"]
        desc_table = cached["desc_table"]
        freq_tables = cached["freq_tables"]

        tab_results, tab_charts = st.tabs(["Results", "Charts"])

        with tab_results:
            if metric_vars:
                st.markdown("### Numeric Variables")
                st.dataframe(desc_table.T, width="stretch")

            if nominal_vars:
                st.markdown("### Categorical Variables")
                for var in nominal_vars:
                    st.markdown(f"**{var}**")
                    st.dataframe(freq_tables[var], width="stretch", hide_index=True)

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

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = []
        if metric_vars:
            _tables.append(_serialize_df(desc_table.T, "Descriptive Statistics"))
        if nominal_vars:
            for _nv in nominal_vars:
                _tables.append(_serialize_df(freq_tables[_nv], f"Frequency: {_nv}"))

        _log_entry = build_log_entry(
            entry_type="descriptive_stats",
            title=f"Descriptive Statistics: {', '.join(selected[:5])}{'...' if len(selected) > 5 else ''}",
            result={},
            tables=_tables,
            variables={"variables": ", ".join(selected)},
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include charts in PDF", value=True, key="desc_pdf_chart")
        if _include_chart and metric_vars:
            _figures = []
            for _mv in metric_vars[:6]:
                _s = pd.to_numeric(df[_mv], errors="coerce").dropna()
                if len(_s) > 0:
                    _hfig = histogram_with_normal(_s, _mv)
                    _figures.append({"label": f"Histogram: {_mv}", "fig_dict": _hfig.to_dict()})
            _log_entry["figures"] = _figures
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="desc_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="descriptive_stats.pdf",
                mime="application/pdf",
            )

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


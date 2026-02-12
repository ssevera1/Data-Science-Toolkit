"""Reusable variable picker dropdowns."""

import streamlit as st
from core.state import get_df, get_var_type


def select_metric_variable(label="Select metric variable", key=None):
    """Dropdown for selecting a metric (numeric) variable."""
    df = get_df()
    metric_cols = [c for c in df.columns if get_var_type(c) == "Metric"]
    all_cols = list(df.columns)

    # Show metric cols first, then others
    options = [""] + metric_cols + [c for c in all_cols if c not in metric_cols]

    selected = st.selectbox(
        label,
        options=options,
        format_func=lambda x: "-- Select --" if x == "" else f"{x} ({get_var_type(x)})",
        key=key,
    )
    return selected if selected != "" else None


def select_nominal_variable(label="Select grouping variable", key=None):
    """Dropdown for selecting a nominal (categorical) variable."""
    df = get_df()
    nominal_cols = [c for c in df.columns if get_var_type(c) == "Nominal"]
    all_cols = list(df.columns)

    options = [""] + nominal_cols + [c for c in all_cols if c not in nominal_cols]

    selected = st.selectbox(
        label,
        options=options,
        format_func=lambda x: "-- Select --" if x == "" else f"{x} ({get_var_type(x)})",
        key=key,
    )
    return selected if selected != "" else None


def select_any_variable(label="Select variable", key=None, exclude=None):
    """Dropdown for selecting any variable."""
    df = get_df()
    cols = [c for c in df.columns if exclude is None or c not in exclude]
    options = [""] + cols

    selected = st.selectbox(
        label,
        options=options,
        format_func=lambda x: "-- Select --" if x == "" else f"{x} ({get_var_type(x)})",
        key=key,
    )
    return selected if selected != "" else None


def select_multiple_variables(label="Select variables", key=None, var_type=None):
    """Multi-select for variables."""
    df = get_df()
    if var_type:
        cols = [c for c in df.columns if get_var_type(c) == var_type]
    else:
        cols = list(df.columns)

    selected = st.multiselect(
        label,
        options=cols,
        format_func=lambda x: f"{x} ({get_var_type(x)})",
        key=key,
    )
    return selected

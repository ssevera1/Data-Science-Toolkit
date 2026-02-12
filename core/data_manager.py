"""Data loading, export, and column management."""

import io
import pandas as pd
import numpy as np
import streamlit as st
from core.state import set_df, get_df, get_var_types, set_var_type


def load_csv(uploaded_file):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        _apply_loaded_df(df)
        return True, None
    except Exception:
        return False, "Unable to load CSV file. Please check the format."


def load_excel(uploaded_file):
    """Load data from an Excel file."""
    try:
        df = pd.read_excel(uploaded_file)
        _apply_loaded_df(df)
        return True, None
    except Exception:
        return False, "Unable to load Excel file. Please check the format."


def load_from_paste(text):
    """Load data from pasted text (tab or comma separated)."""
    try:
        # Try tab-separated first
        df = pd.read_csv(io.StringIO(text), sep="\t")
        if len(df.columns) == 1:
            # Try comma-separated
            df = pd.read_csv(io.StringIO(text), sep=",")
        if len(df.columns) == 1:
            # Try semicolon
            df = pd.read_csv(io.StringIO(text), sep=";")
        _apply_loaded_df(df)
        return True, None
    except Exception:
        return False, "Unable to parse pasted data. Please check the format."


def _apply_loaded_df(df):
    """Apply a loaded DataFrame to session state."""
    # Pad with empty rows so user can add more data
    if len(df) < 20:
        extra = pd.DataFrame(
            np.nan, index=range(len(df), 20), columns=df.columns
        )
        df = pd.concat([df, extra], ignore_index=True)

    set_df(df)

    # Auto-detect variable types
    for col in df.columns:
        _auto_detect_type(col, df)


def _auto_detect_type(col, df):
    """Auto-detect variable type for a column."""
    series = df[col].dropna()
    if len(series) == 0:
        set_var_type(col, "Metric")
        return

    # Check if numeric
    numeric = pd.to_numeric(series, errors="coerce")
    non_null_numeric = numeric.notna().sum()

    if non_null_numeric / len(series) > 0.5:
        n_unique = numeric.dropna().nunique()
        if n_unique <= 2:
            set_var_type(col, "Nominal")
        elif n_unique <= 7 and n_unique < len(series) * 0.3:
            set_var_type(col, "Ordinal")
        else:
            set_var_type(col, "Metric")
    else:
        n_unique = series.nunique()
        if n_unique <= 10:
            set_var_type(col, "Nominal")
        else:
            set_var_type(col, "Text")


def add_column():
    """Add a new column to the DataFrame."""
    df = get_df()
    st.session_state.stats_col_counter += 1
    new_col = f"Var{st.session_state.stats_col_counter}"
    df[new_col] = np.nan
    set_df(df)
    set_var_type(new_col, "Metric")


def add_rows(n=10):
    """Add rows to the DataFrame."""
    df = get_df()
    extra = pd.DataFrame(
        np.nan, index=range(len(df), len(df) + n), columns=df.columns
    )
    set_df(pd.concat([df, extra], ignore_index=True))


def _sanitize_formula_cell(value):
    """Prefix cells starting with formula characters to prevent CSV injection."""
    if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def export_csv():
    """Export the current data as CSV string."""
    df = get_df().dropna(how="all")
    sanitized = df.apply(
        lambda col: col.map(_sanitize_formula_cell) if col.dtype == object else col
    )
    return sanitized.to_csv(index=False)


def export_excel():
    """Export the current data as Excel bytes."""
    df = get_df().dropna(how="all")
    sanitized = df.apply(
        lambda col: col.map(_sanitize_formula_cell) if col.dtype == object else col
    )
    output = io.BytesIO()
    sanitized.to_excel(output, index=False)
    return output.getvalue()

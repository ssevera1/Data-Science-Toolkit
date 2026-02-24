"""Session state initialization and management."""

import pandas as pd
import numpy as np
import streamlit as st
from core.constants import DEFAULT_ROWS, DEFAULT_COLS


def init_state():
    """Initialize all session state variables if not already set.

    The 'df' key is shared between Data Science and Statistics tools.
    It is NOT created here — it's created on first use by the Stats
    Data Input page or by uploading a file on the Home page.
    """
    if "stats_var_types" not in st.session_state:
        st.session_state.stats_var_types = {}

    if "stats_col_counter" not in st.session_state:
        st.session_state.stats_col_counter = DEFAULT_COLS


def _create_empty_df(n_rows=DEFAULT_ROWS, n_cols=DEFAULT_COLS):
    """Create an empty DataFrame with named columns."""
    columns = [f"Var{i+1}" for i in range(n_cols)]
    return pd.DataFrame(
        np.nan,
        index=range(n_rows),
        columns=columns,
    )


def get_df():
    """Get the current DataFrame from session state.

    Lazily creates an empty editor DataFrame if none exists yet.
    """
    if "df" not in st.session_state:
        st.session_state["df"] = _create_empty_df()
    return st.session_state["df"]


def set_df(df):
    """Set the DataFrame in session state (shared between DS and Stats)."""
    st.session_state["df"] = df


def get_var_types():
    """Get the variable types dictionary."""
    return st.session_state.stats_var_types


def set_var_type(col, vtype):
    """Set the variable type for a column."""
    st.session_state.stats_var_types[col] = vtype


def get_var_type(col):
    """Get the variable type for a column, defaulting to Metric."""
    return st.session_state.stats_var_types.get(col, "Metric")


def get_numeric_columns():
    """Get columns that have at least some numeric data."""
    df = get_df()
    cols = []
    for c in df.columns:
        try:
            numeric = pd.to_numeric(df[c], errors="coerce")
            if numeric.notna().any():
                cols.append(c)
        except Exception:
            pass
    return cols


def get_columns_by_type(vtype):
    """Get columns matching a specific variable type."""
    df = get_df()
    return [c for c in df.columns if get_var_type(c) == vtype]


def get_clean_df():
    """Get the DataFrame with rows that have all NaN removed."""
    df = get_df()
    return df.dropna(how="all").reset_index(drop=True)


_MAX_LOG_ENTRIES = 100


def log_result(entry: dict) -> bool:
    """Append a result entry to the session report log.

    Returns False if the log is full.
    """
    if "report_log" not in st.session_state:
        st.session_state["report_log"] = []
    if len(st.session_state["report_log"]) >= _MAX_LOG_ENTRIES:
        return False
    st.session_state["report_log"].append(entry)
    return True


def get_report_log() -> list:
    """Return the current report log."""
    return st.session_state.get("report_log", [])


def clear_report_log() -> None:
    """Clear all entries from the report log."""
    st.session_state["report_log"] = []

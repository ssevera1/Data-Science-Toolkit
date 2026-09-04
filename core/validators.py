"""Input validation for statistical tests."""

import pandas as pd
import numpy as np
from core.state import get_df, get_var_type


def validate_column_exists(col_name):
    """Check if column exists in the DataFrame."""
    df = get_df()
    if col_name not in df.columns:
        return False, f"Column '{col_name}' not found."
    return True, None


def validate_metric(col_name):
    """Validate that a column has numeric data."""
    df = get_df()
    series = pd.to_numeric(df[col_name], errors="coerce").dropna()
    if len(series) < 2:
        return False, f"'{col_name}' needs at least 2 numeric values."
    return True, None


def validate_groups(metric_col, group_col, min_groups=2, max_groups=None):
    """Validate grouping variable has the right number of groups with data."""
    df = get_df()
    clean = df[[metric_col, group_col]].dropna()
    clean[metric_col] = pd.to_numeric(clean[metric_col], errors="coerce")
    clean = clean.dropna()

    if len(clean) < 2:
        return False, f"Insufficient data after numeric coercion in '{metric_col}' and '{group_col}'."

    groups = clean[group_col].unique()
    n_groups = len(groups)

    if n_groups < min_groups:
        return False, f"'{group_col}' needs at least {min_groups} groups, found {n_groups}."

    if max_groups and n_groups > max_groups:
        return False, f"'{group_col}' has {n_groups} groups, maximum is {max_groups}."

    # Check each group has enough data
    for g in groups:
        n = len(clean[clean[group_col] == g])
        if n < 2:
            return False, f"Group '{g}' in '{group_col}' has fewer than 2 observations."

    return True, None


def validate_two_metrics(col1, col2):
    """Validate two metric columns have paired data."""
    df = get_df()
    clean = df[[col1, col2]].copy()
    clean[col1] = pd.to_numeric(clean[col1], errors="coerce")
    clean[col2] = pd.to_numeric(clean[col2], errors="coerce")
    clean = clean.dropna()

    if len(clean) < 3:
        return False, f"Need at least 3 paired observations, found {len(clean)}."
    return True, None


def validate_binary(col_name):
    """Validate that a column has exactly 2 unique non-null values."""
    df = get_df()
    values = df[col_name].dropna().unique()
    if len(values) != 2:
        return False, f"'{col_name}' must have exactly 2 categories, found {len(values)}."
    return True, None


def validate_nominal(col_name, min_categories=2):
    """Validate a nominal variable."""
    df = get_df()
    values = df[col_name].dropna().unique()
    if len(values) < min_categories:
        return False, f"'{col_name}' needs at least {min_categories} categories, found {len(values)}."
    return True, None


def validate_survival_inputs(time_col, event_col):
    """Validate survival analysis time and event columns."""
    df = get_df()

    # Time: numeric and non-negative
    time_vals = pd.to_numeric(df[time_col], errors="coerce").dropna()
    if len(time_vals) < 2:
        return False, f"'{time_col}' needs at least 2 numeric values."
    if (time_vals < 0).any():
        return False, f"'{time_col}' contains negative values. Time must be non-negative."

    # Event: binary 0/1 only
    event_vals = pd.to_numeric(df[event_col], errors="coerce").dropna()
    unique_events = set(event_vals.unique())
    if not unique_events.issubset({0, 1, 0.0, 1.0}):
        return False, f"'{event_col}' must contain only 0 (censored) and 1 (event) values."
    if 1 not in unique_events and 1.0 not in unique_events:
        return False, f"'{event_col}' has no events (no 1 values). Need at least one event."

    # Check at least 2 complete pairs
    clean = df[[time_col, event_col]].copy()
    clean[time_col] = pd.to_numeric(clean[time_col], errors="coerce")
    clean[event_col] = pd.to_numeric(clean[event_col], errors="coerce")
    clean = clean.dropna()
    if len(clean) < 2:
        return False, "Need at least 2 complete time-event pairs."

    return True, None


def get_clean_data(columns):
    """Get a DataFrame with only the specified columns, dropping all-NA rows."""
    df = get_df()
    subset = df[columns].copy()
    return subset.dropna()


def coerce_numeric(df, columns):
    """Coerce specified columns to numeric."""
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    result = df.dropna(subset=columns)
    if len(result) < 2:
        raise ValueError(f"Insufficient data after numeric coercion: {len(result)} rows remain (minimum 2 required).")
    return result

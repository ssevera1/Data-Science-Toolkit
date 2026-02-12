"""Descriptive statistics: mean, median, SD, quartiles, skewness, kurtosis."""

import pandas as pd
import numpy as np
from scipy import stats


def compute_descriptives(series):
    """Compute comprehensive descriptive statistics for a numeric series."""
    data = pd.to_numeric(series, errors="coerce").dropna()

    if len(data) == 0:
        return {}

    result = {
        "N": int(len(data)),
        "Missing": int(series.isna().sum() + (pd.to_numeric(series, errors="coerce").isna().sum() - series.isna().sum())),
        "Mean": data.mean(),
        "Median": data.median(),
        "Mode": data.mode().iloc[0] if len(data.mode()) > 0 else np.nan,
        "Std. Deviation": data.std(ddof=1),
        "Variance": data.var(ddof=1),
        "Std. Error": data.std(ddof=1) / np.sqrt(len(data)),
        "Min": data.min(),
        "Max": data.max(),
        "Range": data.max() - data.min(),
        "Q1 (25th)": data.quantile(0.25),
        "Q3 (75th)": data.quantile(0.75),
        "IQR": data.quantile(0.75) - data.quantile(0.25),
        "Skewness": stats.skew(data, bias=False),
        "Kurtosis": stats.kurtosis(data, bias=False),
    }

    return result


def compute_descriptives_table(df, columns):
    """Compute descriptive stats for multiple columns, return as DataFrame."""
    records = []
    for col in columns:
        desc = compute_descriptives(df[col])
        desc["Variable"] = col
        records.append(desc)

    result_df = pd.DataFrame(records)
    # Reorder columns
    col_order = ["Variable", "N", "Missing", "Mean", "Median", "Mode",
                  "Std. Deviation", "Variance", "Std. Error",
                  "Min", "Max", "Range", "Q1 (25th)", "Q3 (75th)", "IQR",
                  "Skewness", "Kurtosis"]
    result_df = result_df[[c for c in col_order if c in result_df.columns]]
    return result_df.set_index("Variable")


def compute_frequency_table(series):
    """Compute frequency table for a categorical variable."""
    counts = series.value_counts().reset_index()
    counts.columns = ["Value", "Frequency"]
    counts["Percentage"] = (counts["Frequency"] / counts["Frequency"].sum() * 100)
    counts["Cumulative %"] = counts["Percentage"].cumsum()
    return counts

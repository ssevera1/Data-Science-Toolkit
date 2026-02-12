"""Correlation: Pearson, Spearman."""

import numpy as np
import pandas as pd
from scipy import stats
from stats.assumptions import shapiro_wilk


def pearson_correlation(x, y, alpha=0.05):
    """Pearson product-moment correlation."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")

    combined = pd.DataFrame({"x": x.values, "y": y.values}).dropna()
    x_clean = combined["x"].values
    y_clean = combined["y"].values

    r, p_value = stats.pearsonr(x_clean, y_clean)
    n = len(x_clean)
    df = n - 2

    # Confidence interval for r using Fisher's z transformation
    z_r = np.arctanh(r)
    se_z = 1 / np.sqrt(n - 3) if n > 3 else np.inf
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = np.tanh(z_r - z_crit * se_z)
    ci_upper = np.tanh(z_r + z_crit * se_z)

    # R-squared
    r_squared = r ** 2

    # Assumptions: normality of both variables
    norm_x = shapiro_wilk(x_clean, alpha)
    norm_y = shapiro_wilk(y_clean, alpha)

    return {
        "test": "Pearson Correlation",
        "r": r,
        "p": p_value,
        "df": df,
        "n": n,
        "r_squared": r_squared,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "assumptions": {
            "normality_x": norm_x,
            "normality_y": norm_y,
        },
    }


def spearman_correlation(x, y, alpha=0.05):
    """Spearman rank-order correlation."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")

    combined = pd.DataFrame({"x": x.values, "y": y.values}).dropna()
    x_clean = combined["x"].values
    y_clean = combined["y"].values

    rho, p_value = stats.spearmanr(x_clean, y_clean)
    n = len(x_clean)

    # CI using Fisher's z (approximation)
    z_r = np.arctanh(rho)
    se_z = 1 / np.sqrt(n - 3) if n > 3 else np.inf
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = np.tanh(z_r - z_crit * se_z)
    ci_upper = np.tanh(z_r + z_crit * se_z)

    return {
        "test": "Spearman Correlation",
        "rho": rho,
        "p": p_value,
        "n": n,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

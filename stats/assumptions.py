"""Assumption tests: Shapiro-Wilk, Levene's, Mauchly's sphericity."""

import numpy as np
import pandas as pd
from scipy import stats
from core.constants import ALPHA


def shapiro_wilk(data, alpha=ALPHA):
    """Shapiro-Wilk test for normality."""
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]

    if len(data) < 3:
        return {"statistic": None, "p_value": None, "passed": None, "detail": "Need at least 3 observations."}

    # Shapiro-Wilk has a limit of 5000
    if len(data) > 5000:
        rng = np.random.RandomState(42)
        data = rng.choice(data, 5000, replace=False)

    stat, p = stats.shapiro(data)
    return {
        "statistic": stat,
        "p_value": p,
        "passed": p >= alpha,
        "detail": "Data appears normally distributed." if p >= alpha else "Normality assumption violated.",
    }


def levene_test(groups, alpha=ALPHA):
    """Levene's test for homogeneity of variances."""
    clean_groups = [np.array(g, dtype=float)[~np.isnan(np.array(g, dtype=float))] for g in groups]
    clean_groups = [g for g in clean_groups if len(g) >= 2]

    if len(clean_groups) < 2:
        return {"statistic": None, "p_value": None, "passed": None, "detail": "Need at least 2 groups."}

    stat, p = stats.levene(*clean_groups)
    return {
        "statistic": stat,
        "p_value": p,
        "passed": p >= alpha,
        "detail": "Variances are homogeneous." if p >= alpha else "Homogeneity of variances violated.",
    }


def mauchly_sphericity(data_wide, alpha=ALPHA):
    """Mauchly's test for sphericity (for repeated measures).

    data_wide: DataFrame where each column is a condition/timepoint.
    """
    try:
        import pingouin as pg
        # pingouin's sphericity test expects wide-format data
        spher, W, chi2, dof, p = pg.sphericity(data_wide)
        return {
            "statistic": W,
            "chi2": chi2,
            "df": dof,
            "p_value": p,
            "passed": p >= alpha,
            "detail": "Sphericity assumption met." if p >= alpha else "Sphericity assumption violated. Consider Greenhouse-Geisser correction.",
        }
    except Exception as e:
        return {
            "statistic": None,
            "p_value": None,
            "passed": None,
            "detail": f"Could not compute sphericity: {str(e)}",
        }


def normality_per_group(df, value_col, group_col, alpha=ALPHA):
    """Run Shapiro-Wilk for each group."""
    results = {}
    for name, group in df.groupby(group_col):
        data = pd.to_numeric(group[value_col], errors="coerce").dropna().values
        results[str(name)] = shapiro_wilk(data, alpha)
    return results

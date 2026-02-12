"""Chi-squared test of independence."""

import numpy as np
import pandas as pd
from scipy import stats
from stats.effect_size import cramers_v


def chi_squared_test(df, var1, var2, alpha=0.05):
    """Chi-squared test of independence."""
    clean = df[[var1, var2]].dropna()

    # Create contingency table
    contingency = pd.crosstab(clean[var1], clean[var2])

    # Chi-squared test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Effect size: Cramer's V
    n = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    v = cramers_v(chi2, n, min(contingency.shape))

    # Expected frequencies table
    expected_df = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns,
    )

    # Check assumption: expected frequencies >= 5
    cells_below_5 = (expected < 5).sum()
    total_cells = expected.size
    pct_below_5 = cells_below_5 / total_cells * 100

    assumption_met = cells_below_5 == 0

    return {
        "test": "Chi-Squared Test of Independence",
        "chi2": chi2,
        "df": dof,
        "p": p_value,
        "cramers_v": v,
        "n": n,
        "contingency": contingency,
        "expected": expected_df,
        "assumptions": {
            "expected_frequencies": {
                "passed": assumption_met,
                "cells_below_5": int(cells_below_5),
                "total_cells": int(total_cells),
                "pct_below_5": pct_below_5,
                "detail": "All expected frequencies ≥ 5." if assumption_met
                    else f"{cells_below_5}/{total_cells} cells ({pct_below_5:.1f}%) have expected frequency < 5.",
            }
        },
    }

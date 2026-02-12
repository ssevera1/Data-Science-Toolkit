"""Binomial test."""

import numpy as np
import pandas as pd
from scipy import stats


def binomial_test(series, test_prop=0.5, alpha=0.05):
    """Binomial test for a binary variable against a hypothesized proportion."""
    data = series.dropna()
    values = data.unique()

    if len(values) != 2:
        return {
            "test": "Binomial Test",
            "error": f"Variable must have exactly 2 categories, found {len(values)}.",
        }

    # Count successes (first category alphabetically as "success")
    categories = sorted(values, key=str)
    success_label = categories[0]
    failure_label = categories[1]

    n_success = (data == success_label).sum()
    n_total = len(data)
    observed_prop = n_success / n_total

    # Binomial test
    result = stats.binomtest(n_success, n_total, test_prop, alternative="two-sided")
    p_value = result.pvalue
    ci = result.proportion_ci(confidence_level=1 - alpha)

    return {
        "test": "Binomial Test",
        "n": n_total,
        "n_success": int(n_success),
        "n_failure": int(n_total - n_success),
        "success_label": str(success_label),
        "failure_label": str(failure_label),
        "observed_prop": observed_prop,
        "test_prop": test_prop,
        "p": p_value,
        "ci_lower": ci.low,
        "ci_upper": ci.high,
    }

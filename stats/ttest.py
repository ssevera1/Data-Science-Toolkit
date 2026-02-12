"""T-test implementations: one-sample, independent, paired."""

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from stats.effect_size import cohens_d_one_sample, cohens_d_independent, cohens_d_paired
from stats.assumptions import shapiro_wilk, levene_test


def one_sample_ttest(data, mu=0, alpha=0.05):
    """One-sample t-test."""
    data = pd.to_numeric(pd.Series(data), errors="coerce").dropna().values

    # Test
    t_stat, p_value = stats.ttest_1samp(data, mu)
    df = len(data) - 1

    # Effect size
    d = cohens_d_one_sample(data, mu)

    # CI for mean
    se = stats.sem(data)
    ci = stats.t.interval(1 - alpha, df, loc=data.mean(), scale=se)

    # Assumptions
    normality = shapiro_wilk(data, alpha)

    return {
        "test": "One-Sample t-Test",
        "t": t_stat,
        "df": df,
        "p": p_value,
        "mean": data.mean(),
        "test_value": mu,
        "mean_diff": data.mean() - mu,
        "se": se,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "cohens_d": d,
        "n": len(data),
        "assumptions": {"normality": normality},
    }


def independent_ttest(group1, group2, equal_var=True, alpha=0.05):
    """Independent samples t-test."""
    g1 = pd.to_numeric(pd.Series(group1), errors="coerce").dropna().values
    g2 = pd.to_numeric(pd.Series(group2), errors="coerce").dropna().values

    # Test
    t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=equal_var)

    if equal_var:
        df = len(g1) + len(g2) - 2
    else:
        # Welch's df
        s1, s2 = g1.var(ddof=1), g2.var(ddof=1)
        n1, n2 = len(g1), len(g2)
        num = (s1/n1 + s2/n2)**2
        den = (s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1)
        df = num / den if den > 0 else n1 + n2 - 2

    # Effect size
    d = cohens_d_independent(g1, g2)

    # Mean difference CI
    mean_diff = g1.mean() - g2.mean()
    se_diff = np.sqrt(g1.var(ddof=1)/len(g1) + g2.var(ddof=1)/len(g2))
    ci = stats.t.interval(1 - alpha, df, loc=mean_diff, scale=se_diff)

    # Assumptions
    norm1 = shapiro_wilk(g1, alpha)
    norm2 = shapiro_wilk(g2, alpha)
    homogeneity = levene_test([g1, g2], alpha)

    return {
        "test": "Independent Samples t-Test" if equal_var else "Welch's t-Test",
        "t": t_stat,
        "df": df,
        "p": p_value,
        "mean1": g1.mean(),
        "mean2": g2.mean(),
        "mean_diff": mean_diff,
        "se_diff": se_diff,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "cohens_d": d,
        "n1": len(g1),
        "n2": len(g2),
        "sd1": g1.std(ddof=1),
        "sd2": g2.std(ddof=1),
        "equal_var": equal_var,
        "assumptions": {
            "normality_group1": norm1,
            "normality_group2": norm2,
            "homogeneity": homogeneity,
        },
    }


def paired_ttest(data1, data2, alpha=0.05):
    """Paired samples t-test."""
    d1 = pd.to_numeric(pd.Series(data1), errors="coerce")
    d2 = pd.to_numeric(pd.Series(data2), errors="coerce")

    # Align and drop NAs
    combined = pd.DataFrame({"a": d1.values, "b": d2.values}).dropna()
    a = combined["a"].values
    b = combined["b"].values
    diff = a - b

    # Test
    t_stat, p_value = stats.ttest_rel(a, b)
    df = len(diff) - 1

    # Effect size
    d = cohens_d_paired(diff)

    # CI for mean difference
    se = stats.sem(diff)
    ci = stats.t.interval(1 - alpha, df, loc=diff.mean(), scale=se)

    # Assumptions
    normality = shapiro_wilk(diff, alpha)

    return {
        "test": "Paired Samples t-Test",
        "t": t_stat,
        "df": df,
        "p": p_value,
        "mean1": a.mean(),
        "mean2": b.mean(),
        "mean_diff": diff.mean(),
        "se_diff": se,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "cohens_d": d,
        "n": len(diff),
        "sd_diff": diff.std(ddof=1),
        "assumptions": {"normality_of_differences": normality},
    }

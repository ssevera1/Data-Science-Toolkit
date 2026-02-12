"""Effect size calculations: Cohen's d, eta-squared, Cramer's V, rank-biserial."""

import numpy as np
from scipy import stats


def cohens_d_one_sample(data, mu):
    """Cohen's d for one-sample t-test."""
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    return (data.mean() - mu) / data.std(ddof=1)


def cohens_d_independent(group1, group2):
    """Cohen's d for independent samples t-test (pooled SD)."""
    g1 = np.array(group1, dtype=float)
    g2 = np.array(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)

    # Pooled standard deviation
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if sp == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / sp


def cohens_d_paired(diff):
    """Cohen's d for paired samples (using SD of differences)."""
    diff = np.array(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return diff.mean() / sd


def eta_squared(ss_between, ss_total):
    """Eta-squared effect size for ANOVA."""
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def partial_eta_squared(ss_effect, ss_error):
    """Partial eta-squared effect size."""
    if (ss_effect + ss_error) == 0:
        return 0.0
    return ss_effect / (ss_effect + ss_error)


def omega_squared(ss_between, ms_within, ss_total, df_between):
    """Omega-squared effect size for ANOVA (less biased than eta-squared)."""
    numerator = ss_between - df_between * ms_within
    denominator = ss_total + ms_within
    if denominator == 0:
        return 0.0
    return max(0, numerator / denominator)


def cramers_v(chi2, n, min_dim):
    """Cramer's V effect size for chi-squared test."""
    if n == 0 or min_dim <= 1:
        return 0.0
    return np.sqrt(chi2 / (n * (min_dim - 1)))


def rank_biserial(U, n1, n2):
    """Rank-biserial correlation for Mann-Whitney U."""
    if n1 * n2 == 0:
        return 0.0
    return 1 - (2 * U) / (n1 * n2)


def rank_biserial_wilcoxon(W_plus, n):
    """Rank-biserial correlation for Wilcoxon signed-rank test."""
    total = n * (n + 1) / 2
    if total == 0:
        return 0.0
    return (W_plus - (total / 2)) / (total / 2)


def r_squared(r):
    """R-squared from correlation coefficient."""
    return r ** 2

"""Non-parametric tests: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman."""

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from stats.effect_size import rank_biserial


def mann_whitney(group1, group2, alternative="two-sided", alpha=0.05):
    """Mann-Whitney U test."""
    g1 = pd.to_numeric(pd.Series(group1), errors="coerce").dropna().values
    g2 = pd.to_numeric(pd.Series(group2), errors="coerce").dropna().values

    U_stat, p_value = stats.mannwhitneyu(g1, g2, alternative=alternative)

    # Rank-biserial correlation
    r_rb = rank_biserial(U_stat, len(g1), len(g2))

    # Compute mean ranks
    n1, n2 = len(g1), len(g2)
    combined = np.concatenate([g1, g2])
    ranks = stats.rankdata(combined)
    mean_rank1 = ranks[:n1].mean()
    mean_rank2 = ranks[n1:].mean()

    return {
        "test": "Mann-Whitney U Test",
        "U": U_stat,
        "p": p_value,
        "rank_biserial": r_rb,
        "n1": n1,
        "n2": n2,
        "median1": np.median(g1),
        "median2": np.median(g2),
        "mean_rank1": mean_rank1,
        "mean_rank2": mean_rank2,
        "alternative": alternative,
    }


def wilcoxon_signed_rank(data1, data2, alternative="two-sided", alpha=0.05):
    """Wilcoxon signed-rank test."""
    d1 = pd.to_numeric(pd.Series(data1), errors="coerce")
    d2 = pd.to_numeric(pd.Series(data2), errors="coerce")

    combined = pd.DataFrame({"a": d1.values, "b": d2.values}).dropna()
    a = combined["a"].values
    b = combined["b"].values
    diff = a - b

    # Remove zero differences
    nonzero_mask = diff != 0
    diff_nonzero = diff[nonzero_mask]

    if len(diff_nonzero) < 1:
        return {
            "test": "Wilcoxon Signed-Rank Test",
            "W": None, "p": None,
            "detail": "All differences are zero.",
        }

    W_stat, p_value = stats.wilcoxon(a, b, alternative=alternative)

    # Effect size: r = Z / sqrt(N)
    n = len(diff_nonzero)
    # Compute Z directly from the test statistic (more accurate than back-calculating from p)
    rank_sum = n * (n + 1) / 2
    expected = rank_sum / 2
    var = n * (n + 1) * (2 * n + 1) / 24
    z_score = (W_stat - expected) / np.sqrt(var) if var > 0 else 0.0
    r_effect = abs(z_score) / np.sqrt(n) if n > 0 else 0

    return {
        "test": "Wilcoxon Signed-Rank Test",
        "W": W_stat,
        "p": p_value,
        "z": z_score,
        "r_effect": r_effect,
        "n": n,
        "n_total": len(diff),
        "median_diff": np.median(diff),
        "median1": np.median(a),
        "median2": np.median(b),
        "alternative": alternative,
    }


def kruskal_wallis(df, value_col, group_col, alpha=0.05):
    """Kruskal-Wallis H test."""
    clean = df[[value_col, group_col]].dropna()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna()

    groups = [g[value_col].values for _, g in clean.groupby(group_col)]
    group_names = list(clean[group_col].unique())

    H_stat, p_value = stats.kruskal(*groups)

    # Effect size: epsilon-squared = H / (n - 1)
    n = len(clean)
    k = len(groups)
    epsilon_sq = H_stat / (n - 1) if n > 1 else 0

    # Post-hoc: Dunn's test
    posthoc = None
    if p_value < alpha and len(group_names) > 2:
        try:
            posthoc = pg.pairwise_tests(
                data=clean, dv=value_col, between=group_col,
                parametric=False, padjust="bonf",
            )
        except Exception:
            pass

    # Group descriptives
    group_desc = clean.groupby(group_col)[value_col].agg(
        ["count", "median", "mean"]
    ).reset_index()
    group_desc.columns = ["Group", "N", "Median", "Mean"]

    return {
        "test": "Kruskal-Wallis H Test",
        "H": H_stat,
        "df": k - 1,
        "p": p_value,
        "epsilon_squared": epsilon_sq,
        "posthoc": posthoc,
        "group_desc": group_desc,
    }


def friedman_test(df, value_col, within_col, subject_col, alpha=0.05):
    """Friedman test for repeated measures."""
    clean = df[[value_col, within_col, subject_col]].dropna()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna()

    # Pivot to wide format
    wide = clean.pivot(index=subject_col, columns=within_col, values=value_col).dropna()

    if wide.shape[1] < 3:
        return {"test": "Friedman Test", "chi2": None, "p": None, "detail": "Need at least 3 conditions."}

    groups = [wide[col].values for col in wide.columns]
    chi2_stat, p_value = stats.friedmanchisquare(*groups)

    k = wide.shape[1]
    n = wide.shape[0]

    # Effect size: Kendall's W
    W = chi2_stat / (n * (k - 1)) if (n * (k - 1)) > 0 else 0

    # Group descriptives
    group_desc = clean.groupby(within_col)[value_col].agg(
        ["count", "median", "mean"]
    ).reset_index()
    group_desc.columns = ["Condition", "N", "Median", "Mean"]

    # Post-hoc
    posthoc = None
    if p_value < alpha and k > 2:
        try:
            posthoc = pg.pairwise_tests(
                data=clean, dv=value_col, within=within_col, subject=subject_col,
                parametric=False, padjust="bonf",
            )
        except Exception:
            pass

    return {
        "test": "Friedman Test",
        "chi2": chi2_stat,
        "df": k - 1,
        "p": p_value,
        "kendalls_w": W,
        "n_subjects": n,
        "n_conditions": k,
        "posthoc": posthoc,
        "group_desc": group_desc,
    }

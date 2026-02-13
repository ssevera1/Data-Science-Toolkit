"""MANOVA: Multivariate Analysis of Variance."""

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from statsmodels.multivariate.manova import MANOVA as SM_MANOVA
from stats.assumptions import levene_test


def manova(df, dv_cols, group_col, alpha=0.05):
    """Multivariate Analysis of Variance (MANOVA).

    Parameters
    ----------
    df : DataFrame
    dv_cols : list of str
        Two or more metric dependent variables.
    group_col : str
        Categorical grouping variable.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys: test, manova_table, univariate_anovas, group_desc,
                    assumptions, posthoc, n
    """
    cols = dv_cols + [group_col]
    clean = df[cols].dropna()
    for c in dv_cols:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna()

    n = len(clean)

    # Pre-compute grouped data once — reused for descriptives & Levene's
    grouped = clean.groupby(group_col)
    group_names = list(grouped.groups.keys())
    groups_per_dv = {
        dv: [g[dv].values for _, g in grouped] for dv in dv_cols
    }

    # ── MANOVA via statsmodels ────────────────────────────────────────────
    formula = " + ".join(dv_cols) + " ~ " + group_col
    mv = SM_MANOVA.from_formula(formula, data=clean)
    mv_result = mv.mv_test()

    # Parse the multivariate test table from mv_test()
    test_names = ["Wilks' lambda", "Pillai's trace",
                  "Hotelling-Lawley trace", "Roy's greatest root"]
    group_key = None
    for key in mv_result.results:
        if key.lower() != "intercept":
            group_key = key
            break

    manova_rows = []
    if group_key is not None:
        stat_table = mv_result.results[group_key]["stat"]
        for i, tname in enumerate(test_names):
            row = stat_table.iloc[i]
            manova_rows.append({
                "Test": tname,
                "Value": row["Value"],
                "Num DF": row["Num DF"],
                "Den DF": row["Den DF"],
                "F": row["F Value"],
                "p": row["Pr > F"],
            })

    manova_table = pd.DataFrame(manova_rows)

    # Overall p-value from Wilks' lambda for significance decision
    overall_p = manova_table["p"].iloc[0] if len(manova_table) > 0 else 1.0

    # ── Univariate follow-up ANOVAs ───────────────────────────────────────
    univariate_anovas = [
        {"dv": dv, "anova_table": pg.anova(data=clean, dv=dv, between=group_col, detailed=True)}
        for dv in dv_cols
    ]

    # ── Group descriptives per DV (single groupby, already computed) ──────
    desc_rows = []
    for dv in dv_cols:
        gd = grouped[dv].agg(["count", "mean", "std"]).reset_index()
        gd.columns = ["Group", "N", "Mean", "SD"]
        gd.insert(0, "DV", dv)
        desc_rows.append(gd)
    group_desc = pd.concat(desc_rows, ignore_index=True)

    # ── Assumptions ───────────────────────────────────────────────────────
    # Multivariate normality (Henze-Zirkler)
    try:
        hz_stat, hz_p, hz_normal = pg.multivariate_normality(
            clean[dv_cols], alpha=alpha
        )
        mv_normality = {
            "statistic": float(hz_stat),
            "p_value": float(hz_p),
            "passed": bool(hz_normal),
            "detail": ("Multivariate normality holds."
                       if hz_normal
                       else "Multivariate normality violated."),
        }
    except Exception as e:
        mv_normality = {
            "statistic": None,
            "p_value": None,
            "passed": None,
            "detail": f"Could not compute: {e}",
        }

    # Box's M (homogeneity of covariance matrices)
    try:
        box_result = pg.box_m(clean, dv_cols, group_col)
        box_chi2 = float(box_result["Chi2"].iloc[0])
        box_p = float(box_result["pval"].iloc[0])
        box_m = {
            "statistic": box_chi2,
            "p_value": box_p,
            "passed": box_p >= alpha,
            "detail": ("Covariance matrices are homogeneous."
                       if box_p >= alpha
                       else "Homogeneity of covariance matrices violated."),
        }
    except Exception as e:
        box_m = {
            "statistic": None,
            "p_value": None,
            "passed": None,
            "detail": f"Could not compute: {e}",
        }

    # Levene's test per DV (reuse pre-computed groups)
    levene_results = {
        dv: levene_test(groups_per_dv[dv], alpha) for dv in dv_cols
    }

    assumptions = {
        "multivariate_normality": mv_normality,
        "box_m": box_m,
        "homogeneity": levene_results,
    }

    # ── Post-hoc (Tukey per DV, only if overall MANOVA significant) ──────
    posthoc = None
    if overall_p < alpha and len(group_names) > 2:
        posthoc_list = []
        for dv in dv_cols:
            ph = pg.pairwise_tukey(data=clean, dv=dv, between=group_col)
            ph.insert(0, "DV", dv)
            posthoc_list.append(ph)
        if posthoc_list:
            posthoc = pd.concat(posthoc_list, ignore_index=True)

    return {
        "test": "MANOVA",
        "manova_table": manova_table,
        "univariate_anovas": univariate_anovas,
        "group_desc": group_desc,
        "assumptions": assumptions,
        "posthoc": posthoc,
        "n": n,
        "overall_p": overall_p,
    }

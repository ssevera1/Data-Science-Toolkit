"""ANOVA implementations: one-way, two-way, repeated measures, mixed."""

import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
from stats.assumptions import shapiro_wilk, levene_test, normality_per_group


def oneway_anova(df, value_col, group_col, alpha=0.05):
    """One-way ANOVA."""
    clean = df[[value_col, group_col]].dropna()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna()

    groups = [g[value_col].values for name, g in clean.groupby(group_col)]
    group_names = list(clean[group_col].unique())

    # ANOVA
    F_stat, p_value = stats.f_oneway(*groups)

    # Detailed ANOVA table using pingouin
    aov = pg.anova(data=clean, dv=value_col, between=group_col, detailed=True)

    # Effect sizes
    ss_between = aov["SS"].iloc[0]
    ss_within = aov["SS"].iloc[1]
    ss_total = ss_between + ss_within
    eta2 = ss_between / ss_total
    df_between = aov["DF"].iloc[0]
    ms_within = aov["MS"].iloc[1]
    omega2 = max(0, (ss_between - df_between * ms_within) / (ss_total + ms_within))

    # Post-hoc (Tukey HSD)
    posthoc = None
    if p_value < alpha and len(group_names) > 2:
        posthoc = pg.pairwise_tukey(data=clean, dv=value_col, between=group_col)

    # Group descriptives
    group_desc = clean.groupby(group_col)[value_col].agg(["count", "mean", "std"]).reset_index()
    group_desc.columns = ["Group", "N", "Mean", "SD"]

    # Assumptions
    normality = normality_per_group(clean, value_col, group_col, alpha)
    homogeneity = levene_test(groups, alpha)

    return {
        "test": "One-Way ANOVA",
        "F": F_stat,
        "df_between": int(aov["DF"].iloc[0]),
        "df_within": int(aov["DF"].iloc[1]),
        "p": p_value,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ms_between": aov["MS"].iloc[0],
        "ms_within": ms_within,
        "eta_squared": eta2,
        "omega_squared": omega2,
        "anova_table": aov,
        "posthoc": posthoc,
        "group_desc": group_desc,
        "assumptions": {
            "normality": normality,
            "homogeneity": homogeneity,
        },
    }


def twoway_anova(df, value_col, factor1, factor2, alpha=0.05):
    """Two-way ANOVA."""
    clean = df[[value_col, factor1, factor2]].dropna()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna()

    aov = pg.anova(data=clean, dv=value_col, between=[factor1, factor2], detailed=True)

    # Group descriptives
    group_desc = clean.groupby([factor1, factor2])[value_col].agg(["count", "mean", "std"]).reset_index()
    group_desc.columns = [factor1, factor2, "N", "Mean", "SD"]

    # Assumptions
    groups = [g[value_col].values for _, g in clean.groupby([factor1, factor2])]
    homogeneity = levene_test(groups, alpha)

    return {
        "test": "Two-Way ANOVA",
        "anova_table": aov,
        "group_desc": group_desc,
        "assumptions": {"homogeneity": homogeneity},
    }


def repeated_measures_anova(df, value_col, within_col, subject_col, alpha=0.05):
    """Repeated measures ANOVA."""
    clean = df[[value_col, within_col, subject_col]].dropna()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna()

    aov = pg.rm_anova(
        data=clean,
        dv=value_col,
        within=within_col,
        subject=subject_col,
        detailed=True,
    )

    # Post-hoc
    posthoc = None
    conditions = clean[within_col].unique()
    if len(conditions) > 2:
        posthoc = pg.pairwise_tests(
            data=clean, dv=value_col, within=within_col, subject=subject_col,
            padjust="bonf",
        )

    # Group descriptives
    group_desc = clean.groupby(within_col)[value_col].agg(["count", "mean", "std"]).reset_index()
    group_desc.columns = ["Condition", "N", "Mean", "SD"]

    # Sphericity
    sphericity = None
    if len(conditions) > 2:
        try:
            spher, W, chi2, dof, p = pg.sphericity(
                data=clean, dv=value_col, within=within_col, subject=subject_col
            )
            sphericity = {
                "statistic": W, "chi2": chi2, "df": dof, "p_value": p,
                "passed": p >= alpha,
                "detail": "Sphericity met." if p >= alpha else "Sphericity violated.",
            }
        except Exception:
            pass

    return {
        "test": "Repeated Measures ANOVA",
        "anova_table": aov,
        "posthoc": posthoc,
        "group_desc": group_desc,
        "assumptions": {"sphericity": sphericity},
    }


def mixed_anova(df, value_col, within_col, between_col, subject_col, alpha=0.05):
    """Mixed ANOVA (one within, one between)."""
    clean = df[[value_col, within_col, between_col, subject_col]].dropna()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna()

    aov = pg.mixed_anova(
        data=clean,
        dv=value_col,
        within=within_col,
        between=between_col,
        subject=subject_col,
    )

    # Group descriptives
    group_desc = clean.groupby([between_col, within_col])[value_col].agg(
        ["count", "mean", "std"]
    ).reset_index()
    group_desc.columns = [between_col, within_col, "N", "Mean", "SD"]

    return {
        "test": "Mixed ANOVA",
        "anova_table": aov,
        "group_desc": group_desc,
        "assumptions": {},
    }

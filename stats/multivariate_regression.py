"""Multivariate Linear Regression: multiple DVs predicted by continuous IVs."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pingouin as pg
from statsmodels.multivariate.manova import MANOVA as SM_MANOVA
from stats.assumptions import shapiro_wilk


def multivariate_regression(df, dv_cols, predictors, alpha=0.05):
    """Multivariate linear regression.

    Predicts multiple dependent variables simultaneously from one or more
    continuous predictors.  Returns multivariate test statistics (Wilks' etc.)
    plus individual OLS models per DV.

    Parameters
    ----------
    df : DataFrame
    dv_cols : list of str
        Two or more metric dependent variables.
    predictors : list of str
        One or more metric predictor variables.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys: test, multivariate_tests, individual_models, n,
                    assumptions
    """
    cols = dv_cols + predictors
    clean = df[cols].dropna()
    for c in cols:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna()

    n = len(clean)

    # ── Multivariate tests via statsmodels MANOVA machinery ───────────────
    # Build formula: "y1 + y2 ~ x1 + x2"
    lhs = " + ".join(dv_cols)
    rhs = " + ".join(predictors)
    formula = f"{lhs} ~ {rhs}"

    multivariate_tests_list = []
    try:
        mv = SM_MANOVA.from_formula(formula, data=clean)
        mv_result = mv.mv_test()

        test_names = ["Wilks' lambda", "Pillai's trace",
                      "Hotelling-Lawley trace", "Roy's greatest root"]

        # Extract results for each predictor (skip Intercept)
        for key in mv_result.results:
            if key.lower() == "intercept":
                continue
            stat_table = mv_result.results[key]["stat"]
            for i, tname in enumerate(test_names):
                row = stat_table.iloc[i]
                multivariate_tests_list.append({
                    "Predictor": key,
                    "Test": tname,
                    "Value": row["Value"],
                    "Num DF": row["Num DF"],
                    "Den DF": row["Den DF"],
                    "F": row["F Value"],
                    "p": row["Pr > F"],
                })
    except Exception:
        pass

    multivariate_tests = pd.DataFrame(multivariate_tests_list)

    # ── Individual OLS models per DV ──────────────────────────────────────
    # Pre-compute design matrix and predictor SDs once (shared across DVs)
    X = clean[predictors].values
    X_const = sm.add_constant(X)
    sd_x = clean[predictors].std().values
    coef_names = ["(Intercept)"] + predictors

    individual_models = []
    residual_matrix = []

    for dv in dv_cols:
        y = clean[dv].values
        model = sm.OLS(y, X_const).fit()

        ci = model.conf_int(alpha)
        coef_table = pd.DataFrame({
            "Variable": coef_names,
            "B": model.params,
            "Std. Error": model.bse,
            "t": model.tvalues,
            "p": model.pvalues,
            "CI Lower": ci[:, 0],
            "CI Upper": ci[:, 1],
        })

        # Standardized coefficients (beta) — vectorized
        sd_y = y.std()
        if sd_y > 0:
            betas = np.empty(len(coef_names))
            betas[0] = np.nan
            betas[1:] = model.params[1:] * sd_x / sd_y
        else:
            betas = np.full(len(coef_names), np.nan)
            betas[0] = np.nan
        coef_table["Beta"] = betas

        residual_matrix.append(model.resid)

        individual_models.append({
            "dv": dv,
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
            "f_stat": model.fvalue,
            "f_p": model.f_pvalue,
            "df_model": int(model.df_model),
            "df_resid": int(model.df_resid),
            "aic": model.aic,
            "bic": model.bic,
            "coef_table": coef_table,
            "residuals": model.resid,
            "fitted": model.fittedvalues,
        })

    # ── Assumptions ───────────────────────────────────────────────────────
    # Multivariate normality on residuals (Henze-Zirkler)
    resid_df = pd.DataFrame(
        np.column_stack(residual_matrix), columns=dv_cols
    )
    try:
        hz_stat, hz_p, hz_normal = pg.multivariate_normality(
            resid_df, alpha=alpha
        )
        mv_normality = {
            "statistic": float(hz_stat),
            "p_value": float(hz_p),
            "passed": bool(hz_normal),
            "detail": ("Multivariate normality of residuals holds."
                       if hz_normal
                       else "Multivariate normality of residuals violated."),
        }
    except Exception as e:
        mv_normality = {
            "statistic": None,
            "p_value": None,
            "passed": None,
            "detail": f"Could not compute: {e}",
        }

    # Shapiro-Wilk per DV residuals
    residual_normality = {}
    for m in individual_models:
        residual_normality[m["dv"]] = shapiro_wilk(m["residuals"], alpha)

    assumptions = {
        "multivariate_normality": mv_normality,
        "residual_normality": residual_normality,
    }

    return {
        "test": "Multivariate Linear Regression",
        "multivariate_tests": multivariate_tests,
        "individual_models": individual_models,
        "n": n,
        "assumptions": assumptions,
    }

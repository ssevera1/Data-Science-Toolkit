"""Regression: Linear (OLS), Logistic (Logit)."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from stats.assumptions import shapiro_wilk


def linear_regression(df, dv, predictors, alpha=0.05):
    """Multiple linear regression (OLS)."""
    cols = [dv] + predictors
    clean = df[cols].dropna().copy()
    for c in cols:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna()

    y = clean[dv].values
    X = clean[predictors].values
    X_const = sm.add_constant(X)

    # Guard: detect zero-variance predictors that cause singular matrices
    zero_var = [p for p in predictors if clean[p].std() == 0]
    if zero_var:
        return {"test": "Linear Regression",
                "error": f"Zero-variance predictor(s): {', '.join(zero_var)}. Remove constant columns."}

    model = sm.OLS(y, X_const).fit()

    # Residual normality
    residual_normality = shapiro_wilk(model.resid, alpha)

    # Build coefficient table
    coef_names = ["(Intercept)"] + predictors
    ci = model.conf_int(alpha)
    coef_table = pd.DataFrame({
        "Variable": coef_names,
        "B": model.params[:len(coef_names)],
        "Std. Error": model.bse[:len(coef_names)],
        "t": model.tvalues[:len(coef_names)],
        "p": model.pvalues[:len(coef_names)],
        "CI Lower": ci[:len(coef_names), 0],
        "CI Upper": ci[:len(coef_names), 1],
    })

    # Standardized coefficients (beta)
    if len(predictors) > 0:
        sd_y = y.std()
        sd_x = clean[predictors].std().values
        betas = [np.nan]  # intercept has no standardized beta
        for i, p in enumerate(predictors):
            betas.append(model.params[i + 1] * sd_x[i] / sd_y if sd_y > 0 else 0)
        coef_table["Beta"] = betas

    return {
        "test": "Linear Regression",
        "model": model,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_p": model.f_pvalue,
        "df_model": int(model.df_model),
        "df_resid": int(model.df_resid),
        "n": int(model.nobs),
        "aic": model.aic,
        "bic": model.bic,
        "coef_table": coef_table,
        "residuals": model.resid,
        "fitted": model.fittedvalues,
        "assumptions": {"residual_normality": residual_normality},
    }


def logistic_regression(df, dv, predictors, alpha=0.05):
    """Logistic regression (Logit)."""
    cols = [dv] + predictors
    clean = df[cols].dropna().copy()
    for c in cols:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna()

    y = clean[dv].values
    X = clean[predictors].values
    X_const = sm.add_constant(X)

    # Ensure binary DV
    unique_vals = np.unique(y)
    if len(unique_vals) != 2:
        return {"test": "Logistic Regression", "error": f"DV must be binary, found {len(unique_vals)} unique values."}

    model = sm.Logit(y, X_const).fit(disp=0)

    # Coefficient table — clip params before exp() to avoid overflow
    coef_names = ["(Intercept)"] + predictors
    params_clipped = np.clip(model.params, -700, 700)
    ci = model.conf_int(alpha)
    ci_clipped = np.clip(ci, -700, 700)
    coef_table = pd.DataFrame({
        "Variable": coef_names,
        "B": model.params,
        "Std. Error": model.bse,
        "z": model.tvalues,
        "p": model.pvalues,
        "OR": np.exp(params_clipped),
        "CI Lower (OR)": np.exp(ci_clipped[:, 0]),
        "CI Upper (OR)": np.exp(ci_clipped[:, 1]),
    })

    # Pseudo R-squared (McFadden)
    pseudo_r2 = model.prsquared

    # Classification accuracy
    predicted_probs = model.predict(X_const)
    predicted_class = (predicted_probs >= 0.5).astype(int)
    accuracy = (predicted_class == y).mean()

    return {
        "test": "Logistic Regression",
        "model": model,
        "pseudo_r_squared": pseudo_r2,
        "log_likelihood": model.llf,
        "aic": model.aic,
        "bic": model.bic,
        "n": int(model.nobs),
        "coef_table": coef_table,
        "accuracy": accuracy,
        "predicted_probs": predicted_probs,
        "chi2": model.llr,
        "chi2_p": model.llr_pvalue,
    }

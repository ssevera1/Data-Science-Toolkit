"""Survival analysis: Kaplan-Meier, Log-Rank, Cox PH, Extended Cox."""

import numpy as np
import pandas as pd


def _survival_at_times(sf, times):
    """Look up survival probabilities at arbitrary times from a KM curve.

    Uses the step-function convention: S(t) is the last known value at or
    before *t*.  Safe for times outside the fitted range — returns 1.0
    before the first event and the last value after the final event.
    """
    timeline = sf.index.values
    values = sf.iloc[:, 0].values
    indices = np.searchsorted(timeline, times, side="right") - 1
    return [
        float(values[max(i, 0)]) if i >= 0 else 1.0
        for i in indices
    ]


def kaplan_meier(df, time_col, event_col, group_col=None, alpha=0.05):
    """Kaplan-Meier estimation with optional log-rank test.

    Parameters
    ----------
    df : DataFrame
    time_col : str — time-to-event column (numeric, non-negative)
    event_col : str — event indicator (0=censored, 1=event)
    group_col : str, optional — grouping variable for stratified analysis
    alpha : float — significance level

    Returns
    -------
    dict with standard result keys
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test

    cols = [time_col, event_col] + ([group_col] if group_col else [])
    clean = df[cols].copy()
    clean[time_col] = pd.to_numeric(clean[time_col], errors="coerce")
    clean[event_col] = pd.to_numeric(clean[event_col], errors="coerce")
    clean = clean.dropna()

    T = clean[time_col].values
    E = clean[event_col].astype(int).values
    n = len(T)

    if n < 2:
        return {
            "test_name": "Kaplan-Meier Estimate",
            "error": f"Need at least 2 observations, found {n}.",
        }

    n_events = int(E.sum())
    n_censored = n - n_events

    result = {
        "test_name": "Kaplan-Meier Estimate",
        "n": n,
        "n_events": n_events,
        "n_censored": n_censored,
        "assumptions": {},
    }

    if n_events == 0:
        result["warning"] = (
            "No events observed — all observations are censored. "
            "Median survival cannot be estimated."
        )

    if group_col is None:
        # ── Single KM curve ─────────────────────────────────────────
        kmf = KaplanMeierFitter()
        kmf.fit(T, E, alpha=alpha)

        median = kmf.median_survival_time_
        result["median_survival"] = float(median) if np.isfinite(median) else None
        try:
            ci = kmf.confidence_interval_median_survival_time_
            result["median_ci_lower"] = (
                float(ci.iloc[0, 0]) if np.isfinite(ci.iloc[0, 0]) else None
            )
            result["median_ci_upper"] = (
                float(ci.iloc[0, 1]) if np.isfinite(ci.iloc[0, 1]) else None
            )
        except Exception:
            result["median_ci_lower"] = None
            result["median_ci_upper"] = None

        sf = kmf.survival_function_
        ci_df = kmf.confidence_interval_

        curve = {
            "label": "Overall",
            "timeline": sf.index.tolist(),
            "survival": sf.iloc[:, 0].tolist(),
            "ci_lower": ci_df.iloc[:, 0].tolist(),
            "ci_upper": ci_df.iloc[:, 1].tolist(),
        }

        # Censoring tick marks
        censor_mask = E == 0
        if censor_mask.any():
            ct = T[censor_mask]
            cs = _survival_at_times(sf, ct)
            curve["censor_times"] = ct.tolist()
            curve["censor_survivals"] = cs
        else:
            curve["censor_times"] = []
            curve["censor_survivals"] = []

        result["curves"] = [curve]

    else:
        # ── Grouped KM curves ──────────────────────────────────────
        groups = sorted(clean[group_col].unique(), key=str)
        n_groups = len(groups)

        if n_groups < 2:
            return kaplan_meier(df, time_col, event_col, group_col=None, alpha=alpha)

        curves = []
        group_summary = []

        for g in groups:
            mask = clean[group_col] == g
            T_g = clean.loc[mask, time_col].values
            E_g = clean.loc[mask, event_col].astype(int).values

            kmf = KaplanMeierFitter()
            kmf.fit(T_g, E_g, alpha=alpha, label=str(g))

            median = kmf.median_survival_time_
            sf = kmf.survival_function_
            ci_df = kmf.confidence_interval_

            curve = {
                "label": str(g),
                "timeline": sf.index.tolist(),
                "survival": sf.iloc[:, 0].tolist(),
                "ci_lower": ci_df.iloc[:, 0].tolist(),
                "ci_upper": ci_df.iloc[:, 1].tolist(),
            }

            censor_mask = E_g == 0
            if censor_mask.any():
                ct = T_g[censor_mask]
                cs = _survival_at_times(sf, ct)
                curve["censor_times"] = ct.tolist()
                curve["censor_survivals"] = cs
            else:
                curve["censor_times"] = []
                curve["censor_survivals"] = []

            curves.append(curve)

            group_summary.append({
                "Group": str(g),
                "N": int(len(T_g)),
                "Events": int(E_g.sum()),
                "Censored": int(len(T_g) - E_g.sum()),
                "Median Survival": (
                    round(float(median), 4) if np.isfinite(median) else None
                ),
            })

        result["curves"] = curves
        result["group_summary"] = group_summary
        result["n_groups"] = n_groups

        if n_groups > 8:
            result["warning"] = (
                f"{n_groups} groups detected — the survival plot may be cluttered."
            )

        # Log-rank test
        if n_groups == 2:
            g1_mask = clean[group_col] == groups[0]
            lr = logrank_test(
                clean.loc[g1_mask, time_col],
                clean.loc[~g1_mask, time_col],
                clean.loc[g1_mask, event_col],
                clean.loc[~g1_mask, event_col],
                alpha=alpha,
            )
            result["test_name"] = "Kaplan-Meier with Log-Rank Test"
            result["logrank_statistic"] = float(lr.test_statistic)
            result["logrank_p"] = float(lr.p_value)
            result["logrank_df"] = 1

            # Hazard ratio via Cox for 2-group comparison
            from lifelines import CoxPHFitter

            _cox_df = clean[[time_col, event_col, group_col]].copy()
            _cox_df[group_col] = (_cox_df[group_col] == groups[1]).astype(int)
            try:
                _cph = CoxPHFitter()
                _cph.fit(_cox_df, duration_col=time_col, event_col=event_col)
                hr = float(np.exp(_cph.params_.iloc[0]))
                hr_ci = np.exp(_cph.confidence_intervals_.values[0])
                result["hazard_ratio"] = hr
                result["hr_ci_lower"] = float(hr_ci[0])
                result["hr_ci_upper"] = float(hr_ci[1])
                result["hr_reference"] = str(groups[0])
            except Exception:
                pass
        else:
            lr = multivariate_logrank_test(
                clean[time_col],
                clean[group_col],
                clean[event_col],
            )
            result["test_name"] = "Kaplan-Meier with Log-Rank Test"
            result["logrank_statistic"] = float(lr.test_statistic)
            result["logrank_p"] = float(lr.p_value)
            result["logrank_df"] = int(n_groups - 1)

    return result


def cox_regression(df, time_col, event_col, predictors, nominal_preds=None,
                   alpha=0.05):
    """Cox Proportional Hazards regression.

    Parameters
    ----------
    df : DataFrame
    time_col : str
    event_col : str
    predictors : list of str
    nominal_preds : list of str, optional — predictors to dummy-encode
    alpha : float

    Returns
    -------
    dict with coefficient table, concordance, Schoenfeld test results
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    nominal_preds = set(nominal_preds or [])

    cols = [time_col, event_col] + list(predictors)
    clean = df[cols].copy()
    clean[time_col] = pd.to_numeric(clean[time_col], errors="coerce")
    clean[event_col] = pd.to_numeric(clean[event_col], errors="coerce")
    clean = clean.dropna()

    # Dummy-encode nominal predictors
    cols_to_encode = [p for p in predictors if p in nominal_preds]
    if cols_to_encode:
        clean = pd.get_dummies(
            clean, columns=cols_to_encode, drop_first=True, dtype=float,
        )

    # Coerce remaining predictors to numeric
    model_preds = [c for c in clean.columns if c not in (time_col, event_col)]
    for c in model_preds:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna()

    n = len(clean)
    n_events = int(clean[event_col].sum())

    if n < 5:
        return {
            "test_name": "Cox Proportional Hazards",
            "error": f"Need at least 5 observations, found {n}.",
        }
    if n_events < 2:
        return {
            "test_name": "Cox Proportional Hazards",
            "error": f"Need at least 2 events, found {n_events}.",
        }

    # Fit model
    try:
        cph = CoxPHFitter(alpha=alpha)
        cph.fit(clean, duration_col=time_col, event_col=event_col)
    except Exception as e:
        return {
            "test_name": "Cox Proportional Hazards",
            "error": f"Model fitting failed: {e}",
        }

    # Build coefficient table
    summary = cph.summary
    ci_pct = int((1 - alpha) * 100)
    coef_table = pd.DataFrame({
        "Covariate": summary.index.tolist(),
        "log(HR)": summary["coef"].values.round(4),
        "HR": summary["exp(coef)"].values.round(4),
        "SE": summary["se(coef)"].values.round(4),
        "z": summary["z"].values.round(4),
        "p": summary["p"].values.round(4),
        f"HR {ci_pct}% CI Lower": summary[
            f"exp(coef) lower {ci_pct}%"
        ].values.round(4),
        f"HR {ci_pct}% CI Upper": summary[
            f"exp(coef) upper {ci_pct}%"
        ].values.round(4),
    })

    result = {
        "test_name": "Cox Proportional Hazards",
        "n": n,
        "n_events": n_events,
        "concordance_index": float(cph.concordance_index_),
        "partial_aic": float(cph.AIC_partial_),
        "log_likelihood": float(cph.log_likelihood_),
        "coef_table": coef_table,
        "assumptions": {},
    }

    # ── Schoenfeld residuals test (PH / decay test) ─────────────────
    try:
        ph_test = proportional_hazard_test(cph, clean, time_transform="rank")
        ph_summary = ph_test.summary

        ph_results = {}
        for idx in ph_summary.index:
            cov = str(idx) if not isinstance(idx, tuple) else str(idx[0])
            stat = float(ph_summary.loc[idx, "test_statistic"])
            p_val = float(ph_summary.loc[idx, "p"])
            ph_results[cov] = {
                "statistic": stat,
                "p_value": p_val,
                "passed": p_val >= alpha,
                "detail": (
                    "PH assumption holds for this covariate."
                    if p_val >= alpha
                    else "PH assumption may be violated — consider "
                         "time-varying coefficients (Extended Cox Model)."
                ),
            }

        result["assumptions"]["proportional_hazards"] = ph_results

        # Overall PH test (minimum p across covariates)
        min_p = min(r["p_value"] for r in ph_results.values())
        result["ph_global_p"] = min_p

    except Exception:
        result["assumptions"]["proportional_hazards"] = {
            "overall": {
                "statistic": None,
                "p_value": None,
                "passed": True,
                "detail": "Could not compute PH test.",
            },
        }

    # ── Schoenfeld residuals for plotting ───────────────────────────
    try:
        schoenfeld_resids = cph.compute_residuals(clean, kind="schoenfeld")
        result["schoenfeld_data"] = {
            "time": schoenfeld_resids.index.tolist(),
            "covariates": {
                col: schoenfeld_resids[col].tolist()
                for col in schoenfeld_resids.columns
            },
        }
    except Exception:
        pass

    # ── Forest plot data ────────────────────────────────────────────
    ci_lo_col = f"HR {ci_pct}% CI Lower"
    ci_hi_col = f"HR {ci_pct}% CI Upper"
    result["forest_data"] = [
        {
            "covariate": row["Covariate"],
            "hr": float(row["HR"]),
            "ci_lower": float(row[ci_lo_col]),
            "ci_upper": float(row[ci_hi_col]),
            "p": float(row["p"]),
        }
        for _, row in coef_table.iterrows()
    ]

    return result


def extended_cox_model(df, time_col, event_col, predictors,
                       nominal_preds=None, alpha=0.05, stop_col=None,
                       penalizer=0.0):
    """Extended Cox model with time-varying coefficients.

    Adds covariate * log(time) interaction terms to detect and model
    non-proportional hazards (decay in covariate effects over time).

    Parameters
    ----------
    df : DataFrame
    time_col : str — duration column, or start time when stop_col is provided
    event_col : str
    predictors : list of str
    nominal_preds : list of str, optional — predictors to dummy-encode
    alpha : float
    stop_col : str, optional — stop time column; if provided, duration is
        computed as stop_col - time_col

    Returns
    -------
    dict with base + interaction coefficients, decay test results
    """
    from lifelines import CoxPHFitter
    from scipy.stats import chi2

    nominal_preds = set(nominal_preds or [])

    cols = [time_col, event_col] + list(predictors)
    if stop_col:
        cols = [time_col, stop_col, event_col] + list(predictors)
    clean = df[cols].copy()
    clean[time_col] = pd.to_numeric(clean[time_col], errors="coerce")
    if stop_col:
        clean[stop_col] = pd.to_numeric(clean[stop_col], errors="coerce")
    clean[event_col] = pd.to_numeric(clean[event_col], errors="coerce")
    clean = clean.dropna()

    # Compute duration from start/stop if provided
    if stop_col:
        clean["_duration"] = clean[stop_col] - clean[time_col]
        invalid = (clean["_duration"] <= 0).sum()
        if invalid > 0:
            clean = clean[clean["_duration"] > 0].copy()
        duration_col = "_duration"
    else:
        duration_col = time_col

    # Dummy-encode nominal predictors
    cols_to_encode = [p for p in predictors if p in nominal_preds]
    if cols_to_encode:
        clean = pd.get_dummies(
            clean, columns=cols_to_encode, drop_first=True, dtype=float,
        )

    model_preds = [
        c for c in clean.columns
        if c not in (time_col, stop_col, event_col, "_duration")
    ]
    for c in model_preds:
        clean[c] = pd.to_numeric(clean[c], errors="coerce")
    clean = clean.dropna()

    n = len(clean)
    n_events = int(clean[event_col].sum())

    if n < 10:
        return {
            "test_name": "Extended Cox Model",
            "error": f"Need at least 10 observations, found {n}.",
        }
    if n_events < 5:
        return {
            "test_name": "Extended Cox Model",
            "error": f"Need at least 5 events, found {n_events}.",
        }

    # ── Base Cox model (for comparison) ─────────────────────────────
    try:
        cph_base = CoxPHFitter(alpha=alpha, penalizer=penalizer)
        cph_base.fit(clean, duration_col=duration_col, event_col=event_col)
        base_aic = float(cph_base.AIC_partial_)
        base_ll = float(cph_base.log_likelihood_)
        base_cindex = float(cph_base.concordance_index_)
    except Exception as e:
        return {
            "test_name": "Extended Cox Model",
            "error": f"Base model fitting failed: {e}",
        }

    # ── Extended model: covariate * log(time) interactions ──────────
    extended = clean.copy()
    log_time = np.log(extended[duration_col].clip(lower=1e-10))

    interaction_cols = []
    for pred in model_preds:
        int_col = f"{pred}_x_log(t)"
        extended[int_col] = extended[pred] * log_time
        interaction_cols.append(int_col)

    try:
        cph_ext = CoxPHFitter(alpha=alpha, penalizer=penalizer)
        cph_ext.fit(extended, duration_col=duration_col, event_col=event_col)
    except Exception as e:
        return {
            "test_name": "Extended Cox Model",
            "error": f"Extended model fitting failed: {e}",
        }

    # Build coefficient table
    summary = cph_ext.summary
    ci_pct = int((1 - alpha) * 100)
    coef_table = pd.DataFrame({
        "Covariate": summary.index.tolist(),
        "log(HR)": summary["coef"].values.round(4),
        "HR": summary["exp(coef)"].values.round(4),
        "SE": summary["se(coef)"].values.round(4),
        "z": summary["z"].values.round(4),
        "p": summary["p"].values.round(4),
        f"HR {ci_pct}% CI Lower": summary[
            f"exp(coef) lower {ci_pct}%"
        ].values.round(4),
        f"HR {ci_pct}% CI Upper": summary[
            f"exp(coef) upper {ci_pct}%"
        ].values.round(4),
    })

    # Decay detection: significance of interaction terms
    decay_results = []
    for pred in model_preds:
        int_col = f"{pred}_x_log(t)"
        if int_col in summary.index:
            row = summary.loc[int_col]
            p_val = float(row["p"])
            coef_val = float(row["coef"])
            decay_results.append({
                "Covariate": pred,
                "Interaction Coef": round(coef_val, 4),
                "Interaction HR": round(float(row["exp(coef)"]), 4),
                "p-value": round(p_val, 4),
                "Significant": p_val < alpha,
                "Direction": (
                    "Effect increases over time" if coef_val > 0
                    else "Effect decays over time"
                ) if p_val < alpha else "No significant change",
            })

    # Likelihood ratio test: extended vs base
    lr_stat = max(0, 2 * (float(cph_ext.log_likelihood_) - base_ll))
    lr_df = len(interaction_cols)
    lr_p = float(chi2.sf(lr_stat, lr_df)) if lr_df > 0 else 1.0

    result = {
        "test_name": "Extended Cox Model (Time-Varying Coefficients)",
        "n": n,
        "n_events": n_events,
        "concordance_index": float(cph_ext.concordance_index_),
        "partial_aic": float(cph_ext.AIC_partial_),
        "log_likelihood": float(cph_ext.log_likelihood_),
        "base_aic": base_aic,
        "base_log_likelihood": base_ll,
        "base_concordance": base_cindex,
        "coef_table": coef_table,
        "decay_results": decay_results,
        "lr_test_statistic": round(lr_stat, 4),
        "lr_test_p": round(lr_p, 4),
        "lr_test_df": lr_df,
        "assumptions": {},
    }

    # Forest plot data
    ci_lo_col = f"HR {ci_pct}% CI Lower"
    ci_hi_col = f"HR {ci_pct}% CI Upper"
    result["forest_data"] = [
        {
            "covariate": row["Covariate"],
            "hr": float(row["HR"]),
            "ci_lower": float(row[ci_lo_col]),
            "ci_upper": float(row[ci_hi_col]),
            "p": float(row["p"]),
        }
        for _, row in coef_table.iterrows()
    ]

    return result

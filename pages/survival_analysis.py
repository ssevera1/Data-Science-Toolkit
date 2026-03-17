"""Survival Analysis page."""

import streamlit as st
import pandas as pd

from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import (
    render_significance_result, render_assumption_check, render_effect_size,
)
from stats.survival import kaplan_meier, cox_regression, extended_cox_model
from charts.survival_plot import kaplan_meier_plot, cox_forest_plot, schoenfeld_plot
from core.state import get_df, get_var_type, log_result
from core.validators import validate_survival_inputs
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df

_CACHE_KEY = "_result_survival"


def render():
    st.title("Survival Analysis")
    st.markdown(
        "Time-to-event analysis: Kaplan-Meier estimation, log-rank test, "
        "Cox Proportional Hazards regression, and extended models with "
        "time-varying coefficients."
    )

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    # ── Analysis mode ─────────────────────────────────────────────────
    mode = st.radio(
        "Analysis mode",
        [
            "Kaplan-Meier / Log-Rank",
            "Cox Proportional Hazards",
            "Extended Cox Model",
        ],
        horizontal=True,
        key="surv_mode",
    )

    # ── Variable selection (mode-dependent) ──────────────────────────
    stop_var = None
    if mode == "Extended Cox Model":
        col1, col2, col3 = st.columns(3)
        with col1:
            time_var = select_metric_variable(
                "Start time", key="surv_time",
            )
        with col2:
            stop_var = select_metric_variable(
                "Stop time", key="surv_stop",
            )
        with col3:
            event_var = select_nominal_variable(
                "Event indicator (0=censored, 1=event)", key="surv_event",
            )
    else:
        col1, col2 = st.columns(2)
        with col1:
            time_var = select_metric_variable(
                "Time variable (duration)", key="surv_time",
            )
        with col2:
            event_var = select_nominal_variable(
                "Event indicator (0=censored, 1=event)", key="surv_event",
            )

    # ── Mode-specific inputs ──────────────────────────────────────────
    group_var = None
    predictors = None
    nominal_preds = None

    if mode == "Kaplan-Meier / Log-Rank":
        group_var = select_nominal_variable(
            "Grouping variable (optional)", key="surv_group",
        )
    else:
        # Cox PH or Extended Cox: predictor selection
        exclude = {time_var, stop_var, event_var} - {None}
        available = [c for c in df.columns if c not in exclude]
        predictors = st.multiselect(
            "Predictor variables",
            options=available,
            format_func=lambda x: f"{x} ({get_var_type(x)})",
            key="surv_predictors",
        )
        if predictors:
            nominal_preds = [
                p for p in predictors
                if get_var_type(p) in ("Nominal", "Ordinal")
            ]

    # ── Options ───────────────────────────────────────────────────────
    with st.expander("Options"):
        alpha = st.slider(
            "Significance level (\u03b1)", 0.01, 0.10, 0.05, 0.01,
            key="surv_alpha",
        )
        penalizer = 0.0
        if mode == "Extended Cox Model":
            penalizer = st.number_input(
                "Penalizer (L2 regularization)",
                min_value=0.0, max_value=10.0, value=0.01, step=0.01,
                key="surv_penalizer",
                help=(
                    "Adds L2 regularization to stabilize fitting on sparse "
                    "episodic data. Set to 0.01–0.1 if the model fails with "
                    "a singular matrix error."
                ),
            )

    # ── Calculate ─────────────────────────────────────────────────────
    can_calculate = bool(time_var and event_var)
    if mode == "Extended Cox Model":
        can_calculate = can_calculate and bool(stop_var) and bool(predictors)
    elif mode != "Kaplan-Meier / Log-Rank":
        can_calculate = can_calculate and bool(predictors)

    if can_calculate and st.button("Calculate", type="primary"):
        valid, msg = validate_survival_inputs(time_var, event_var)
        if not valid:
            st.error(msg)
            return

        if mode == "Kaplan-Meier / Log-Rank":
            result = kaplan_meier(
                df, time_var, event_var, group_col=group_var, alpha=alpha,
            )
        elif mode == "Cox Proportional Hazards":
            if time_var in predictors or event_var in predictors:
                st.error("Time and event variables cannot be predictors.")
                return
            result = cox_regression(
                df, time_var, event_var, predictors,
                nominal_preds=nominal_preds, alpha=alpha,
            )
        else:  # Extended Cox Model
            bad = {time_var, stop_var, event_var} - {None}
            if bad & set(predictors):
                st.error("Time and event variables cannot be predictors.")
                return
            result = extended_cox_model(
                df, time_var, event_var, predictors,
                nominal_preds=nominal_preds, alpha=alpha, stop_col=stop_var,
                penalizer=penalizer,
            )

        if "error" in result:
            st.error(result["error"])
            return

        # Cache — extract coef_table (DataFrame) separately
        cache_data = {
            "inputs": (
                mode, time_var, stop_var, event_var, group_var,
                tuple(predictors or []), alpha, penalizer,
            ),
            "result": {k: v for k, v in result.items() if k != "coef_table"},
        }
        if "coef_table" in result:
            cache_data["coef_table"] = result["coef_table"]

        st.session_state[_CACHE_KEY] = cache_data

    # ── Cache invalidation ────────────────────────────────────────────
    current_inputs = (
        mode, time_var, stop_var, event_var, group_var,
        tuple(predictors or []), alpha, penalizer,
    )
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached["inputs"] != current_inputs:
        del st.session_state[_CACHE_KEY]
        cached = None

    if not cached:
        _render_page_guide()
        return

    result = cached["result"]
    coef_table = cached.get("coef_table")

    if result.get("warning"):
        st.warning(result["warning"])

    # ── Display results by mode ───────────────────────────────────────
    if mode == "Kaplan-Meier / Log-Rank":
        _render_km_results(result, alpha, time_var, event_var, group_var)
    elif mode == "Cox Proportional Hazards":
        _render_cox_results(
            result, coef_table, alpha, time_var, event_var, predictors,
        )
    else:
        _render_extended_results(
            result, coef_table, alpha, time_var, stop_var, event_var, predictors,
        )

    _render_page_guide()


# ── Kaplan-Meier results ──────────────────────────────────────────────


def _render_km_results(result, alpha, time_var, event_var, group_var):
    tab_res, tab_chart = st.tabs(["Results", "Charts"])

    with tab_res:
        cols = st.columns(3)
        cols[0].metric("N", result["n"])
        cols[1].metric("Events", result["n_events"])
        cols[2].metric("Censored", result["n_censored"])

        if result.get("median_survival") is not None:
            m_cols = st.columns(3)
            m_cols[0].metric(
                "Median Survival", f"{result['median_survival']:.4f}",
            )
            if result.get("median_ci_lower") is not None:
                m_cols[1].metric(
                    "CI Lower", f"{result['median_ci_lower']:.4f}",
                )
                m_cols[2].metric(
                    "CI Upper", f"{result['median_ci_upper']:.4f}",
                )

        if "group_summary" in result:
            st.markdown("### Group Summary")
            gs_df = pd.DataFrame(result["group_summary"])
            st.dataframe(gs_df, hide_index=True, width="stretch")

        if "logrank_p" in result:
            st.markdown("---")
            render_significance_result(
                "Log-Rank Test", "\u03c7\u00b2",
                result["logrank_statistic"],
                result["logrank_p"],
                result.get("logrank_df"),
                alpha,
            )

        if "hazard_ratio" in result:
            render_effect_size(
                "Hazard Ratio",
                result["hazard_ratio"],
                f"[{result['hr_ci_lower']:.3f}, {result['hr_ci_upper']:.3f}] "
                f"vs {result['hr_reference']}",
            )

    with tab_chart:
        fig = kaplan_meier_plot(result)
        st.plotly_chart(fig, width="stretch")

    # ── PDF export ────────────────────────────────────────────────
    st.divider()
    _tables = []
    if "group_summary" in result:
        _tables.append(
            _serialize_df(pd.DataFrame(result["group_summary"]),
                          "Group Summary"),
        )
    _log_entry = build_log_entry(
        entry_type="survival_km",
        title=(
            f"Kaplan-Meier: {time_var}"
            + (f" by {group_var}" if group_var else "")
        ),
        result=result,
        tables=_tables,
        variables={
            "time": time_var, "event": event_var,
            **({"group": group_var} if group_var else {}),
        },
        alpha=alpha,
        dataset_name=st.session_state.get("file_name", ""),
    )
    _include_chart = st.checkbox(
        "Include chart in PDF", value=True, key="surv_km_pdf_chart",
    )
    if _include_chart:
        _fig = kaplan_meier_plot(result)
        _log_entry["figures"] = [
            {"label": "Kaplan-Meier Curve", "fig_dict": _fig.to_dict()},
        ]
    _export_buttons(_log_entry, _include_chart, "km")


# ── Cox PH results ───────────────────────────────────────────────────


def _render_cox_results(result, coef_table, alpha, time_var, event_var,
                        predictors):
    tab_res, tab_assume, tab_chart = st.tabs(
        ["Results", "Assumptions", "Charts"],
    )

    with tab_res:
        cols = st.columns(4)
        cols[0].metric("N", result["n"])
        cols[1].metric("Events", result["n_events"])
        cols[2].metric("C-Index", f"{result['concordance_index']:.4f}")
        cols[3].metric("Partial AIC", f"{result['partial_aic']:.1f}")

        render_effect_size(
            "Concordance Index",
            result["concordance_index"],
            _interpret_cindex(result["concordance_index"]),
        )

        st.markdown("### Coefficient Table")
        if coef_table is not None:
            st.dataframe(
                coef_table, hide_index=True, width="stretch",
            )

    with tab_assume:
        st.markdown("### Proportional Hazards Test (Schoenfeld Residuals)")
        st.markdown(
            "Tests whether the hazard ratio for each covariate remains "
            "constant over time (**Test for Decay**). A significant result "
            "(p < \u03b1) indicates the PH assumption may be violated — the "
            "covariate's effect decays or changes over time."
        )
        ph = result.get("assumptions", {}).get("proportional_hazards", {})
        if ph:
            for cov_name, check in ph.items():
                render_assumption_check(
                    f"PH: {cov_name}",
                    check.get("statistic"),
                    check.get("p_value"),
                    check.get("passed", True),
                    check.get("detail", ""),
                )
        else:
            st.info("Proportional hazards test could not be computed.")

    with tab_chart:
        if result.get("forest_data"):
            st.markdown("### Forest Plot")
            fig = cox_forest_plot(result["forest_data"], alpha=alpha)
            st.plotly_chart(fig, width="stretch")

        if result.get("schoenfeld_data"):
            st.markdown("### Schoenfeld Residual Plots (Measuring Decay)")
            st.markdown(
                "A flat LOWESS line near zero indicates the PH assumption "
                "holds. A trending line indicates the covariate's effect "
                "changes over time."
            )
            covariates = list(
                result["schoenfeld_data"]["covariates"].keys(),
            )
            for cov in covariates:
                fig = schoenfeld_plot(result["schoenfeld_data"], cov)
                st.plotly_chart(fig, width="stretch")

    # ── PDF export ────────────────────────────────────────────────
    st.divider()
    _tables = []
    if coef_table is not None:
        _tables.append(_serialize_df(coef_table, "Coefficient Table"))
    _log_entry = build_log_entry(
        entry_type="survival_cox",
        title=(
            f"Cox PH: {time_var} ~ "
            f"{' + '.join(predictors[:4])}"
            f"{'...' if len(predictors) > 4 else ''}"
        ),
        result=result,
        tables=_tables,
        variables={
            "time": time_var, "event": event_var,
            "predictors": ", ".join(predictors),
        },
        alpha=alpha,
        dataset_name=st.session_state.get("file_name", ""),
    )
    _include_chart = st.checkbox(
        "Include charts in PDF", value=True, key="surv_cox_pdf_chart",
    )
    if _include_chart:
        _figures = []
        if result.get("forest_data"):
            _ffig = cox_forest_plot(result["forest_data"], alpha=alpha)
            _figures.append(
                {"label": "Forest Plot", "fig_dict": _ffig.to_dict()},
            )
        if result.get("schoenfeld_data"):
            covs = list(result["schoenfeld_data"]["covariates"].keys())
            for cov in covs[:3]:  # limit to first 3 for PDF size
                _sfig = schoenfeld_plot(result["schoenfeld_data"], cov)
                _figures.append(
                    {"label": f"Schoenfeld: {cov}", "fig_dict": _sfig.to_dict()},
                )
        _log_entry["figures"] = _figures
    _export_buttons(_log_entry, _include_chart, "cox")


# ── Extended Cox results ─────────────────────────────────────────────


def _render_extended_results(result, coef_table, alpha, time_var, stop_var,
                             event_var, predictors):
    tab_res, tab_chart = st.tabs(["Results", "Charts"])

    with tab_res:
        cols = st.columns(4)
        cols[0].metric("N", result["n"])
        cols[1].metric("Events", result["n_events"])
        cols[2].metric("C-Index", f"{result['concordance_index']:.4f}")
        cols[3].metric("Partial AIC", f"{result['partial_aic']:.1f}")

        # LR test: extended vs base
        if "lr_test_p" in result:
            st.markdown("---")
            render_significance_result(
                "Likelihood Ratio Test (Extended vs Base)",
                "\u03c7\u00b2",
                result["lr_test_statistic"],
                result["lr_test_p"],
                result.get("lr_test_df"),
                alpha,
            )
            st.caption(
                "Tests whether adding time-varying coefficients "
                "significantly improves model fit."
            )

        # AIC comparison
        if "base_aic" in result:
            st.markdown("### Model Comparison")
            comp_cols = st.columns(3)
            comp_cols[0].metric(
                "Base Model AIC", f"{result['base_aic']:.1f}",
            )
            comp_cols[1].metric(
                "Extended Model AIC", f"{result['partial_aic']:.1f}",
            )
            comp_cols[2].metric(
                "Base C-Index", f"{result['base_concordance']:.4f}",
            )

        # Decay detection results
        if result.get("decay_results"):
            st.markdown("### Time-Varying Effects (Decay Detection)")
            st.markdown(
                "Interaction terms (covariate \u00d7 log(time)) test "
                "whether each covariate's effect changes over time. "
                "A significant p-value indicates non-proportional hazards."
            )
            decay_df = pd.DataFrame(result["decay_results"])
            st.dataframe(
                decay_df, hide_index=True, width="stretch",
            )

        st.markdown("### Full Coefficient Table")
        if coef_table is not None:
            st.dataframe(
                coef_table, hide_index=True, width="stretch",
            )

    with tab_chart:
        if result.get("forest_data"):
            fig = cox_forest_plot(result["forest_data"], alpha=alpha)
            st.plotly_chart(fig, width="stretch")

    # ── PDF export ────────────────────────────────────────────────
    st.divider()
    _tables = []
    if coef_table is not None:
        _tables.append(_serialize_df(coef_table, "Coefficient Table"))
    if result.get("decay_results"):
        _tables.append(
            _serialize_df(
                pd.DataFrame(result["decay_results"]), "Decay Detection",
            ),
        )
    _time_label = f"{time_var} → {stop_var}" if stop_var else time_var
    _log_entry = build_log_entry(
        entry_type="survival_extended",
        title=(
            f"Extended Cox: [{_time_label}] ~ "
            f"{' + '.join(predictors[:3])}"
            f"{'...' if len(predictors) > 3 else ''}"
        ),
        result=result,
        tables=_tables,
        variables={
            "start_time": time_var,
            **({"stop_time": stop_var} if stop_var else {}),
            "event": event_var,
            "predictors": ", ".join(predictors),
        },
        alpha=alpha,
        dataset_name=st.session_state.get("file_name", ""),
    )
    _include_chart = st.checkbox(
        "Include charts in PDF", value=True, key="surv_ext_pdf_chart",
    )
    if _include_chart:
        _figures = []
        if result.get("forest_data"):
            _ffig = cox_forest_plot(result["forest_data"], alpha=alpha)
            _figures.append(
                {"label": "Forest Plot", "fig_dict": _ffig.to_dict()},
            )
        _log_entry["figures"] = _figures
    _export_buttons(_log_entry, _include_chart, "ext")


# ── Helpers ───────────────────────────────────────────────────────────


def _export_buttons(log_entry, include_chart, suffix):
    """Common PDF export buttons."""
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if st.button("Add to Report", key=f"surv_{suffix}_add_report"):
            if log_result(log_entry):
                st.success("Added to report log.")
            else:
                st.error("Report log is full (100 entries). Clear it first.")
    with exp_col2:
        st.download_button(
            "Export PDF",
            data=generate_single_report(
                log_entry, include_charts=include_chart,
            ),
            file_name=f"survival_analysis_{suffix}.pdf",
            mime="application/pdf",
        )


def _interpret_cindex(c):
    """Interpret concordance index."""
    if c >= 0.9:
        return "excellent"
    if c >= 0.8:
        return "strong"
    if c >= 0.7:
        return "moderate"
    if c >= 0.6:
        return "weak"
    return "poor"


def _render_page_guide():
    """Page guide expander."""
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
Survival analysis studies **time-to-event data** — how long it takes before
an event of interest occurs (e.g., death, failure, churn, relapse). It
handles **censored observations**, where the event hasn't occurred by the
end of the study.

#### Key Concepts

**Censoring**: When a subject's event time is unknown because they left the
study, the study ended, or the event hasn't occurred yet. Right-censoring
(the most common type) means we know the subject survived *at least* until
time t. Censored observations are coded as **0** and events as **1**.

**Survival Function S(t)**: The probability of surviving (not experiencing
the event) beyond time t. Starts at 1.0 and decreases over time.

---

#### Kaplan-Meier / Log-Rank Mode

**Kaplan-Meier Estimator**: Non-parametric estimate of the survival
function. Creates a step-function curve that drops at each observed event
time. Censored observations are shown as **+** marks on the curve.

**Median Survival**: The time at which the survival probability drops to
0.50 (50%). If survival never reaches 50%, the median is undefined.

**Log-Rank Test**: Tests whether survival curves differ significantly
between groups. Compares the observed vs. expected number of events across
all time points. A significant p-value indicates different survival
experiences between groups.

**Hazard Ratio (HR)**: For 2-group comparisons, the ratio of hazard rates.
HR > 1 means the second group has higher risk; HR < 1 means lower risk.
HR = 1 means no difference.

---

#### Cox Proportional Hazards Mode

**Cox PH Model**: A semi-parametric regression model that relates survival
time to predictor variables. It models the hazard function (instantaneous
rate of the event) as a function of covariates, without specifying the
baseline hazard.

**Hazard Ratio**: exp(coefficient). For a one-unit increase in the
predictor, the hazard is multiplied by the HR. HR > 1 = increased risk,
HR < 1 = decreased risk.

**Concordance Index (C-Index)**: Measures the model's ability to correctly
rank survival times. Ranges from 0.5 (random) to 1.0 (perfect).
- 0.5 = no discrimination (random)
- 0.6--0.7 = weak
- 0.7--0.8 = moderate
- 0.8--0.9 = strong
- 0.9+ = excellent

**Proportional Hazards Assumption**: The Cox model assumes that hazard
ratios remain **constant over time**.

**Schoenfeld Residuals Test (Test for Decay)**: Tests whether each
covariate's effect changes over time. A significant result (p < alpha)
indicates the PH assumption is violated — the covariate's effect **decays**
or changes direction as time progresses.

**Schoenfeld Residual Plots (Measuring the Decay)**: Visual assessment of
the PH assumption. The LOWESS smoothing line should be approximately flat
if the PH assumption holds. A trending line indicates time-varying effects
— the direction shows whether the effect strengthens or weakens over time.

---

#### Extended Cox Model Mode

**Time-Varying Coefficients**: When the PH assumption is violated, the
Extended Cox model adds **covariate x log(time) interaction terms**. These
model how each covariate's effect changes over time.

**Interpretation of Interaction Terms**:
- Positive interaction coefficient: effect **increases** over time
- Negative interaction coefficient: effect **decays** over time
- Non-significant interaction: effect is approximately constant (PH holds)

**Likelihood Ratio Test**: Compares the extended model (with time
interactions) to the base model (without). A significant result indicates
that at least one covariate has a time-varying effect.

**AIC Comparison**: Lower AIC indicates better model fit. If the extended
model has lower AIC, time-varying coefficients improve the model.
        """)

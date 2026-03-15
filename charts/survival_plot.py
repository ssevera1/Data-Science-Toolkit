"""Survival analysis chart functions."""

import numpy as np
import plotly.graph_objects as go

from charts.theme import apply_theme, get_chart_colors
from utils.theme import get_colors


def _hex_to_rgba(hex_color, alpha=0.2):
    """Convert hex color to rgba string for CI bands."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def kaplan_meier_plot(km_result):
    """Step-function survival curves with CI bands and censoring marks."""
    colors = get_chart_colors()
    curves = km_result.get("curves", [])

    fig = go.Figure()

    if not curves:
        fig.add_annotation(
            text="No survival data to plot", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )
        return apply_theme(fig)

    for i, curve in enumerate(curves):
        color = colors[i % len(colors)]
        timeline = curve["timeline"]
        survival = curve["survival"]
        ci_lower = curve["ci_lower"]
        ci_upper = curve["ci_upper"]
        label = curve["label"]

        # CI upper (invisible line for fill)
        fig.add_trace(go.Scatter(
            x=timeline, y=ci_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            line_shape="hv",
        ))

        # CI lower with fill to CI upper
        fig.add_trace(go.Scatter(
            x=timeline, y=ci_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=_hex_to_rgba(color, 0.15),
            showlegend=False,
            hoverinfo="skip",
            line_shape="hv",
        ))

        # Main survival curve
        fig.add_trace(go.Scatter(
            x=timeline, y=survival,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
            line_shape="hv",
            hovertemplate="t=%{x:.2f}<br>S(t)=%{y:.4f}<extra></extra>",
        ))

        # Censoring tick marks
        censor_times = curve.get("censor_times", [])
        censor_survivals = curve.get("censor_survivals", [])
        if censor_times:
            fig.add_trace(go.Scatter(
                x=censor_times,
                y=censor_survivals,
                mode="markers",
                marker=dict(
                    symbol="cross-thin", size=8, color=color,
                    line=dict(width=1.5, color=color),
                ),
                showlegend=False,
                hovertemplate=(
                    "Censored at t=%{x:.2f}<br>S(t)=%{y:.4f}<extra></extra>"
                ),
            ))

    # Log-rank annotation
    if "logrank_p" in km_result:
        p = km_result["logrank_p"]
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
        annotation = f"Log-Rank {p_text}"
        if "hazard_ratio" in km_result:
            annotation += f"\nHR = {km_result['hazard_ratio']:.3f}"
        fig.add_annotation(
            text=annotation,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(size=11),
            align="right",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=1,
        )

    fig.update_layout(
        title="Kaplan-Meier Survival Curve",
        xaxis_title="Time",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
    )

    return apply_theme(fig)


def cox_forest_plot(forest_data, alpha=0.05):
    """Forest plot of hazard ratios from Cox model."""
    if not forest_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No coefficient data to plot", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        )
        return apply_theme(fig)

    colors = get_chart_colors()
    c = get_colors()

    covariates = [d.get("covariate", "?") for d in forest_data]
    hrs = [d.get("hr", 1.0) for d in forest_data]
    ci_lowers = [d.get("ci_lower", 1.0) for d in forest_data]
    ci_uppers = [d.get("ci_upper", 1.0) for d in forest_data]
    p_values = [d.get("p", 1.0) for d in forest_data]

    fig = go.Figure()

    # Reference line at HR=1
    fig.add_vline(
        x=1.0, line_dash="dash", line_color=c["text_muted"], line_width=1,
    )

    for i, (cov, hr, lo, hi, p) in enumerate(
        zip(covariates, hrs, ci_lowers, ci_uppers, p_values)
    ):
        sig = p < alpha
        color = colors[0] if sig else c["text_muted"]

        # CI whisker
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[i, i],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

        # HR point
        fig.add_trace(go.Scatter(
            x=[hr], y=[i],
            mode="markers",
            marker=dict(size=10, color=color, symbol="diamond"),
            showlegend=False,
            hovertemplate=(
                f"{cov}<br>HR={hr:.3f} [{lo:.3f}, {hi:.3f}]"
                f"<br>p={p:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Forest Plot — Hazard Ratios",
        xaxis_title="Hazard Ratio",
        xaxis_type="log",
        yaxis=dict(
            tickvals=list(range(len(covariates))),
            ticktext=covariates,
        ),
        height=max(300, 60 * len(covariates) + 100),
    )

    return apply_theme(fig)


def schoenfeld_plot(schoenfeld_data, covariate_name):
    """Scatter of Schoenfeld residuals vs time with LOWESS smoothing."""
    colors = get_chart_colors()
    c = get_colors()

    if (
        not schoenfeld_data
        or covariate_name not in schoenfeld_data.get("covariates", {})
    ):
        fig = go.Figure()
        fig.add_annotation(
            text=f"No residual data for {covariate_name}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14),
        )
        return apply_theme(fig)

    times = np.array(schoenfeld_data["time"])
    residuals = np.array(schoenfeld_data["covariates"][covariate_name])

    fig = go.Figure()

    # Reference line at 0
    fig.add_hline(
        y=0, line_dash="dash", line_color=c["text_muted"], line_width=1,
    )

    # Scatter
    fig.add_trace(go.Scatter(
        x=times, y=residuals,
        mode="markers",
        marker=dict(color=colors[0], size=5, opacity=0.5),
        name="Residuals",
    ))

    # LOWESS smoothing line
    if len(times) >= 5:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smooth = lowess(residuals, times, frac=0.6, return_sorted=True)
            fig.add_trace(go.Scatter(
                x=smooth[:, 0], y=smooth[:, 1],
                mode="lines",
                line=dict(color=colors[1], width=2),
                name="LOWESS",
            ))
        except Exception:
            pass

    fig.update_layout(
        title=f"Schoenfeld Residuals — {covariate_name}",
        xaxis_title="Time",
        yaxis_title="Schoenfeld Residual",
    )

    return apply_theme(fig)

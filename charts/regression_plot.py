"""Regression line with confidence band."""

import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
from charts.theme import apply_theme, COLORS


def regression_scatter(df, x_col, y_col, title=None):
    """Scatter plot with OLS regression line and 95% CI band."""
    clean = df[[x_col, y_col]].dropna()
    x = clean[x_col].values
    y = clean[y_col].values

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name="Data",
        marker=dict(color=COLORS[0], size=7, opacity=0.7),
    ))

    # Fit OLS
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Prediction line and CI
    x_sorted = np.sort(x)
    X_pred = sm.add_constant(x_sorted)
    predictions = model.get_prediction(X_pred)
    pred_summary = predictions.summary_frame(alpha=0.05)

    # Regression line
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=pred_summary["mean"],
        mode="lines",
        name="Regression line",
        line=dict(color=COLORS[1], width=2),
    ))

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_sorted, x_sorted[::-1]]),
        y=np.concatenate([pred_summary["mean_ci_upper"], pred_summary["mean_ci_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(26,26,46,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
    ))

    fig.update_layout(
        title=title or f"Regression: {y_col} ~ {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    return apply_theme(fig)


def multi_regression_actual_vs_predicted(y_actual, y_predicted, title=None):
    """Actual vs predicted plot for multiple regression."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_predicted, y=y_actual,
        mode="markers",
        name="Observations",
        marker=dict(color=COLORS[0], size=7, opacity=0.7),
    ))

    # Perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        name="Perfect fit",
        line=dict(color=COLORS[1], width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title or "Actual vs Predicted",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    return apply_theme(fig)

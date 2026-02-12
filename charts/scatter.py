"""Scatter plots for correlation."""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from charts.theme import apply_theme, COLORS


def correlation_scatter(df, x_col, y_col, r_value=None, title=None):
    """Scatter plot with optional trend line for correlation."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        trendline="ols",
        color_discrete_sequence=COLORS,
        title=title or f"{y_col} vs {x_col}",
    )

    fig.update_traces(
        marker=dict(size=7, opacity=0.7),
        selector=dict(mode="markers"),
    )

    # Style the trendline
    fig.update_traces(
        line=dict(color=COLORS[1], width=2),
        selector=dict(mode="lines"),
    )

    if r_value is not None:
        fig.add_annotation(
            text=f"r = {r_value:.3f}",
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            font=dict(size=14, color=COLORS[0]),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=COLORS[0],
            borderwidth=1,
            borderpad=4,
        )

    return apply_theme(fig)

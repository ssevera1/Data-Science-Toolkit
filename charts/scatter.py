"""Scatter plots for correlation."""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from charts.theme import apply_theme, get_chart_colors
from utils.theme import get_colors, hex_to_rgb


def correlation_scatter(df, x_col, y_col, r_value=None, title=None):
    """Scatter plot with optional trend line for correlation."""
    colors = get_chart_colors()
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        trendline="ols",
        color_discrete_sequence=colors,
        title=title or f"{y_col} vs {x_col}",
    )

    fig.update_traces(
        marker=dict(size=7, opacity=0.7),
        selector=dict(mode="markers"),
    )

    # Style the trendline
    fig.update_traces(
        line=dict(color=colors[1], width=2),
        selector=dict(mode="lines"),
    )

    if r_value is not None:
        c = get_colors()
        fig.add_annotation(
            text=f"r = {r_value:.3f}",
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            showarrow=False,
            font=dict(size=14, color=colors[0]),
            bgcolor=f"rgba({hex_to_rgb(c['bg_card'])},0.9)",
            bordercolor=colors[0],
            borderwidth=1,
            borderpad=4,
        )

    return apply_theme(fig)

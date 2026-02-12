"""Histograms with optional normal curve overlay."""

import plotly.graph_objects as go
import numpy as np
from scipy import stats
from charts.theme import apply_theme, get_chart_colors


def histogram_with_normal(series, name, title=None, bins=None):
    """Histogram with optional normal distribution curve."""
    data = series.dropna().values
    colors = get_chart_colors()

    if len(data) == 0:
        fig = go.Figure()
        fig.update_layout(title=title or f"Distribution of {name}",
                          annotations=[dict(text="No data available", showarrow=False,
                                            xref="paper", yref="paper", x=0.5, y=0.5)])
        return apply_theme(fig)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        name=name,
        marker_color=colors[0],
        opacity=0.7,
        nbinsx=bins or min(30, max(10, len(data) // 5)),
        histnorm="probability density",
    ))

    # Normal curve overlay
    x_range = np.linspace(data.min(), data.max(), 200)
    mu, sigma = data.mean(), data.std()
    if sigma > 0:
        normal_curve = stats.norm.pdf(x_range, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_curve,
            mode="lines",
            name="Normal curve",
            line=dict(color=colors[1], width=2),
        ))

    fig.update_layout(
        title=title or f"Distribution of {name}",
        xaxis_title=name,
        yaxis_title="Density",
        barmode="overlay",
    )
    return apply_theme(fig)

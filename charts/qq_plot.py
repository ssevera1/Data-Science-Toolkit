"""QQ plots for normality assessment."""

import plotly.graph_objects as go
import numpy as np
from scipy import stats
from charts.theme import apply_theme, COLORS


def qq_plot(series, name, title=None):
    """QQ plot comparing data to normal distribution."""
    data = np.sort(series.dropna().values)
    n = len(data)

    # Theoretical quantiles
    theoretical = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    fig = go.Figure()

    # QQ points
    fig.add_trace(go.Scatter(
        x=theoretical,
        y=data,
        mode="markers",
        name="Data",
        marker=dict(color=COLORS[0], size=6, opacity=0.7),
    ))

    # Reference line
    slope, intercept = np.polyfit(theoretical, data, 1)
    line_x = np.array([theoretical.min(), theoretical.max()])
    line_y = slope * line_x + intercept

    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="Reference line",
        line=dict(color=COLORS[1], width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title or f"Q-Q Plot: {name}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
    )
    return apply_theme(fig)

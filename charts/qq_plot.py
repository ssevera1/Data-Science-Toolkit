"""QQ plots for normality assessment."""

import plotly.graph_objects as go
import numpy as np
from scipy import stats
from charts.theme import apply_theme, get_chart_colors


def qq_plot(series, name, title=None):
    """QQ plot comparing data to normal distribution."""
    data = np.sort(series.dropna().values)
    n = len(data)
    if n < 2:
        fig = go.Figure()
        fig.update_layout(title=title or f"Q-Q Plot: {name}",
                          annotations=[dict(text="Need at least 2 data points",
                                            xref="paper", yref="paper",
                                            x=0.5, y=0.5, showarrow=False)])
        return apply_theme(fig)
    colors = get_chart_colors()

    # Theoretical quantiles
    theoretical = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    fig = go.Figure()

    # QQ points
    fig.add_trace(go.Scatter(
        x=theoretical,
        y=data,
        mode="markers",
        name="Data",
        marker=dict(color=colors[0], size=6, opacity=0.7),
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
        line=dict(color=colors[1], width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title or f"Q-Q Plot: {name}",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
    )
    return apply_theme(fig)

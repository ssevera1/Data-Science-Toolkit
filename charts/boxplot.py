"""Box plots for t-tests and ANOVA."""

import plotly.express as px
import plotly.graph_objects as go
from charts.theme import apply_theme, COLORS


def grouped_boxplot(df, value_col, group_col, title=None):
    """Box plot with data points, grouped by a categorical variable."""
    fig = px.box(
        df,
        x=group_col,
        y=value_col,
        color=group_col,
        points="all",
        color_discrete_sequence=COLORS,
        title=title or f"{value_col} by {group_col}",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.6))
    fig.update_layout(showlegend=False)
    return apply_theme(fig)


def paired_boxplot(df, col1, col2, title=None):
    """Side-by-side box plots for paired data."""
    import pandas as pd

    melted = pd.DataFrame({
        "Value": list(df[col1]) + list(df[col2]),
        "Condition": [col1] * len(df) + [col2] * len(df),
    })

    fig = px.box(
        melted,
        x="Condition",
        y="Value",
        color="Condition",
        points="all",
        color_discrete_sequence=COLORS,
        title=title or f"{col1} vs {col2}",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.6))
    fig.update_layout(showlegend=False)
    return apply_theme(fig)


def single_boxplot(series, name, title=None):
    """Single box plot for one-sample tests."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=series,
        name=name,
        marker_color=COLORS[0],
        boxpoints="all",
        jitter=0.3,
        pointpos=-1.5,
        marker=dict(size=4, opacity=0.6),
    ))
    fig.update_layout(title=title or f"Distribution of {name}")
    return apply_theme(fig)

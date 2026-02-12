"""Bar charts for group means."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from charts.theme import apply_theme, COLORS


def group_means_bar(df, value_col, group_col, title=None):
    """Bar chart showing group means with error bars (SE)."""
    summary = df.groupby(group_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    fig = go.Figure()
    for i, row in summary.iterrows():
        fig.add_trace(go.Bar(
            x=[str(row[group_col])],
            y=[row["mean"]],
            error_y=dict(type="data", array=[row["se"]], visible=True),
            marker_color=COLORS[i % len(COLORS)],
            name=str(row[group_col]),
        ))

    fig.update_layout(
        title=title or f"Mean {value_col} by {group_col}",
        xaxis_title=group_col,
        yaxis_title=f"Mean {value_col}",
        showlegend=False,
        barmode="group",
    )
    return apply_theme(fig)


def two_way_bar(df, value_col, factor1, factor2, title=None):
    """Grouped bar chart for two-way designs."""
    summary = df.groupby([factor1, factor2])[value_col].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    fig = px.bar(
        summary,
        x=factor1,
        y="mean",
        color=factor2,
        barmode="group",
        error_y="se",
        color_discrete_sequence=COLORS,
        title=title or f"Mean {value_col} by {factor1} and {factor2}",
    )
    fig.update_layout(
        xaxis_title=factor1,
        yaxis_title=f"Mean {value_col}",
    )
    return apply_theme(fig)

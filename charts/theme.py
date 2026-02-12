"""Plotly template with dark theme colors matching DS Power Tools."""

import plotly.graph_objects as go
import plotly.io as pio

# Define the color palette (dark theme)
COLORS = [
    "#667eea",  # accent primary
    "#764ba2",  # accent secondary
    "#3498db",  # info blue
    "#2ecc71",  # success green
    "#f39c12",  # warning yellow
    "#e74c3c",  # error red
    "#1abc9c",  # teal
    "#e67e22",  # orange
    "#9b59b6",  # purple
    "#2c3e50",  # dark
]

BACKGROUND = "#0e1117"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
FONT_FAMILY = "Inter, sans-serif"


def get_template():
    """Get the custom Plotly template."""
    template = go.layout.Template()

    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR, size=13),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        colorway=COLORS,
        title=dict(font=dict(size=16, color="#ffffff")),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            title=dict(font=dict(size=13)),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            title=dict(font=dict(size=13)),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID_COLOR,
            borderwidth=1,
            font=dict(color=TEXT_COLOR),
        ),
        margin=dict(l=60, r=30, t=50, b=60),
    )

    return template


def apply_theme(fig):
    """Apply the dark theme to a Plotly figure."""
    fig.update_layout(template=get_template())
    return fig

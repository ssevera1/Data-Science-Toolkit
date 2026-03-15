"""Plotly template with dynamic theme colors matching DS Power Tools."""

import plotly.graph_objects as go
import plotly.io as pio
from utils.theme import get_colors, FONT_FAMILY


def get_chart_colors():
    """Return the active data-viz colorway list."""
    return get_colors()["viz"]


# Backwards-compatible module-level list (dark defaults).
# New code should call get_chart_colors() instead.
COLORS = get_chart_colors()


def get_template():
    """Get the custom Plotly template using the active palette."""
    c = get_colors()
    template = go.layout.Template()

    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, color=c["text_body"], size=13),
        paper_bgcolor=c["bg_primary"],
        plot_bgcolor=c["bg_primary"],
        colorway=c["viz"],
        title=dict(font=dict(size=16, color=c["text_bright"])),
        xaxis=dict(
            gridcolor=c["border"],
            linecolor=c["border"],
            zerolinecolor=c["border"],
            title=dict(font=dict(size=13)),
        ),
        yaxis=dict(
            gridcolor=c["border"],
            linecolor=c["border"],
            zerolinecolor=c["border"],
            title=dict(font=dict(size=13)),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=c["border"],
            borderwidth=1,
            font=dict(color=c["text_body"]),
        ),
        margin=dict(l=60, r=30, t=50, b=60),
    )

    return template


def apply_theme(fig):
    """Apply the active theme to a Plotly figure."""
    fig.update_layout(template=get_template())
    return fig

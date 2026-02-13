"""
DS Power Tools — Central styling module.

Provides dual-theme styling (Dark / Light) for Streamlit, Plotly, and Matplotlib.
"""

import html
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

# ── Color Palettes ───────────────────────────────────────────────────────────

DARK_COLORS = {
    "bg_primary": "#0e1117",
    "bg_card": "#1a1a2e",
    "bg_sidebar": "#16213e",
    "border": "#2a2a4a",
    "text_body": "#e0e0e0",
    "text_muted": "#a0a0b8",
    "text_bright": "#ffffff",
    "accent_primary": "#4ea8ff",
    "accent_secondary": "#38bdf8",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "error": "#e74c3c",
    "info": "#3498db",
    "title_gradient_start": "#4ea8ff",
    "title_gradient_end": "#38bdf8",
    "hover_bg": "#262640",
    "tab_bg": "#1a1a2e",
    "table_even_row": "rgba(26,26,46,0.5)",
    "scrollbar_track": "#0e1117",
    "footer_text": "#555",
    "drift_highlight": "#3d1f1f",
    # Data viz colorway
    "viz": [
        "#4ea8ff", "#38bdf8", "#3498db", "#2ecc71",
        "#f39c12", "#e74c3c", "#1abc9c", "#e67e22",
        "#9b59b6", "#2c3e50",
    ],
}

LIGHT_COLORS = {
    "bg_primary": "#F6F0E2",
    "bg_card": "#FFFFFF",
    "bg_sidebar": "#EDE7D9",
    "border": "#D5CFC1",
    "text_body": "#000000",
    "text_muted": "#555555",
    "text_bright": "#000000",
    "accent_primary": "#EE0011",
    "accent_secondary": "#FF281E",
    "success": "#00B845",
    "warning": "#FED60E",
    "error": "#FF281E",
    "info": "#0089EC",
    "title_gradient_start": "#EE0011",
    "title_gradient_end": "#FF281E",
    "hover_bg": "#F0EAD6",
    "tab_bg": "#FFFFFF",
    "table_even_row": "rgba(237,231,217,0.5)",
    "scrollbar_track": "#F6F0E2",
    "footer_text": "#888",
    "drift_highlight": "#fde8e8",
    # Data viz colorway
    "viz": [
        "#0089EC", "#00B845", "#FF8027", "#FED60E",
        "#EE0011", "#764ba2", "#1abc9c", "#e67e22",
        "#9b59b6", "#2c3e50",
    ],
}

# Backwards-compatible alias so existing `from utils.theme import COLORS` still works
COLORS = DARK_COLORS


# ── Theme Accessor ───────────────────────────────────────────────────────────

def get_colors():
    """Return the active palette dict based on session-state theme."""
    theme = st.session_state.get("app_theme", "Light")
    return LIGHT_COLORS if theme == "Light" else DARK_COLORS


def _is_light():
    return st.session_state.get("app_theme", "Light") == "Light"


# ── Plotly Template Registration ─────────────────────────────────────────────

def register_plotly_theme():
    """Build and register a Plotly template from the active palette."""
    c = get_colors()
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor=c["bg_primary"],
        plot_bgcolor=c["bg_primary"],
        font=dict(color=c["text_body"], family="Inter, sans-serif"),
        title=dict(font=dict(color=c["text_bright"])),
        xaxis=dict(
            gridcolor=c["border"],
            zerolinecolor=c["border"],
            linecolor=c["border"],
        ),
        yaxis=dict(
            gridcolor=c["border"],
            zerolinecolor=c["border"],
            linecolor=c["border"],
        ),
        colorway=c["viz"],
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates["ds_tools"] = tpl
    pio.templates.default = "ds_tools"


# Register once at import time (dark defaults)
register_plotly_theme()


# ── Global CSS Injection ────────────────────────────────────────────────────

def inject_global_css():
    """Inject theme-aware CSS into the Streamlit page."""
    c = get_colors()
    light = _is_light()

    # Build gradient strings
    accent_grad = f"linear-gradient(90deg, {c['title_gradient_start']} 0%, {c['title_gradient_end']} 100%)"
    card_grad = f"linear-gradient(135deg, {c['bg_card']} 0%, {c['bg_sidebar']} 100%)" if not light else f"linear-gradient(135deg, {c['bg_card']} 0%, {c['bg_card']} 100%)"
    sidebar_grad = f"linear-gradient(180deg, {c['bg_sidebar']} 0%, {c['bg_primary']} 100%)" if not light else f"linear-gradient(180deg, {c['bg_sidebar']} 0%, {c['bg_sidebar']} 100%)"
    accent_bg_start = f"rgba({_hex_to_rgb(c['title_gradient_start'])},0.15)"
    accent_bg_end = f"rgba({_hex_to_rgb(c['title_gradient_end'])},0.10)"
    accent_border_rgba = f"rgba({_hex_to_rgb(c['accent_primary'])},0.3)"
    accent_shadow = f"rgba({_hex_to_rgb(c['accent_primary'])},0.15)"
    accent_shadow_hover = f"rgba({_hex_to_rgb(c['accent_primary'])},0.4)"
    accent_shadow_btn = f"rgba({_hex_to_rgb(c['accent_primary'])},0.25)"

    # For light theme buttons, use white text for contrast on red
    btn_text = "#ffffff"

    css = f"""
    <style>
    /* ── Font ───────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* ── Sidebar ────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {sidebar_grad};
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] a {{
        color: {c['text_muted']} !important;
        transition: color 0.2s;
    }}
    section[data-testid="stSidebar"] .stRadio label:hover,
    section[data-testid="stSidebar"] a:hover {{
        color: {c['accent_primary']} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {{
        background: linear-gradient(90deg, {accent_bg_start} 0%, {accent_bg_end} 100%);
        border-left: 3px solid {c['accent_primary']};
    }}

    /* ── Metric Cards ───────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {card_grad};
        border: 1px solid {c['border']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px {accent_shadow};
    }}
    [data-testid="stMetric"] label {{
        color: {c['text_muted']} !important;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {c['text_bright']} !important;
        font-weight: 700;
    }}

    /* ── Tabs (pill style) ──────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 20px;
        color: {c['text_muted']};
        background-color: {c['tab_bg']};
        border: 1px solid {c['border']};
        transition: all 0.2s;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {c['text_bright']};
        background-color: {c['hover_bg']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {accent_grad} !important;
        color: {btn_text} !important;
        border-color: transparent !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ── Buttons ────────────────────────────────────────────────── */
    .stButton > button {{
        background: {accent_grad};
        color: {btn_text};
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        box-shadow: 0 2px 8px {accent_shadow_btn};
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 16px {accent_shadow_hover};
        opacity: 0.95;
    }}
    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* ── Download Button ────────────────────────────────────────── */
    .stDownloadButton > button {{
        background: transparent;
        color: {c['accent_primary']};
        border: 2px solid {c['accent_primary']};
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }}
    .stDownloadButton > button:hover {{
        background: rgba({_hex_to_rgb(c['accent_primary'])},0.1);
        transform: translateY(-1px);
    }}

    /* ── Dataframe ──────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {c['border']};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* ── Inputs / Selects ───────────────────────────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {{
        border-radius: 8px !important;
        transition: border-color 0.2s;
    }}
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover,
    .stTextInput > div > div > input:hover,
    .stNumberInput > div > div > input:hover {{
        border-color: {c['accent_primary']} !important;
    }}

    /* ── Expander ───────────────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background: {c['bg_card']};
        border-radius: 8px;
        border: 1px solid {c['border']};
        color: {c['text_body']};
        font-weight: 600;
    }}

    /* ── File Uploader ──────────────────────────────────────────── */
    [data-testid="stFileUploader"] {{
        border: 2px dashed {c['border']};
        border-radius: 12px;
        padding: 1rem;
        transition: border-color 0.2s;
    }}
    [data-testid="stFileUploader"]:hover {{
        border-color: {c['accent_primary']};
    }}

    /* ── Alert Boxes ────────────────────────────────────────────── */
    .stAlert [data-testid="stNotification"] {{
        border-radius: 10px;
    }}

    /* ── Divider ────────────────────────────────────────────────── */
    hr {{
        border-color: {c['border']} !important;
    }}

    /* ── Progress Bar ───────────────────────────────────────────── */
    .stProgress > div > div > div > div {{
        background: {accent_grad};
    }}

    /* ── Scrollbar ──────────────────────────────────────────────── */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {c['scrollbar_track']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {c['border']};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {c['accent_primary']};
    }}

    /* ── Hero Section (home page) ───────────────────────────────── */
    .hero-section {{
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }}
    .hero-badge {{
        display: inline-block;
        background: linear-gradient(90deg, {accent_bg_start}, {accent_bg_end});
        border: 1px solid {accent_border_rgba};
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        color: {c['accent_primary']};
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-size: 3rem;
        font-weight: 700;
        color: {c['accent_primary']};
        margin: 0.5rem 0;
        line-height: 1.2;
    }}
    .hero-subtitle {{
        font-size: 1.15rem;
        color: {c['text_muted']};
        margin-top: 0.5rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }}

    /* ── Tool Cards (home page) ─────────────────────────────────── */
    .tool-card {{
        background: {card_grad};
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid {c['border']};
        min-height: 180px;
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }}
    .tool-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px {accent_shadow};
        border-color: {c['accent_primary']};
    }}
    .tool-card h3 {{
        margin-top: 0;
        color: {c['text_bright']};
        font-size: 1.05rem;
    }}
    .tool-card p {{
        color: {c['text_muted']};
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 0;
    }}
    .tool-card-icon {{
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
        display: block;
    }}

    /* ── Footer ─────────────────────────────────────────────────── */
    .app-footer {{
        text-align: center;
        color: {c['footer_text']};
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
    }}
    .app-footer span {{
        color: {c['accent_primary']};
        font-weight: 600;
    }}

    /* ── Statistics Results ──────────────────────────────────────── */
    .p-significant {{
        color: {c['success']};
        font-weight: 700;
    }}
    .p-not-significant {{
        color: {c['error']};
        font-weight: 500;
    }}
    .p-marginal {{
        color: {c['warning']};
        font-weight: 500;
    }}
    .assumption-pass {{
        background-color: rgba({_hex_to_rgb(c['success'])},0.15);
        color: {c['success']};
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .assumption-fail {{
        background-color: rgba({_hex_to_rgb(c['error'])},0.15);
        color: {c['error']};
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: 500;
        font-size: 0.85rem;
        display: inline-block;
    }}
    .results-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9rem;
    }}
    .results-table th {{
        background-color: {c['bg_card']};
        color: {c['text_body']};
        padding: 0.6rem 1rem;
        text-align: left;
        font-weight: 500;
    }}
    .results-table td {{
        padding: 0.5rem 1rem;
        border-bottom: 1px solid {c['border']};
        color: {c['text_body']};
    }}
    .results-table tr:nth-child(even) {{
        background-color: {c['table_even_row']};
    }}
    .info-box {{
        background-color: rgba({_hex_to_rgb(c['accent_primary'])},0.1);
        border-left: 4px solid {c['accent_primary']};
        padding: 1rem;
        border-radius: 0 6px 6px 0;
        margin: 1rem 0;
        color: {c['text_body']};
    }}
    </style>
    """

    # Light-theme overrides for native Streamlit widgets
    if light:
        css += f"""
    <style>
    /* ── Light Theme — Main Area ───────────────────────────────── */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {c['bg_primary']} !important;
    }}
    .stApp > header {{
        background-color: {c['bg_primary']} !important;
    }}
    [data-testid="stHeader"] {{
        background-color: {c['bg_primary']} !important;
    }}

    /* ── Text colors ────────────────────────────────────────────── */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp label,
    .stMarkdown, .stMarkdown p {{
        color: {c['text_body']} !important;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {c['text_bright']} !important;
    }}

    /* ── Sidebar text ───────────────────────────────────────────── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {{
        color: {c['text_body']} !important;
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] a {{
        color: {c['text_muted']} !important;
    }}

    /* ── Inputs ─────────────────────────────────────────────────── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {{
        background-color: {c['bg_card']} !important;
        color: {c['text_body']} !important;
        border-color: {c['border']} !important;
    }}

    /* ── Selectbox dropdown ─────────────────────────────────────── */
    [data-baseweb="popover"] {{
        background-color: {c['bg_card']} !important;
    }}
    [data-baseweb="menu"] {{
        background-color: {c['bg_card']} !important;
    }}
    [data-baseweb="menu"] li {{
        color: {c['text_body']} !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background-color: {c['hover_bg']} !important;
    }}

    /* ── Dataframe ──────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {{
        background-color: {c['bg_card']} !important;
    }}

    /* ── File Uploader ──────────────────────────────────────────── */
    [data-testid="stFileUploader"] {{
        background-color: {c['bg_card']} !important;
        border-color: {c['border']} !important;
    }}
    [data-testid="stFileUploader"] section {{
        background-color: {c['bg_card']} !important;
    }}
    [data-testid="stFileUploader"] section > div {{
        color: {c['text_body']} !important;
    }}
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {{
        color: {c['text_muted']} !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {c['bg_card']} !important;
        color: {c['text_body']} !important;
    }}
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] div {{
        color: {c['text_muted']} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        color: {c['accent_primary']} !important;
    }}

    /* ── Expander ───────────────────────────────────────────────── */
    [data-testid="stExpander"] {{
        background-color: {c['bg_card']} !important;
        border-color: {c['border']} !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    .streamlit-expanderHeader {{
        color: {c['text_body']} !important;
        background-color: {c['bg_card']} !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background-color: {c['bg_card']} !important;
    }}

    /* ── Slider ─────────────────────────────────────────────────── */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] div {{
        color: {c['text_body']} !important;
    }}
    [data-baseweb="slider"] div[role="slider"] {{
        background-color: {c['accent_primary']} !important;
    }}

    /* ── Checkbox / Radio ───────────────────────────────────────── */
    .stCheckbox label span,
    .stRadio label span {{
        color: {c['text_body']} !important;
    }}

    /* ── Metric cards ───────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: {c['bg_card']} !important;
    }}

    /* ── Tooltips / Popovers ────────────────────────────────────── */
    [data-baseweb="tooltip"] {{
        background-color: {c['bg_card']} !important;
        color: {c['text_body']} !important;
    }}

    /* ── Alerts / Notifications ─────────────────────────────────── */
    [data-testid="stNotification"] {{
        background-color: {c['bg_card']} !important;
    }}

    /* ── Bottom toolbar / status bar ────────────────────────────── */
    [data-testid="stBottom"],
    [data-testid="stStatusWidget"],
    footer {{
        background-color: {c['bg_primary']} !important;
        color: {c['text_muted']} !important;
    }}

    /* ── Emoji icons — prevent black square ────────────────────── */
    .emoji-icon {{
        color: initial !important;
        background: transparent !important;
    }}

    /* ── JSON viewer ────────────────────────────────────────────── */
    [data-testid="stJson"],
    [data-testid="stJson"] > div {{
        background-color: {c['bg_card']} !important;
        color: {c['text_body']} !important;
        border-radius: 8px;
    }}
    [data-testid="stJson"] pre {{
        background-color: {c['bg_card']} !important;
        color: {c['text_body']} !important;
    }}
    /* react-json-view overrides */
    [data-testid="stJson"] .react-json-view {{
        background-color: {c['bg_card']} !important;
        color: {c['text_body']} !important;
    }}
    [data-testid="stJson"] .object-key {{
        color: {c['accent_primary']} !important;
    }}
    [data-testid="stJson"] .string-value {{
        color: {c['success']} !important;
    }}
    [data-testid="stJson"] .integer-value,
    [data-testid="stJson"] .float-value {{
        color: {c['info']} !important;
    }}

    /* ── Code blocks ────────────────────────────────────────────── */
    [data-testid="stCode"],
    .stCodeBlock,
    .stCodeBlock code {{
        background-color: {c['bg_card']} !important;
        color: {c['text_body']} !important;
    }}
    pre:not([data-testid="stJson"] pre) {{
        background-color: {c['bg_card']} !important;
        color: {c['text_body']} !important;
    }}

    /* ── Plotly chart containers ────────────────────────────────── */
    [data-testid="stPlotlyChart"],
    .stPlotlyChart {{
        background-color: transparent !important;
    }}

    /* ── Spinner (spinning circle while processing) ────────────── */
    .stSpinner > div {{
        border-top-color: {c['accent_primary']} !important;
    }}
    .stSpinner svg circle {{
        stroke: {c['accent_primary']} !important;
    }}
    .stSpinner svg {{
        color: {c['accent_primary']} !important;
        fill: {c['accent_primary']} !important;
    }}
    .stSpinner > div > span {{
        color: {c['text_body']} !important;
    }}

    /* ── Status widget (top-right running/emoji indicator) ──────── */
    [data-testid="stStatusWidget"] svg {{
        fill: {c['text_body']} !important;
        stroke: {c['text_body']} !important;
        color: {c['text_body']} !important;
    }}
    [data-testid="stStatusWidget"] button {{
        color: {c['text_body']} !important;
    }}
    [data-testid="stStatusWidget"] i,
    [data-testid="stStatusWidget"] span {{
        color: {c['text_body']} !important;
    }}
    [data-testid="stStatusWidget"] button svg {{
        fill: {c['text_muted']} !important;
        stroke: {c['text_muted']} !important;
    }}

    /* ── Multi-select tags ──────────────────────────────────────── */
    [data-baseweb="tag"] {{
        background-color: {c['hover_bg']} !important;
        color: {c['text_body']} !important;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


def hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R,G,B' string for use in rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# Backwards-compatible alias
_hex_to_rgb = hex_to_rgb


# ── Page Header ──────────────────────────────────────────────────────────────

def page_header(title: str, description: str, icon: str = ""):
    """Render a gradient page title with accent underline and description."""
    c = get_colors()
    title = html.escape(title)
    description = html.escape(description)
    icon = html.escape(icon)
    icon_html = f'<span class="emoji-icon" style="font-size:2rem;margin-right:0.5rem;line-height:1;">{icon}</span>' if icon else ""
    accent_grad = f"linear-gradient(90deg,{c['title_gradient_start']} 0%,{c['title_gradient_end']} 100%)"
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
                {icon_html}
                <h1 style="
                    font-size:2rem;
                    font-weight:700;
                    color:{c['accent_primary']};
                    margin:0;
                ">{title}</h1>
            </div>
            <div style="
                width:60px;height:3px;
                background:{accent_grad};
                border-radius:2px;
                margin:0.5rem 0;
            "></div>
            <p style="color:{c['text_muted']};font-size:1rem;margin:0;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Metric Card ──────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, icon: str = ""):
    """Render a styled metric card with optional icon."""
    c = get_colors()
    light = _is_light()
    label = html.escape(label)
    value = html.escape(value)
    icon = html.escape(icon)
    icon_html = f'<span class="emoji-icon" style="font-size:1.5rem;margin-right:0.5rem;line-height:1;">{icon}</span>' if icon else ""
    card_grad = f"linear-gradient(135deg,{c['bg_card']} 0%,{c['bg_sidebar']} 100%)" if not light else f"linear-gradient(135deg,{c['bg_card']} 0%,{c['bg_card']} 100%)"
    st.markdown(
        f"""
        <div style="
            background:{card_grad};
            border:1px solid {c['border']};
            border-radius:12px;
            padding:1rem 1.2rem;
            text-align:center;
            transition:transform 0.2s,box-shadow 0.2s;
        ">
            <div>{icon_html}</div>
            <div style="color:{c['text_muted']};font-size:0.85rem;margin-bottom:0.25rem;">{label}</div>
            <div style="color:{c['text_bright']};font-size:1.5rem;font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Matplotlib Theme ────────────────────────────────────────────────────────

def set_matplotlib_theme():
    """Configure matplotlib rcParams for the active theme."""
    import matplotlib.pyplot as plt
    c = get_colors()

    params = {
        "figure.facecolor": c["bg_primary"],
        "axes.facecolor": c["bg_primary"],
        "axes.edgecolor": c["border"],
        "axes.labelcolor": c["text_body"],
        "text.color": c["text_body"],
        "xtick.color": c["text_muted"],
        "ytick.color": c["text_muted"],
        "grid.color": c["border"],
        "legend.facecolor": c["bg_card"],
        "legend.edgecolor": c["border"],
        "savefig.facecolor": c["bg_primary"],
    }
    plt.rcParams.update(params)


# Backwards-compatible alias
set_matplotlib_dark = set_matplotlib_theme

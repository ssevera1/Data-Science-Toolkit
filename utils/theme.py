"""
DS Power Tools — Central styling module.

Provides dark-themed styling for Streamlit, Plotly, and Matplotlib.
"""

import html
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

# ── Color Palette ─────────────────────────────────────────────────────────────

COLORS = {
    "bg_primary": "#0e1117",
    "bg_card": "#1a1a2e",
    "bg_sidebar": "#16213e",
    "text_body": "#e0e0e0",
    "text_muted": "#a0a0b8",
    "text_bright": "#ffffff",
    "accent_primary": "#667eea",
    "accent_secondary": "#764ba2",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "error": "#e74c3c",
    "info": "#3498db",
}

# ── Plotly Dark Template ──────────────────────────────────────────────────────

_ds_dark = go.layout.Template()
_ds_dark.layout = go.Layout(
    paper_bgcolor=COLORS["bg_primary"],
    plot_bgcolor=COLORS["bg_primary"],
    font=dict(color=COLORS["text_body"], family="Inter, sans-serif"),
    title=dict(font=dict(color=COLORS["text_bright"])),
    xaxis=dict(
        gridcolor="#2a2a4a",
        zerolinecolor="#2a2a4a",
        linecolor="#2a2a4a",
    ),
    yaxis=dict(
        gridcolor="#2a2a4a",
        zerolinecolor="#2a2a4a",
        linecolor="#2a2a4a",
    ),
    colorway=[
        COLORS["accent_primary"],
        COLORS["accent_secondary"],
        COLORS["info"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["error"],
        "#1abc9c",
        "#e67e22",
        "#9b59b6",
        "#2c3e50",
    ],
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
pio.templates["ds_dark"] = _ds_dark
pio.templates.default = "ds_dark"


# ── Global CSS Injection ──────────────────────────────────────────────────────

def inject_global_css():
    """Inject comprehensive dark-theme CSS into the Streamlit page."""
    st.markdown(
        """
        <style>
        /* ── Font ───────────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Sidebar ────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #16213e 0%, #0e1117 100%);
        }
        section[data-testid="stSidebar"] .stRadio label,
        section[data-testid="stSidebar"] a {
            color: #a0a0b8 !important;
            transition: color 0.2s;
        }
        section[data-testid="stSidebar"] .stRadio label:hover,
        section[data-testid="stSidebar"] a:hover {
            color: #667eea !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: linear-gradient(90deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.10) 100%);
            border-left: 3px solid #667eea;
        }

        /* ── Metric Cards ───────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102,126,234,0.15);
        }
        [data-testid="stMetric"] label {
            color: #a0a0b8 !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 700;
        }

        /* ── Tabs (pill style) ──────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 20px;
            color: #a0a0b8;
            background-color: #1a1a2e;
            border: 1px solid #2a2a4a;
            transition: all 0.2s;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #ffffff;
            background-color: #262640;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            color: #ffffff !important;
            border-color: transparent !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        /* ── Buttons ────────────────────────────────────────────────── */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: #ffffff;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(102,126,234,0.25);
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(102,126,234,0.4);
            opacity: 0.95;
        }
        .stButton > button:active {
            transform: translateY(0);
        }

        /* ── Download Button ────────────────────────────────────────── */
        .stDownloadButton > button {
            background: transparent;
            color: #667eea;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stDownloadButton > button:hover {
            background: rgba(102,126,234,0.1);
            transform: translateY(-1px);
        }

        /* ── Dataframe ──────────────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border: 1px solid #2a2a4a;
            border-radius: 10px;
            overflow: hidden;
        }

        /* ── Inputs / Selects ───────────────────────────────────────── */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            border-radius: 8px !important;
            transition: border-color 0.2s;
        }
        .stSelectbox > div > div:hover,
        .stMultiSelect > div > div:hover,
        .stTextInput > div > div > input:hover,
        .stNumberInput > div > div > input:hover {
            border-color: #667eea !important;
        }

        /* ── Expander ───────────────────────────────────────────────── */
        .streamlit-expanderHeader {
            background: #1a1a2e;
            border-radius: 8px;
            border: 1px solid #2a2a4a;
            color: #e0e0e0;
            font-weight: 600;
        }

        /* ── File Uploader ──────────────────────────────────────────── */
        [data-testid="stFileUploader"] {
            border: 2px dashed #2a2a4a;
            border-radius: 12px;
            padding: 1rem;
            transition: border-color 0.2s;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #667eea;
        }

        /* ── Alert Boxes ────────────────────────────────────────────── */
        .stAlert [data-testid="stNotification"] {
            border-radius: 10px;
        }

        /* ── Divider ────────────────────────────────────────────────── */
        hr {
            border-color: #2a2a4a !important;
        }

        /* ── Progress Bar ───────────────────────────────────────────── */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }

        /* ── Scrollbar ──────────────────────────────────────────────── */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0e1117;
        }
        ::-webkit-scrollbar-thumb {
            background: #2a2a4a;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #667eea;
        }

        /* ── Hero Section (home page) ───────────────────────────────── */
        .hero-section {
            text-align: center;
            padding: 2rem 0 1.5rem 0;
        }
        .hero-badge {
            display: inline-block;
            background: linear-gradient(90deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
            border: 1px solid rgba(102,126,234,0.3);
            border-radius: 20px;
            padding: 0.3rem 1rem;
            font-size: 0.85rem;
            color: #667eea;
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.5rem 0;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 1.15rem;
            color: #a0a0b8;
            margin-top: 0.5rem;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
            text-align: center;
        }

        /* ── Tool Cards (home page) ─────────────────────────────────── */
        .tool-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 14px;
            padding: 1.5rem;
            border: 1px solid #2a2a4a;
            min-height: 180px;
            transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
        }
        .tool-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(102,126,234,0.15);
            border-color: #667eea;
        }
        .tool-card h3 {
            margin-top: 0;
            color: #ffffff;
            font-size: 1.05rem;
        }
        .tool-card p {
            color: #a0a0b8;
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 0;
        }
        .tool-card-icon {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            display: block;
        }

        /* ── Footer ─────────────────────────────────────────────────── */
        .app-footer {
            text-align: center;
            color: #555;
            font-size: 0.85rem;
            padding: 2rem 0 1rem 0;
        }
        .app-footer span {
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600;
        }

        /* ── Statistics Results (dark theme) ───────────────────────── */
        .p-significant {
            color: #2ecc71;
            font-weight: 700;
        }
        .p-not-significant {
            color: #e74c3c;
            font-weight: 500;
        }
        .p-marginal {
            color: #f39c12;
            font-weight: 500;
        }
        .assumption-pass {
            background-color: rgba(46,204,113,0.15);
            color: #2ecc71;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-weight: 500;
            font-size: 0.85rem;
            display: inline-block;
        }
        .assumption-fail {
            background-color: rgba(231,76,60,0.15);
            color: #e74c3c;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-weight: 500;
            font-size: 0.85rem;
            display: inline-block;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }
        .results-table th {
            background-color: #1a1a2e;
            color: #e0e0e0;
            padding: 0.6rem 1rem;
            text-align: left;
            font-weight: 500;
        }
        .results-table td {
            padding: 0.5rem 1rem;
            border-bottom: 1px solid #2a2a4a;
            color: #e0e0e0;
        }
        .results-table tr:nth-child(even) {
            background-color: rgba(26,26,46,0.5);
        }
        .info-box {
            background-color: rgba(102,126,234,0.1);
            border-left: 4px solid #667eea;
            padding: 1rem;
            border-radius: 0 6px 6px 0;
            margin: 1rem 0;
            color: #e0e0e0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Page Header ───────────────────────────────────────────────────────────────

def page_header(title: str, description: str, icon: str = ""):
    """Render a gradient page title with accent underline and description."""
    title = html.escape(title)
    description = html.escape(description)
    icon = html.escape(icon)
    icon_html = f'<span style="font-size:2rem;margin-right:0.5rem;">{icon}</span>' if icon else ""
    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
                {icon_html}
                <h1 style="
                    font-size:2rem;
                    font-weight:700;
                    background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                    margin:0;
                ">{title}</h1>
            </div>
            <div style="
                width:60px;height:3px;
                background:linear-gradient(90deg,#667eea,#764ba2);
                border-radius:2px;
                margin:0.5rem 0;
            "></div>
            <p style="color:#a0a0b8;font-size:1rem;margin:0;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Metric Card ───────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, icon: str = ""):
    """Render a styled metric card with optional icon."""
    label = html.escape(label)
    value = html.escape(value)
    icon = html.escape(icon)
    icon_html = f'<span style="font-size:1.5rem;margin-right:0.5rem;">{icon}</span>' if icon else ""
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
            border:1px solid #2a2a4a;
            border-radius:12px;
            padding:1rem 1.2rem;
            text-align:center;
            transition:transform 0.2s,box-shadow 0.2s;
        ">
            <div>{icon_html}</div>
            <div style="color:#a0a0b8;font-size:0.85rem;margin-bottom:0.25rem;">{label}</div>
            <div style="color:#ffffff;font-size:1.5rem;font-weight:700;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Matplotlib Dark Theme ────────────────────────────────────────────────────

def set_matplotlib_dark():
    """Configure matplotlib rcParams for dark-themed SHAP/PDP plots."""
    import matplotlib.pyplot as plt

    dark_params = {
        "figure.facecolor": COLORS["bg_primary"],
        "axes.facecolor": COLORS["bg_primary"],
        "axes.edgecolor": "#2a2a4a",
        "axes.labelcolor": COLORS["text_body"],
        "text.color": COLORS["text_body"],
        "xtick.color": COLORS["text_muted"],
        "ytick.color": COLORS["text_muted"],
        "grid.color": "#2a2a4a",
        "legend.facecolor": COLORS["bg_card"],
        "legend.edgecolor": "#2a2a4a",
        "savefig.facecolor": COLORS["bg_primary"],
    }
    plt.rcParams.update(dark_params)

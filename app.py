"""DS Power Tools — Combined Data Science & Statistics Toolkit.
Entry point: navigation, styling, state initialization.
"""

import streamlit as st

st.set_page_config(
    page_title="DS Power Tools",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme selector (must run before CSS injection) ────────────────────────────
if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "Light"

theme_choice = st.sidebar.selectbox(
    "Theme",
    ["Light", "Dark"],
    index=["Light", "Dark"].index(st.session_state["app_theme"]),
    key="_theme_selector",
)

if theme_choice != st.session_state["app_theme"]:
    st.session_state["app_theme"] = theme_choice
    st.rerun()

# ── AI Settings ──────────────────────────────────────────────────────────────
with st.sidebar.expander("AI Settings", expanded=False):
    _prev_key = st.session_state.get("gemini_api_key") or ""
    _gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=_prev_key,
        help=(
            "Optional. Enter a Google Gemini API key for AI-powered "
            "interpretations. You can also set GEMINI_API_KEY in "
            ".streamlit/secrets.toml or as an environment variable."
        ),
        key="_gemini_key_input",
    )
    if _gemini_key != _prev_key:
        st.session_state["gemini_api_key"] = _gemini_key if _gemini_key else None
        st.session_state["_gemini_cache"] = {}

    from core.gemini import is_api_available
    if is_api_available():
        st.success("Gemini API connected", icon="✅")
    else:
        st.caption("No API key — clipboard fallback mode")

# ── Styling & state init ─────────────────────────────────────────────────────
from utils.theme import inject_global_css, register_plotly_theme, get_colors
from core.state import init_state

register_plotly_theme()
inject_global_css()
init_state()

# ── Sidebar branding ─────────────────────────────────────────────────────────
c = get_colors()
accent_grad = f"linear-gradient(90deg,{c['title_gradient_start']} 0%,{c['title_gradient_end']} 100%)"

st.sidebar.markdown(
    f"""
    <div style="text-align:center;padding:1.5rem 1rem 1rem 1rem;border-bottom:1px solid {c['border']};margin-bottom:1rem;">
        <div style="
            font-size:1.6rem;
            font-weight:700;
            color:{c['accent_primary']};
        ">DS Power Tools</div>
        <p style="color:{c['text_muted']};font-size:0.8rem;margin-top:0.25rem;">
            Data Science &amp; Statistics Toolkit
        </p>
        <p style="color:{c['text_muted']};font-size:0.75rem;font-style:italic;">
            Created by Scott Severance
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Import page modules
from pages import home
from pages import data_profiler
from pages import smart_cleaning
from pages import feature_engineering
from pages import feature_selection
from pages import class_imbalance
from pages import model_arena
from pages import hyperparameter_tuning
from pages import explainability
from pages import data_drift
from pages import data_input
from pages import descriptive_stats
from pages import one_sample_ttest
from pages import independent_ttest
from pages import paired_ttest
from pages import oneway_anova
from pages import twoway_anova
from pages import repeated_anova
from pages import mixed_anova
from pages import mann_whitney
from pages import wilcoxon
from pages import kruskal_wallis
from pages import friedman
from pages import pearson_correlation
from pages import spearman_correlation
from pages import linear_regression
from pages import logistic_regression
from pages import manova
from pages import multivariate_regression
from pages import chi_squared_test
from pages import binomial_test
from pages import survival_analysis
from pages import export_report

# Navigation with collapsible sections
def _page(func, title, url_path, icon, **kwargs):
    """Create a st.Page and register it in the page map for card navigation."""
    p = st.Page(func, title=title, icon=icon, url_path=url_path, **kwargs)
    _page_map[url_path] = p
    return p

_page_map = {}

pages = {
    "": [
        _page(home.render, "Home", "home", "⚡", default=True),
    ],
    "Data Science Tools": [
        _page(data_profiler.render, "Data Profiler", "data-profiler", "📊"),
        _page(smart_cleaning.render, "Smart Cleaning", "smart-cleaning", "🧹"),
        _page(feature_engineering.render, "Feature Engineering", "feature-engineering", "🔧"),
        _page(feature_selection.render, "Feature Selection", "feature-selection", "🎯"),
        _page(class_imbalance.render, "Class Imbalance", "class-imbalance", "⚖️"),
        _page(model_arena.render, "Model Arena", "model-arena", "🏟️"),
        _page(hyperparameter_tuning.render, "Hyperparameter Tuning", "hyperparameter-tuning", "🎛️"),
        _page(explainability.render, "Explainability", "explainability", "🔍"),
        _page(data_drift.render, "Data Drift", "data-drift", "📈"),
    ],
    "Statistics Tools": [
        _page(data_input.render, "Data Input", "stats-data-input", "📋"),
        _page(descriptive_stats.render, "Descriptive Statistics", "descriptive", "📈"),
        _page(one_sample_ttest.render, "One-Sample t-Test", "one-sample-ttest", "1️⃣"),
        _page(independent_ttest.render, "Independent t-Test", "independent-ttest", "↔️"),
        _page(paired_ttest.render, "Paired t-Test", "paired-ttest", "🔗"),
        _page(oneway_anova.render, "One-Way ANOVA", "oneway-anova", "📊"),
        _page(twoway_anova.render, "Two-Way ANOVA", "twoway-anova", "📊"),
        _page(repeated_anova.render, "Repeated Measures ANOVA", "repeated-anova", "🔄"),
        _page(mixed_anova.render, "Mixed ANOVA", "mixed-anova", "🔀"),
        _page(manova.render, "MANOVA", "manova", "📊"),
        _page(mann_whitney.render, "Mann-Whitney U", "mann-whitney", "📉"),
        _page(wilcoxon.render, "Wilcoxon Signed-Rank", "wilcoxon", "📉"),
        _page(kruskal_wallis.render, "Kruskal-Wallis", "kruskal-wallis", "📉"),
        _page(friedman.render, "Friedman Test", "friedman", "📉"),
        _page(pearson_correlation.render, "Pearson Correlation", "pearson", "🔵"),
        _page(spearman_correlation.render, "Spearman Correlation", "spearman", "🔵"),
        _page(linear_regression.render, "Linear Regression", "linear-regression", "📐"),
        _page(logistic_regression.render, "Logistic Regression", "logistic-regression", "📐"),
        _page(multivariate_regression.render, "Multivariate Regression", "multivariate-regression", "📐"),
        _page(chi_squared_test.render, "Chi-Squared Test", "chi-squared", "🔲"),
        _page(binomial_test.render, "Binomial Test", "binomial", "🎯"),
        _page(survival_analysis.render, "Survival Analysis", "survival-analysis", "⏳"),
    ],
    "Reports": [
        _page(export_report.render, "Report Builder", "report-builder", "📄"),
    ],
}

st.session_state["_page_map"] = _page_map

nav = st.navigation(pages)
nav.run()

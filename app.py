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
    st.session_state["app_theme"] = "Dark"

theme_choice = st.sidebar.selectbox(
    "Theme",
    ["Dark", "Retro Light"],
    index=["Dark", "Retro Light"].index(st.session_state["app_theme"]),
    key="_theme_selector",
)

if theme_choice != st.session_state["app_theme"]:
    st.session_state["app_theme"] = theme_choice
    st.rerun()

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
            background:{accent_grad};
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
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
from pages import chi_squared_test
from pages import binomial_test

# Navigation with collapsible sections
pages = {
    "": [
        st.Page(home.render, title="Home", icon="⚡", default=True, url_path="home"),
    ],
    "Data Science Tools": [
        st.Page(data_profiler.render, title="Data Profiler", icon="📊", url_path="data-profiler"),
        st.Page(smart_cleaning.render, title="Smart Cleaning", icon="🧹", url_path="smart-cleaning"),
        st.Page(feature_engineering.render, title="Feature Engineering", icon="🔧", url_path="feature-engineering"),
        st.Page(feature_selection.render, title="Feature Selection", icon="🎯", url_path="feature-selection"),
        st.Page(class_imbalance.render, title="Class Imbalance", icon="⚖️", url_path="class-imbalance"),
        st.Page(model_arena.render, title="Model Arena", icon="🏟️", url_path="model-arena"),
        st.Page(hyperparameter_tuning.render, title="Hyperparameter Tuning", icon="🎛️", url_path="hyperparameter-tuning"),
        st.Page(explainability.render, title="Explainability", icon="🔍", url_path="explainability"),
        st.Page(data_drift.render, title="Data Drift", icon="📈", url_path="data-drift"),
    ],
    "Statistics Tools": [
        st.Page(data_input.render, title="Data Input", icon="📋", url_path="stats-data-input"),
        st.Page(descriptive_stats.render, title="Descriptive Statistics", icon="📈", url_path="descriptive"),
        st.Page(one_sample_ttest.render, title="One-Sample t-Test", icon="1️⃣", url_path="one-sample-ttest"),
        st.Page(independent_ttest.render, title="Independent t-Test", icon="↔️", url_path="independent-ttest"),
        st.Page(paired_ttest.render, title="Paired t-Test", icon="🔗", url_path="paired-ttest"),
        st.Page(oneway_anova.render, title="One-Way ANOVA", icon="📊", url_path="oneway-anova"),
        st.Page(twoway_anova.render, title="Two-Way ANOVA", icon="📊", url_path="twoway-anova"),
        st.Page(repeated_anova.render, title="Repeated Measures ANOVA", icon="🔄", url_path="repeated-anova"),
        st.Page(mixed_anova.render, title="Mixed ANOVA", icon="🔀", url_path="mixed-anova"),
        st.Page(mann_whitney.render, title="Mann-Whitney U", icon="📉", url_path="mann-whitney"),
        st.Page(wilcoxon.render, title="Wilcoxon Signed-Rank", icon="📉", url_path="wilcoxon"),
        st.Page(kruskal_wallis.render, title="Kruskal-Wallis", icon="📉", url_path="kruskal-wallis"),
        st.Page(friedman.render, title="Friedman Test", icon="📉", url_path="friedman"),
        st.Page(pearson_correlation.render, title="Pearson Correlation", icon="🔵", url_path="pearson"),
        st.Page(spearman_correlation.render, title="Spearman Correlation", icon="🔵", url_path="spearman"),
        st.Page(linear_regression.render, title="Linear Regression", icon="📐", url_path="linear-regression"),
        st.Page(logistic_regression.render, title="Logistic Regression", icon="📐", url_path="logistic-regression"),
        st.Page(chi_squared_test.render, title="Chi-Squared Test", icon="🔲", url_path="chi-squared"),
        st.Page(binomial_test.render, title="Binomial Test", icon="🎯", url_path="binomial"),
    ],
}

nav = st.navigation(pages)
nav.run()

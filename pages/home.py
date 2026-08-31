"""Home page — hero section, prominent data import, quick stats, tool cards."""

import streamlit as st
import pandas as pd
import numpy as np
from core.state import set_var_type, clear_result_caches
from utils.theme import get_colors, hex_to_rgb, _is_light


def _auto_detect_var_types(df):
    """Auto-detect variable types (Metric/Nominal/Ordinal) for statistics tools."""
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            set_var_type(col, "Metric")
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        non_null_numeric = numeric.notna().sum()
        if non_null_numeric / len(series) > 0.5:
            n_unique = numeric.dropna().nunique()
            if n_unique <= 2:
                set_var_type(col, "Nominal")
            elif n_unique <= 7 and n_unique < len(series) * 0.3:
                set_var_type(col, "Ordinal")
            else:
                set_var_type(col, "Metric")
        else:
            set_var_type(col, "Nominal")


def _render_tool_cards(tools, c, card_grad, prefix):
    """Render tool cards in a 3-column grid with clickable navigation buttons."""
    page_map = st.session_state.get("_page_map", {})
    rows = [tools[i:i + 3] for i in range(0, len(tools), 3)]
    for row_idx, row in enumerate(rows):
        # Render the row as a single CSS-grid HTML block so all cards share equal height
        row_html = ""
        for icon, title, desc, _url in row:
            row_html += (
                f'<div style="background:{card_grad};border-radius:14px;padding:1.25rem;'
                f'border:1px solid {c["border"]};">'
                f'<span style="font-size:1.8rem;display:block;margin-bottom:0.5rem;">{icon}</span>'
                f'<h3 style="margin-top:0;color:{c["text_bright"]};font-size:1.05rem;">{title}</h3>'
                f'<p style="color:{c["text_muted"]};font-size:0.9rem;line-height:1.5;margin-bottom:0;">{desc}</p>'
                f'</div>'
            )
        ncols = len(row)
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat({ncols},1fr);gap:1rem;'
            f'align-items:stretch;margin-bottom:0.25rem;">{row_html}</div>',
            unsafe_allow_html=True,
        )
        # Navigation buttons below the cards
        cols = st.columns(3)
        for col_idx, (_, title, _, url_path) in enumerate(row):
            with cols[col_idx]:
                page_obj = page_map.get(url_path)
                if page_obj is not None:
                    if st.button(f"Open {title}", key=f"{prefix}_{row_idx}_{col_idx}", width="stretch"):
                        st.switch_page(page_obj)


def render():
    st.markdown("""<style>
.block-container {
    padding-top: 1rem;
}
</style>""", unsafe_allow_html=True)
    c = get_colors()
    light = _is_light()
    accent_grad = f"linear-gradient(90deg,{c['title_gradient_start']} 0%,{c['title_gradient_end']} 100%)"
    card_grad = f"linear-gradient(135deg,{c['bg_card']} 0%,{c['bg_card']} 100%)" if light else f"linear-gradient(135deg,{c['bg_card']} 0%,{c['bg_sidebar']} 100%)"
    accent_bg_start = f"rgba({hex_to_rgb(c['title_gradient_start'])},0.12)"
    accent_bg_end = f"rgba({hex_to_rgb(c['title_gradient_end'])},0.08)"
    accent_border = f"rgba({hex_to_rgb(c['accent_primary'])},0.3)"

    # ── Hero Section ──────────────────────────────────────────────────────
    # Hero image — change the path below to display your own logo/banner.
    # See README.md "Customizing the Hero Image" for details.
    HERO_IMAGE = "assets/hero.png"

    import os
    if os.path.exists(HERO_IMAGE):
        _hero_col1, _hero_col2, _hero_col3 = st.columns([1, 2, 1])
        with _hero_col2:
            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
            with open(HERO_IMAGE, "rb") as _f:
                _hero_bytes = _f.read()
            st.image(_hero_bytes, width="stretch")

    st.markdown(
        f"""
        <div style="text-align:center; padding-top: 0rem; padding-bottom: 0; margin-top: 0rem;">
            <div class="hero-badge">⚡ Open-Source Data Science &amp; Statistics Toolkit</div>
            <div class="hero-title">DS Power Tools</div>
            <p style="font-size:1.15rem;color:{c['text_muted']};margin:0.5rem auto 0 auto;max-width:700px;text-align:center;">
                Eliminate the hardest parts of data science — automated profiling,
                cleaning, feature engineering, model selection, explainability,
                and a full suite of statistical tests.
            </p>
            <p style="font-size:0.95rem;color:{c['text_muted']};margin:0.75rem auto 0 auto;font-style:italic;">
                Created by Scott Severance
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Data Import (shared across ALL tools) ─────────────────────────────
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,{accent_bg_start} 0%,{accent_bg_end} 100%);
            border:1px solid {accent_border};
            border-radius:14px;
            padding:1.5rem 2rem;
            margin-bottom:1.5rem;
        ">
            <h2 style="
                margin:0 0 0.25rem 0;
                color:{c['accent_primary']};
                font-size:1.5rem;
            ">Import Your Data</h2>
            <p style="color:{c['text_muted']};margin:0;font-size:0.95rem;">
                Upload a dataset here to use across <strong style="color:{c['text_body']};">all</strong> Data Science
                and Statistics tools. You can also enter data manually via
                <strong style="color:{c['text_body']};">Statistics Tools &gt; Data Input</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop a CSV or Excel file to get started",
        type=["csv", "xlsx", "xls"],
        help="Your data stays local — nothing is sent to any server. Max 200 MB.",
    )

    if uploaded_file is not None:
        MAX_SIZE_MB = 200
        if uploaded_file.size > MAX_SIZE_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_SIZE_MB} MB limit. Please upload a smaller file.")
        else:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file, low_memory=False)
                else:
                    df = pd.read_excel(uploaded_file)

                if df.shape[0] > 500_000 or df.shape[1] > 500:
                    st.error(
                        f"Dataset too large ({df.shape[0]:,} rows x {df.shape[1]} columns). "
                        "Maximum supported size is 500,000 rows and 500 columns."
                    )
                else:
                    st.session_state["df"] = df
                    st.session_state["original_df"] = df.copy()
                    st.session_state["file_name"] = uploaded_file.name
                    clear_result_caches()
                    _auto_detect_var_types(df)
                    st.success(
                        f"Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows x {df.shape[1]} columns. "
                        "Data is now available in all Data Science and Statistics tools."
                    )
            except Exception:
                st.error("Error loading file. Please check that the file is a valid CSV or Excel document.")

    # ── Quick Stats ───────────────────────────────────────────────────────
    if "df" in st.session_state and not st.session_state["df"].dropna(how="all").empty:
        df = st.session_state["df"]

        st.subheader("Quick Overview")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with c2:
            st.metric("Columns", f"{df.shape[1]}")
        with c3:
            st.metric("Numeric", f"{df.select_dtypes(include='number').shape[1]}")
        with c4:
            st.metric("Categorical", f"{df.select_dtypes(include=['object', 'category']).shape[1]}")
        with c5:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Missing %", f"{missing_pct:.1f}%")

        with st.expander("Preview Data", expanded=True):
            st.dataframe(df.head(50), width="stretch", height=300)

        with st.expander("Column Types & Info"):
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Null": df.isnull().sum().values,
                "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
                "Unique": df.nunique().values,
            })
            st.dataframe(info_df, width="stretch", hide_index=True)

        # AI-recommended analysis plan
        from components.ai_advisor import render_data_plan
        _col_summaries = []
        for col in df.columns:
            _col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique()),
                "n_missing": int(df[col].isnull().sum()),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                _col_info["mean"] = float(df[col].mean()) if df[col].notna().any() else None
                _col_info["std"] = float(df[col].std()) if df[col].notna().any() else None
                _col_info["min"] = float(df[col].min()) if df[col].notna().any() else None
                _col_info["max"] = float(df[col].max()) if df[col].notna().any() else None
            _col_summaries.append(_col_info)

        render_data_plan(
            df_summary={
                "columns": _col_summaries,
                "n_rows": df.shape[0],
                "n_cols": df.shape[1],
            },
            page_key="home",
        )

        st.divider()

    # ── Tool Cards ────────────────────────────────────────────────────────
    st.subheader("Available Tools")
    st.caption("Upload data above, then navigate to any tool from the sidebar.")

    st.markdown("#### Data Science Tools")
    ds_tools = [
        ("📊", "Data Profiler", "Deep automated EDA — distributions, correlations, anomalies, missing patterns.", "data-profiler"),
        ("🧹", "Smart Cleaning", "One-click missing value imputation, outlier treatment, encoding, deduplication.", "smart-cleaning"),
        ("🔧", "Feature Engineering", "Auto-generate polynomial, interaction, datetime & binned features.", "feature-engineering"),
        ("🎯", "Feature Selection", "Correlation filters, mutual info, variance threshold, RFE — ranked results.", "feature-selection"),
        ("⚖️", "Class Imbalance", "Detect skew, apply SMOTE / random over/under-sampling, compare distributions.", "class-imbalance"),
        ("🏟️", "Model Arena", "Benchmark 10+ algorithms side-by-side with proper cross-validation.", "model-arena"),
        ("🎛️", "Hyperparameter Tuning", "Bayesian optimization via Optuna with live trial visualizations.", "hyperparameter-tuning"),
        ("🔍", "Explainability", "SHAP values, feature importance, partial dependence — for any model.", "explainability"),
        ("📈", "Data Drift", "Upload a reference & current dataset — detect drift with statistical tests.", "data-drift"),
    ]

    _render_tool_cards(ds_tools, c, card_grad, "ds")

    st.markdown("#### Statistics Tools")
    stats_tools = [
        ("📋", "Data Input", "Enter data manually, upload CSV/Excel, or paste from a spreadsheet.", "stats-data-input"),
        ("📈", "Descriptive Stats", "Mean, median, standard deviation, skewness, kurtosis, and more.", "descriptive"),
        ("1️⃣", "One-Sample t-Test", "Test whether a sample mean differs from a hypothesized value.", "one-sample-ttest"),
        ("↔️", "Independent t-Test", "Compare means between two independent groups.", "independent-ttest"),
        ("🔗", "Paired t-Test", "Compare means of two related measurements.", "paired-ttest"),
        ("📊", "One-Way ANOVA", "Compare means across three or more independent groups.", "oneway-anova"),
        ("📊", "Two-Way ANOVA", "Test effects of two factors and their interaction.", "twoway-anova"),
        ("🔄", "Repeated Measures ANOVA", "Compare means across related conditions (within-subjects).", "repeated-anova"),
        ("🔀", "Mixed ANOVA", "Test within-subjects and between-subjects factors together.", "mixed-anova"),
        ("📊", "MANOVA", "Test group differences across multiple dependent variables simultaneously.", "manova"),
        ("📉", "Mann-Whitney U", "Non-parametric alternative to the independent t-test.", "mann-whitney"),
        ("📉", "Wilcoxon Signed-Rank", "Non-parametric alternative to the paired t-test.", "wilcoxon"),
        ("📉", "Kruskal-Wallis", "Non-parametric alternative to one-way ANOVA.", "kruskal-wallis"),
        ("📉", "Friedman Test", "Non-parametric alternative to repeated measures ANOVA.", "friedman"),
        ("🔵", "Pearson Correlation", "Measure the linear relationship between two variables.", "pearson"),
        ("🔵", "Spearman Correlation", "Measure the monotonic relationship using ranks.", "spearman"),
        ("📐", "Linear Regression", "Predict a continuous outcome from predictors (OLS).", "linear-regression"),
        ("📐", "Logistic Regression", "Predict a binary outcome from predictors.", "logistic-regression"),
        ("📐", "Multivariate Regression", "Predict multiple outcomes from predictor variables.", "multivariate-regression"),
        ("🔲", "Chi-Squared Test", "Test association between two categorical variables.", "chi-squared"),
        ("🎯", "Binomial Test", "Test whether a proportion matches a hypothesized value.", "binomial"),
    ]

    _render_tool_cards(stats_tools, c, card_grad, "stats")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Welcome to DS Power Tools

This is the **Home** page — your central hub for importing data and navigating to all available tools.

---

#### Hero Section
- Displays the app title, tagline, and a brief description of what DS Power Tools offers: automated profiling, cleaning, feature engineering, model selection, explainability, and statistical testing.

#### Import Data
- **Upload a CSV or Excel file** (`.csv`, `.xlsx`, `.xls`) using the file uploader.
- **200 MB size limit** per file; datasets are capped at **500,000 rows** and **500 columns**.
- Once uploaded, your data is stored in the session and **shared across all Data Science and Statistics tools** — no need to re-upload on each page.
- Variable types (Metric, Nominal, Ordinal) are **auto-detected** for use in the Statistics tools.
- You can also enter data manually via **Statistics Tools > Data Input**.

#### Quick Overview
- After a dataset is loaded, five summary metrics appear:
  - **Rows** — total number of observations
  - **Columns** — total number of features
  - **Numeric** — count of numeric (int/float) columns
  - **Categorical** — count of object/category columns
  - **Missing %** — percentage of all cells that contain null values
- **Preview Data** — shows the first 50 rows of your dataset in an interactive table.
- **Column Types & Info** — lists every column with its data type, non-null count, null count, null percentage, and number of unique values.

#### Tool Cards
- **Data Science Tools** — nine tools covering the full ML pipeline:
  - Data Profiler, Smart Cleaning, Feature Engineering, Feature Selection, Class Imbalance, Model Arena, Hyperparameter Tuning, Explainability, and Data Drift.
- **Statistics Tools** — eight tools for classical statistical analysis:
  - Data Input, Descriptive Stats, t-Tests, ANOVA, Non-Parametric Tests, Correlation, Regression, and Other Tests (Chi-squared, Binomial).
- Navigate to any tool from the **sidebar** on the left.
        """)

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="app-footer"><span>DS Power Tools</span> — built to eliminate '
        "the 80% of data science that isn't data science.</div>",
        unsafe_allow_html=True,
    )

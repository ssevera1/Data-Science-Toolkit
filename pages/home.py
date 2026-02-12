"""Home page — hero section, file upload for DS tools, quick stats, tool cards."""

import streamlit as st
import pandas as pd
import numpy as np


def render():
    # ── Hero Section ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:2rem 0 1.5rem 0;">
            <div class="hero-badge">⚡ Open-Source Data Science &amp; Statistics Toolkit</div>
            <div class="hero-title">DS Power Tools</div>
            <p style="font-size:1.15rem;color:#a0a0b8;margin:0.5rem auto 0 auto;max-width:700px;text-align:center;">
                Eliminate the hardest parts of data science — automated profiling,
                cleaning, feature engineering, model selection, explainability,
                and a full suite of statistical tests.
            </p>
            <p style="font-size:0.95rem;color:#a0a0b8;margin:0.75rem auto 0 auto;font-style:italic;">
                Created by Scott Severance
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── File Upload (for Data Science tools) ──────────────────────────────
    st.subheader("Upload Your Dataset")
    st.caption("Upload data here for Data Science tools. Statistics tools have their own data input page.")

    uploaded_file = st.file_uploader(
        "Drop a CSV or Excel file to get started",
        type=["csv", "xlsx", "xls"],
        help="Your data stays local — nothing is sent to any server.",
    )

    if uploaded_file is not None:
        MAX_SIZE_MB = 200
        if uploaded_file.size > MAX_SIZE_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_SIZE_MB} MB limit. Please upload a smaller file.")
        else:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
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
                    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows x {df.shape[1]} columns")
            except Exception:
                st.error("Error loading file. Please check that the file is a valid CSV or Excel document.")

    # ── Quick Stats ───────────────────────────────────────────────────────
    if "df" in st.session_state:
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
            st.dataframe(df.head(50), use_container_width=True, height=300)

        with st.expander("Column Types & Info"):
            info_df = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.astype(str).values,
                "Non-Null": df.notnull().sum().values,
                "Null": df.isnull().sum().values,
                "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
                "Unique": df.nunique().values,
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)

        st.divider()

    # ── Tool Cards ────────────────────────────────────────────────────────
    st.subheader("Available Tools")
    st.caption("Upload data above, then navigate to any tool from the sidebar.")

    st.markdown("#### Data Science Tools")
    ds_tools = [
        ("📊", "Data Profiler", "Deep automated EDA — distributions, correlations, anomalies, missing patterns."),
        ("🧹", "Smart Cleaning", "One-click missing value imputation, outlier treatment, encoding, deduplication."),
        ("🔧", "Feature Engineering", "Auto-generate polynomial, interaction, datetime & binned features."),
        ("🎯", "Feature Selection", "Correlation filters, mutual info, variance threshold, RFE — ranked results."),
        ("⚖️", "Class Imbalance", "Detect skew, apply SMOTE / random over/under-sampling, compare distributions."),
        ("🏟️", "Model Arena", "Benchmark 10+ algorithms side-by-side with proper cross-validation."),
        ("🎛️", "Hyperparameter Tuning", "Bayesian optimization via Optuna with live trial visualizations."),
        ("🔍", "Explainability", "SHAP values, feature importance, partial dependence — for any model."),
        ("📈", "Data Drift", "Upload a reference & current dataset — detect drift with statistical tests."),
    ]

    cards_html = ""
    for icon, title, desc in ds_tools:
        cards_html += f"""
        <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:14px;padding:1.5rem;border:1px solid #2a2a4a;">
            <span style="font-size:1.8rem;display:block;margin-bottom:0.5rem;">{icon}</span>
            <h3 style="margin-top:0;color:#ffffff;font-size:1.05rem;">{title}</h3>
            <p style="color:#a0a0b8;font-size:0.9rem;line-height:1.5;margin-bottom:0;">{desc}</p>
        </div>"""

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">{cards_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Statistics Tools")
    stats_tools = [
        ("📋", "Data Input", "Enter data manually, upload CSV/Excel, or paste from a spreadsheet."),
        ("📈", "Descriptive Stats", "Mean, median, standard deviation, skewness, kurtosis, and more."),
        ("🧪", "t-Tests", "One-sample, independent, and paired t-tests with effect sizes."),
        ("📊", "ANOVA", "One-way, two-way, repeated measures, and mixed ANOVA."),
        ("📉", "Non-Parametric", "Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman tests."),
        ("🔵", "Correlation", "Pearson and Spearman correlation with scatter plots."),
        ("📐", "Regression", "Linear (OLS) and logistic regression with diagnostics."),
        ("🔲", "Other Tests", "Chi-squared test of independence and binomial test."),
    ]

    cards_html = ""
    for icon, title, desc in stats_tools:
        cards_html += f"""
        <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border-radius:14px;padding:1.5rem;border:1px solid #2a2a4a;">
            <span style="font-size:1.8rem;display:block;margin-bottom:0.5rem;">{icon}</span>
            <h3 style="margin-top:0;color:#ffffff;font-size:1.05rem;">{title}</h3>
            <p style="color:#a0a0b8;font-size:0.9rem;line-height:1.5;margin-bottom:0;">{desc}</p>
        </div>"""

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;">{cards_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="app-footer"><span>DS Power Tools</span> — built to eliminate '
        "the 80% of data science that isn't data science.</div>",
        unsafe_allow_html=True,
    )

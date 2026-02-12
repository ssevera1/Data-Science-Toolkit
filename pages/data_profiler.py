import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.theme import page_header, get_colors


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


def render():
    page_header("Data Profiler", "Automated exploratory data analysis — distributions, correlations, missing patterns, and outliers.", "📊")

    _guard()
    df = st.session_state["df"]

    tab_overview, tab_dist, tab_corr, tab_missing, tab_outliers = st.tabs(
        ["Overview", "Distributions", "Correlations", "Missing Values", "Outliers"]
    )

    # ── Overview ───────────────────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Dataset Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]}")
        num_dups = df.duplicated().sum()
        c3.metric("Duplicate Rows", f"{num_dups:,}")
        mem = df.memory_usage(deep=True).sum()
        c4.metric("Memory", f"{mem / 1024**2:.2f} MB")

        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe(include="all").T, width="stretch")

        st.markdown("#### Column Types")
        type_counts = df.dtypes.astype(str).value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig = px.bar(type_counts, x="Type", y="Count", color="Type", text_auto=True)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, width="stretch")

    # ── Distributions ──────────────────────────────────────────────────────────
    with tab_dist:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        st.subheader("Numeric Distributions")
        if num_cols:
            sel_num = st.multiselect("Select numeric columns", num_cols, default=num_cols[:6])
            if sel_num:
                ncols = min(3, len(sel_num))
                nrows = (len(sel_num) + ncols - 1) // ncols
                fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=sel_num)
                for i, col in enumerate(sel_num):
                    r, c = divmod(i, ncols)
                    fig.add_trace(
                        go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                        row=r + 1, col=c + 1,
                    )
                fig.update_layout(height=300 * nrows)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numeric columns found.")

        st.subheader("Categorical Distributions")
        if cat_cols:
            sel_cat = st.multiselect("Select categorical columns", cat_cols, default=cat_cols[:4])
            for col in sel_cat:
                vc = df[col].value_counts().head(20).reset_index()
                vc.columns = [col, "count"]
                fig = px.bar(vc, x=col, y="count", title=col, text_auto=True)
                fig.update_layout(height=350)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No categorical columns found.")

    # ── Correlations ───────────────────────────────────────────────────────────
    with tab_corr:
        st.subheader("Correlation Matrix")
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
            corr = num_df.corr(method=method)

            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                aspect="auto",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, width="stretch")

            st.markdown("#### Highly Correlated Pairs (|r| > 0.8)")
            pairs = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.8:
                        pairs.append({
                            "Feature A": corr.columns[i],
                            "Feature B": corr.columns[j],
                            "Correlation": round(val, 4),
                        })
            if pairs:
                st.dataframe(pd.DataFrame(pairs), width="stretch", hide_index=True)
            else:
                st.success("No highly correlated pairs found (|r| > 0.8).")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # ── Missing Values ─────────────────────────────────────────────────────────
    with tab_missing:
        st.subheader("Missing Value Analysis")
        miss = df.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)

        if len(miss) == 0:
            st.success("No missing values found!")
        else:
            miss_df = pd.DataFrame({
                "Column": miss.index,
                "Missing": miss.values,
                "% Missing": (miss.values / len(df) * 100).round(2),
            })
            st.dataframe(miss_df, width="stretch", hide_index=True)

            fig = px.bar(miss_df, x="Column", y="% Missing", color="% Missing",
                         color_continuous_scale="Reds", text_auto=".1f")
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

            # Missing value heatmap (sample if large)
            st.markdown("#### Missing Value Pattern")
            sample = df[miss.index].head(200)
            fig = px.imshow(
                sample.isnull().astype(int),
                color_continuous_scale=[get_colors()["bg_primary"], get_colors()["error"]],
                aspect="auto",
                labels=dict(color="Missing"),
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

    # ── Outliers ───────────────────────────────────────────────────────────────
    with tab_outliers:
        st.subheader("Outlier Detection (IQR Method)")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            iqr_mult = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
            outlier_summary = []
            for col in num_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_mult * iqr
                upper = q3 + iqr_mult * iqr
                n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                outlier_summary.append({
                    "Column": col,
                    "Q1": round(q1, 4),
                    "Q3": round(q3, 4),
                    "IQR": round(iqr, 4),
                    "Lower Bound": round(lower, 4),
                    "Upper Bound": round(upper, 4),
                    "Outliers": n_outliers,
                    "% Outliers": round(n_outliers / len(df) * 100, 2),
                })
            out_df = pd.DataFrame(outlier_summary)
            st.dataframe(out_df, width="stretch", hide_index=True)

            sel_box = st.multiselect("Box plots for", num_cols, default=num_cols[:6])
            if sel_box:
                fig = px.box(df[sel_box].melt(), x="variable", y="value", color="variable")
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numeric columns found.")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Data Profiler — Automated Exploratory Data Analysis

This page provides a comprehensive, automated EDA of your dataset across five tabs.

---

#### Overview Tab
- **Dataset Summary** — key metrics at a glance:
  - **Rows** and **Columns** — the shape of your data.
  - **Duplicate Rows** — number of exact duplicate rows detected.
  - **Memory** — approximate memory usage of the DataFrame in MB.
- **Descriptive Statistics** — a transposed summary table (`df.describe(include='all')`) showing count, mean, std, min, quartiles, max for numeric columns, and count, unique, top, freq for categorical columns.
- **Column Types** — a bar chart showing the distribution of data types (e.g., `int64`, `float64`, `object`) across all columns.

#### Distributions Tab
- **Numeric Distributions** — interactive **histograms** (via Plotly) for selected numeric columns. Select up to all numeric columns; charts are arranged in a 3-column grid.
- **Categorical Distributions** — **bar charts** showing the top 20 most frequent values for each selected categorical column.

#### Correlations Tab
- **Correlation Matrix** — a heatmap of pairwise correlations between all numeric columns.
  - Choose from **Pearson** (linear), **Spearman** (rank-based), or **Kendall** (concordance-based) methods.
  - Color scale ranges from -1 (strong negative) to +1 (strong positive).
- **Highly Correlated Pairs** — automatically identifies all feature pairs where **|r| > 0.8**, which may indicate multicollinearity. Consider removing one feature from each highly correlated pair before modeling.

#### Missing Values Tab
- **Missing Value Summary** — a table listing each column with missing data, the count, and the percentage of missing values.
- **Percentage Bar Chart** — a visual bar chart colored by severity (using a red color scale).
- **Missing Value Pattern Heatmap** — shows the first 200 rows of columns with missing data. Red cells indicate missing values. This helps identify whether missingness follows a pattern (e.g., columns that are always missing together).

#### Outliers Tab
- Uses the **IQR (Interquartile Range) method** to detect outliers.
  - **IQR** = Q3 - Q1; outlier boundaries are Q1 - k*IQR and Q3 + k*IQR.
  - The **IQR Multiplier (k)** is configurable from 1.0 to 3.0 (default 1.5). Lower values flag more outliers; higher values are more permissive.
- **Outlier Summary Table** — shows Q1, Q3, IQR, lower/upper bounds, outlier count, and outlier percentage for each numeric column.
- **Box Plots** — interactive box plots for selected columns, providing a visual overview of the spread and outliers.
        """)

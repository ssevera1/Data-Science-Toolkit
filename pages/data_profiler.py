import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.theme import page_header, get_colors
from core.state import log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


def render():
    page_header("Data Profiler", "Automated exploratory data analysis — distributions, correlations, missing patterns, and outliers.", "📊")

    _guard()
    pd.set_option("future.no_silent_downcasting", True)
    df = st.session_state["df"]

    # ── Pre-compute data for tabs and PDF export ──────────────────────────────
    num_dups = df.duplicated().sum()
    mem = df.memory_usage(deep=True).sum()
    mem_str = f"{mem / 1024**2:.2f} MB"
    total_cells = df.shape[0] * df.shape[1]
    missing_pct_str = f"{df.isnull().sum().sum() / total_cells * 100:.2f}%" if total_cells > 0 else "0.00%"
    desc_stats_df = df.describe(include="all").T

    # Column types chart (built once, used in tab + PDF)
    type_counts = df.dtypes.astype(str).value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    type_chart_fig = px.bar(type_counts, x="Type", y="Count", color="Type", text_auto=True)
    type_chart_fig.update_layout(showlegend=False, height=300)

    # Outlier summary (IQR method, default multiplier 1.5)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    _default_iqr_mult = 1.5
    outlier_summary_default = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - _default_iqr_mult * iqr
        upper = q3 + _default_iqr_mult * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary_default.append({
            "Column": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "Lower Bound": round(lower, 4),
            "Upper Bound": round(upper, 4),
            "Outliers": n_outliers,
            "% Outliers": round(n_outliers / len(df) * 100, 2),
        })
    out_df_default = pd.DataFrame(outlier_summary_default) if outlier_summary_default else pd.DataFrame()

    # Correlation data (Pearson, pre-computed for tab + PDF)
    num_df = df.select_dtypes(include="number")
    corr_pairs_df = pd.DataFrame()
    corr_heatmap_fig = None
    if num_df.shape[1] >= 2:
        corr_matrix = num_df.corr(method="pearson")
        corr_heatmap_fig = px.imshow(
            corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, aspect="auto",
        )
        corr_heatmap_fig.update_layout(height=600)
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.8:
                    pairs.append({
                        "Feature A": corr_matrix.columns[i],
                        "Feature B": corr_matrix.columns[j],
                        "Correlation": round(val, 4),
                    })
        if pairs:
            corr_pairs_df = pd.DataFrame(pairs)

    # Missing values summary (pre-computed for tab + PDF)
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    miss_df = pd.DataFrame()
    miss_chart_fig = None
    if len(miss) > 0:
        miss_df = pd.DataFrame({
            "Column": miss.index,
            "Missing": miss.values,
            "% Missing": (miss.values / len(df) * 100).round(2),
        })
        miss_chart_fig = px.bar(miss_df, x="Column", y="% Missing", color="% Missing",
                                color_continuous_scale="Reds", text_auto=".1f")
        miss_chart_fig.update_layout(height=400)

    # Missing value pattern heatmap (pre-computed for tab + PDF)
    miss_pattern_fig = None
    if len(miss) > 0:
        sample = df[miss.index].head(200)
        miss_pattern_fig = px.imshow(
            sample.isnull().astype(int),
            color_continuous_scale=[get_colors()["bg_primary"], get_colors()["error"]],
            aspect="auto",
            labels=dict(color="Missing"),
        )
        miss_pattern_fig.update_layout(height=400)

    # Numeric distribution histograms (pre-computed for PDF; first 6 columns)
    num_dist_fig = None
    if num_cols:
        _pdf_num = num_cols[:6]
        _ncols = min(3, len(_pdf_num))
        _nrows = (len(_pdf_num) + _ncols - 1) // _ncols
        num_dist_fig = make_subplots(rows=_nrows, cols=_ncols, subplot_titles=_pdf_num)
        for _i, _col in enumerate(_pdf_num):
            _r, _c = divmod(_i, _ncols)
            num_dist_fig.add_trace(
                go.Histogram(x=df[_col].dropna(), name=_col, showlegend=False),
                row=_r + 1, col=_c + 1,
            )
        num_dist_fig.update_layout(height=300 * _nrows)

    # Box plots (pre-computed for PDF; first 6 numeric columns)
    box_fig = None
    if num_cols:
        _box_cols = num_cols[:6]
        box_fig = px.box(df[_box_cols].melt(), x="variable", y="value", color="variable")
        box_fig.update_layout(height=500, showlegend=False)

    tab_overview, tab_dist, tab_corr, tab_missing, tab_outliers = st.tabs(
        ["Overview", "Distributions", "Correlations", "Missing Values", "Outliers"]
    )

    # ── Overview ───────────────────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Dataset Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Duplicate Rows", f"{num_dups:,}")
        c4.metric("Memory", mem_str)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(desc_stats_df.fillna("").infer_objects(copy=False).astype(str), width="stretch")

        st.markdown("#### Column Types")
        st.plotly_chart(type_chart_fig, width="stretch")

    # ── Distributions ──────────────────────────────────────────────────────────
    with tab_dist:
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
        if num_df.shape[1] >= 2:
            method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
            # Use pre-computed Pearson or recompute for other methods
            if method == "pearson":
                corr = corr_matrix
                fig = corr_heatmap_fig
            else:
                corr = num_df.corr(method=method)
                fig = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                )
                fig.update_layout(height=600)
            st.plotly_chart(fig, width="stretch")

            st.markdown("#### Highly Correlated Pairs (|r| > 0.8)")
            # Recompute for the selected method
            pairs_display = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.8:
                        pairs_display.append({
                            "Feature A": corr.columns[i],
                            "Feature B": corr.columns[j],
                            "Correlation": round(val, 4),
                        })
            if pairs_display:
                st.dataframe(pd.DataFrame(pairs_display), width="stretch", hide_index=True)
            else:
                st.success("No highly correlated pairs found (|r| > 0.8).")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    # ── Missing Values ─────────────────────────────────────────────────────────
    with tab_missing:
        st.subheader("Missing Value Analysis")

        if miss_df.empty:
            st.success("No missing values found!")
        else:
            st.dataframe(miss_df, width="stretch", hide_index=True)
            st.plotly_chart(miss_chart_fig, width="stretch")

            # Missing value heatmap (sample if large)
            st.markdown("#### Missing Value Pattern")
            st.plotly_chart(miss_pattern_fig, width="stretch")

    # ── Outliers ───────────────────────────────────────────────────────────────
    with tab_outliers:
        st.subheader("Outlier Detection (IQR Method)")
        if num_cols:
            iqr_mult = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
            # Recompute with user-selected multiplier for display
            if iqr_mult != _default_iqr_mult:
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
            else:
                out_df = out_df_default
            st.dataframe(out_df, width="stretch", hide_index=True)

            sel_box = st.multiselect("Box plots for", num_cols, default=num_cols[:6])
            if sel_box:
                fig = px.box(df[sel_box].melt(), x="variable", y="value", color="variable")
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No numeric columns found.")

    # ── AI Interpretation ──────────────────────────────────────────────────
    from components.ai_advisor import render_ai_interpretation
    ai_texts = render_ai_interpretation(
        entry_type="data_profiler",
        result={
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "duplicates": int(num_dups),
            "memory": mem_str,
            "missing_pct": missing_pct_str,
        },
        variables={},
        page_key="prof",
    )

    # ── PDF Export ─────────────────────────────────────────────────────────
    st.divider()
    _tables = [_serialize_df(desc_stats_df, "Descriptive Statistics")]
    if not corr_pairs_df.empty:
        _tables.append(_serialize_df(corr_pairs_df, "Highly Correlated Pairs (|r| > 0.8)"))
    if not miss_df.empty:
        _tables.append(_serialize_df(miss_df, "Missing Value Summary"))
    if not out_df_default.empty:
        _tables.append(_serialize_df(out_df_default, "Outlier Summary (IQR)"))

    _log_entry = build_log_entry(
        entry_type="data_profiler",
        title=f"Data Profiler: {st.session_state.get('file_name', 'Dataset')}",
        result={
            "rows": len(df),
            "columns": df.shape[1],
            "duplicates": int(num_dups),
            "memory": mem_str,
            "missing_pct": missing_pct_str,
        },
        tables=_tables,
        variables={},
        dataset_name=st.session_state.get("file_name", ""),
    )
    if ai_texts.get("brief"):
        _log_entry["ai_interpretation"] = ai_texts["brief"]
    if ai_texts.get("deep_dive"):
        _log_entry["ai_deep_dive"] = ai_texts["deep_dive"]

    _include_chart = st.checkbox("Include charts in PDF", value=True, key="prof_pdf_chart")
    if _include_chart:
        _figures = [{"label": "Column Types", "fig_dict": type_chart_fig.to_dict()}]
        if num_dist_fig is not None:
            _figures.append({"label": "Numeric Distributions", "fig_dict": num_dist_fig.to_dict()})
        if corr_heatmap_fig is not None:
            _figures.append({"label": "Correlation Matrix (Pearson)", "fig_dict": corr_heatmap_fig.to_dict()})
        if miss_chart_fig is not None:
            _figures.append({"label": "Missing Values", "fig_dict": miss_chart_fig.to_dict()})
        if miss_pattern_fig is not None:
            _figures.append({"label": "Missing Value Pattern", "fig_dict": miss_pattern_fig.to_dict()})
        if box_fig is not None:
            _figures.append({"label": "Box Plots (Outliers)", "fig_dict": box_fig.to_dict()})
        _log_entry["figures"] = _figures

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if st.button("Add to Report", key="prof_add_report"):
            if log_result(_log_entry):
                st.success("Added to report log.")
            else:
                st.error("Report log is full (100 entries). Clear it first.")
    with exp_col2:
        st.download_button(
            "Export PDF",
            data=generate_single_report(_log_entry, include_charts=_include_chart),
            file_name="data_profiler.pdf",
            mime="application/pdf",
            key="prof_export_pdf",
        )

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

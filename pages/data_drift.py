import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from utils.theme import page_header, get_colors
from core.data_manager import sanitize_csv as _sanitize_csv
from core.state import log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df


MAX_SIZE_MB = 50
MAX_ROWS = 500_000
MAX_COLS = 500


def _load_upload(file, label):
    """Load an uploaded file with size and shape guards."""
    if file.size > MAX_SIZE_MB * 1024 * 1024:
        st.error(f"{label} file exceeds the {MAX_SIZE_MB} MB limit. Please upload a smaller file.")
        return None
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    except Exception:
        st.error(f"Error loading {label.lower()} file. Please check that it is a valid CSV or Excel document.")
        return None
    if df.shape[0] > MAX_ROWS or df.shape[1] > MAX_COLS:
        st.error(
            f"{label} dataset too large ({df.shape[0]:,} rows x {df.shape[1]} columns). "
            f"Maximum supported size is {MAX_ROWS:,} rows and {MAX_COLS} columns."
        )
        return None
    return df


def render():
    page_header("Data Drift Detection", "Upload a reference and current dataset — detect feature drift with statistical tests.", "📈")

    # ── Data Input ─────────────────────────────────────────────────────────────
    st.subheader("1. Upload Datasets")
    mode = st.radio("Mode", [
        "Upload two separate files (reference vs current)",
        "Split current dataset by time/index",
    ], horizontal=True)

    ref_df = None
    cur_df = None

    if mode.startswith("Upload"):
        c1, c2 = st.columns(2)
        with c1:
            ref_file = st.file_uploader("Reference dataset (baseline)", type=["csv", "xlsx"], key="ref")
            if ref_file:
                ref_df = _load_upload(ref_file, "Reference")
                if ref_df is not None:
                    st.write(f"Reference: {ref_df.shape[0]:,} rows x {ref_df.shape[1]} cols")
        with c2:
            cur_file = st.file_uploader("Current dataset", type=["csv", "xlsx"], key="cur")
            if cur_file:
                cur_df = _load_upload(cur_file, "Current")
                if cur_df is not None:
                    st.write(f"Current: {cur_df.shape[0]:,} rows x {cur_df.shape[1]} cols")

    else:
        if "df" not in st.session_state:
            st.warning("Upload a dataset on the **Home** page first.")
            st.stop()
        full_df = st.session_state["df"]
        split_pct = st.slider("Reference split % (first N% = reference)", 10, 90, 50)
        split_idx = int(len(full_df) * split_pct / 100)
        ref_df = full_df.iloc[:split_idx].copy()
        cur_df = full_df.iloc[split_idx:].copy()
        st.write(f"Reference: {len(ref_df):,} rows | Current: {len(cur_df):,} rows")

    if ref_df is None or cur_df is None:
        st.info("Provide both reference and current datasets to proceed.")
        st.stop()

    # Find common numeric columns
    common_cols = list(set(ref_df.select_dtypes(include="number").columns)
                       & set(cur_df.select_dtypes(include="number").columns))
    common_cat_cols = list(set(ref_df.select_dtypes(include=["object", "category"]).columns)
                           & set(cur_df.select_dtypes(include=["object", "category"]).columns))

    if not common_cols and not common_cat_cols:
        st.warning("No common columns found between datasets.")
        st.stop()

    # ── Statistical Tests ──────────────────────────────────────────────────────
    st.subheader("2. Drift Detection Results")
    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)

    # ── Compute drift results before tabs (needed for both display and PDF) ──
    drift_df = None
    cat_df = None

    if common_cols:
        drift_results = []
        for col in common_cols:
            ref_vals = ref_df[col].dropna()
            cur_vals = cur_df[col].dropna()

            # KS Test
            ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)

            # Welch's t-test
            t_stat, t_p = stats.ttest_ind(ref_vals, cur_vals, equal_var=False)

            # Population Stability Index (PSI)
            def compute_psi(ref, cur, bins=10):
                breakpoints = np.linspace(min(ref.min(), cur.min()),
                                          max(ref.max(), cur.max()), bins + 1)
                ref_pct = np.histogram(ref, breakpoints)[0] / len(ref)
                cur_pct = np.histogram(cur, breakpoints)[0] / len(cur)
                ref_pct = np.clip(ref_pct, 1e-6, None)
                cur_pct = np.clip(cur_pct, 1e-6, None)
                psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
                return psi

            psi = compute_psi(ref_vals, cur_vals)

            drifted = ks_p < alpha
            drift_results.append({
                "Feature": col,
                "KS Statistic": round(ks_stat, 4),
                "KS p-value": round(ks_p, 6),
                "T-test p-value": round(t_p, 6),
                "PSI": round(psi, 4),
                "Ref Mean": round(ref_vals.mean(), 4),
                "Cur Mean": round(cur_vals.mean(), 4),
                "Mean Shift %": round(abs(cur_vals.mean() - ref_vals.mean()) / (abs(ref_vals.mean()) + 1e-10) * 100, 2),
                "Drift Detected": drifted,
            })

        drift_df = pd.DataFrame(drift_results).sort_values("KS p-value")

    if common_cat_cols:
        cat_results = []
        for col in common_cat_cols:
            ref_vc = ref_df[col].value_counts(normalize=True)
            cur_vc = cur_df[col].value_counts(normalize=True)

            # Chi-square test
            all_cats = set(ref_vc.index) | set(cur_vc.index)
            ref_counts = np.array([ref_df[col].value_counts().get(c, 0) for c in all_cats])
            cur_counts = np.array([cur_df[col].value_counts().get(c, 0) for c in all_cats])

            if len(all_cats) > 1:
                contingency = np.array([ref_counts, cur_counts])
                chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
            else:
                chi2, chi_p, dof = 0, 1, 0

            cat_results.append({
                "Feature": col,
                "Unique (Ref)": ref_df[col].nunique(),
                "Unique (Cur)": cur_df[col].nunique(),
                "Chi2 Statistic": round(chi2, 4),
                "Chi2 p-value": round(chi_p, 6),
                "Drift Detected": chi_p < alpha,
            })

        cat_df = pd.DataFrame(cat_results).sort_values("Chi2 p-value")

    tab_num, tab_cat, tab_viz = st.tabs(["Numeric Features", "Categorical Features", "Distributions"])

    with tab_num:
        if drift_df is None:
            st.info("No common numeric columns.")
        else:
            n_drifted = drift_df["Drift Detected"].sum()
            c1, c2, c3 = st.columns(3)
            c1.metric("Features Tested", len(common_cols))
            c2.metric("Drift Detected", n_drifted)
            c3.metric("Drift %", f"{n_drifted / len(common_cols) * 100:.1f}%")

            # Color code
            _c = get_colors()
            _drift_bg = _c["drift_highlight"]
            def highlight_drift(row):
                if row["Drift Detected"]:
                    return [f"background-color: {_drift_bg}"] * len(row)
                return [""] * len(row)

            st.dataframe(drift_df.style.apply(highlight_drift, axis=1),
                         width="stretch", hide_index=True)

            drift_csv = _sanitize_csv(drift_df).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Numeric Drift CSV",
                data=drift_csv,
                file_name="numeric_drift_results.csv",
                mime="text/csv",
            )

            # PSI interpretation
            st.markdown("""
            **PSI Interpretation:**
            - PSI < 0.1: No significant shift
            - 0.1 < PSI < 0.25: Moderate shift — investigate
            - PSI > 0.25: Significant shift — model retraining likely needed
            """)

    with tab_cat:
        if cat_df is None:
            st.info("No common categorical columns.")
        else:
            st.dataframe(cat_df, width="stretch", hide_index=True)

            cat_csv = _sanitize_csv(cat_df).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Categorical Drift CSV",
                data=cat_csv,
                file_name="categorical_drift_results.csv",
                mime="text/csv",
            )

    with tab_viz:
        st.subheader("Distribution Comparison")
        if common_cols:
            sel = st.multiselect("Numeric features to compare", common_cols,
                                 default=common_cols[:4])
            if sel:
                for col in sel:
                    fig = go.Figure()
                    _dc = get_colors()
                    fig.add_trace(go.Histogram(x=ref_df[col].dropna(), name="Reference",
                                               opacity=0.6, marker_color=_dc["info"]))
                    fig.add_trace(go.Histogram(x=cur_df[col].dropna(), name="Current",
                                               opacity=0.6, marker_color=_dc["error"]))
                    fig.update_layout(barmode="overlay", title=col, height=350)
                    st.plotly_chart(fig, width="stretch")

        if common_cat_cols:
            sel_cat = st.multiselect("Categorical features to compare", common_cat_cols,
                                      default=common_cat_cols[:2])
            for col in sel_cat:
                ref_vc = ref_df[col].value_counts().head(15).reset_index()
                ref_vc.columns = [col, "Count"]
                ref_vc["Source"] = "Reference"
                cur_vc = cur_df[col].value_counts().head(15).reset_index()
                cur_vc.columns = [col, "Count"]
                cur_vc["Source"] = "Current"
                combined = pd.concat([ref_vc, cur_vc])
                _dc2 = get_colors()
                fig = px.bar(combined, x=col, y="Count", color="Source", barmode="group",
                             title=col, color_discrete_map={"Reference": _dc2["info"], "Current": _dc2["error"]})
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")

    # ── AI Interpretation ──────────────────────────────────────────────────
    from components.ai_advisor import render_ai_interpretation
    _n_drifted_num_ai = int(drift_df["Drift Detected"].sum()) if drift_df is not None else 0
    _n_drifted_cat_ai = int(cat_df["Drift Detected"].sum()) if cat_df is not None else 0
    _n_drifted_ai = _n_drifted_num_ai + _n_drifted_cat_ai
    _n_features_ai = len(common_cols) + len(common_cat_cols)
    _drift_pct_ai = f"{_n_drifted_ai / _n_features_ai * 100:.1f}%" if _n_features_ai > 0 else "0.0%"
    ai_texts = render_ai_interpretation(
        entry_type="data_drift",
        result={
            "n_features": _n_features_ai,
            "n_drifted": _n_drifted_ai,
            "drift_pct": _drift_pct_ai,
            "alpha": alpha,
        },
        variables={
            "numeric_features": len(common_cols),
            "categorical_features": len(common_cat_cols),
        },
        alpha=alpha,
        page_key="drift",
    )

    # ── PDF Export ─────────────────────────────────────────────────────────
    st.divider()

    _n_features = len(common_cols) + len(common_cat_cols)
    _n_drifted_num = int(drift_df["Drift Detected"].sum()) if drift_df is not None else 0
    _n_drifted_cat = int(cat_df["Drift Detected"].sum()) if cat_df is not None else 0
    _n_drifted_total = _n_drifted_num + _n_drifted_cat
    _drift_pct = f"{_n_drifted_total / _n_features * 100:.1f}%" if _n_features > 0 else "0.0%"

    _tables = []
    if drift_df is not None:
        _tables.append(_serialize_df(drift_df, "Numeric Drift Results"))
    if cat_df is not None:
        _tables.append(_serialize_df(cat_df, "Categorical Drift Results"))

    _log_entry = build_log_entry(
        entry_type="data_drift",
        title="Data Drift Detection",
        result={
            "n_features": _n_features,
            "n_drifted": _n_drifted_total,
            "drift_pct": _drift_pct,
            "alpha": alpha,
        },
        tables=_tables,
        variables={
            "numeric_features": len(common_cols),
            "categorical_features": len(common_cat_cols),
        },
        dataset_name=st.session_state.get("file_name", ""),
        alpha=alpha,
    )
    if ai_texts.get("brief"):
        _log_entry["ai_interpretation"] = ai_texts["brief"]
    if ai_texts.get("deep_dive"):
        _log_entry["ai_deep_dive"] = ai_texts["deep_dive"]

    _include_chart = st.checkbox("Include charts in PDF", value=True, key="drift_pdf_chart")
    if _include_chart and common_cols:
        _figures = []
        for _col in common_cols[:6]:
            _fig = go.Figure()
            _dc = get_colors()
            _fig.add_trace(go.Histogram(x=ref_df[_col].dropna(), name="Reference",
                                        opacity=0.6, marker_color=_dc["info"]))
            _fig.add_trace(go.Histogram(x=cur_df[_col].dropna(), name="Current",
                                        opacity=0.6, marker_color=_dc["error"]))
            _fig.update_layout(barmode="overlay", title=_col, height=350)
            _figures.append({"label": f"Distribution: {_col}", "fig_dict": _fig.to_dict()})
        _log_entry["figures"] = _figures

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if st.button("Add to Report", key="drift_add_report"):
            if log_result(_log_entry):
                st.success("Added to report log.")
            else:
                st.error("Report log is full (100 entries). Clear it first.")
    with exp_col2:
        st.download_button(
            "Export PDF",
            data=generate_single_report(_log_entry, include_charts=_include_chart),
            file_name="data_drift.pdf",
            mime="application/pdf",
            key="drift_export_pdf",
        )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Data Drift Detection — Monitor Feature Distribution Changes

This page detects whether the statistical distribution of your features has changed between a reference (baseline) dataset and a current (production) dataset.

---

#### Data Input Modes
- **Upload two separate files** — provide a reference dataset (e.g., training data) and a current dataset (e.g., new production data) as CSV or Excel files.
- **Split current dataset** — divide the dataset already loaded in the app by a percentage. The first N% becomes the reference and the remainder becomes the current dataset. Useful for time-based splits.

#### Numeric Drift Detection
Three statistical tests are applied to each common numeric feature:
- **Kolmogorov-Smirnov (KS) Test** — compares the **entire distributions** of two samples. The KS statistic measures the maximum distance between the two cumulative distribution functions. A low p-value indicates the distributions are significantly different.
- **Welch's t-test** — compares the **means** of two samples without assuming equal variances. A low p-value indicates the means are significantly different. Less sensitive to shape changes than KS.
- **Population Stability Index (PSI)** — a widely-used metric in model monitoring:
  - **PSI < 0.1** — no significant shift.
  - **0.1 < PSI < 0.25** — moderate shift; investigate further.
  - **PSI > 0.25** — significant shift; model retraining is likely needed.
  - PSI is computed by binning both distributions and comparing the proportions in each bin.

#### Categorical Drift Detection
- **Chi-squared test of independence** — compares the frequency distribution of categories between the reference and current datasets. A low p-value indicates the categorical distribution has shifted significantly.
- Reports the number of unique categories in each dataset for comparison.

#### Distribution Visualization
- **Numeric features** — overlaid histograms (blue = reference, red = current) for visual comparison of distribution shapes.
- **Categorical features** — grouped bar charts showing the top 15 category counts side by side.

#### Drift Interpretation
- **Significance level (alpha)** — configurable from 0.01 to 0.10 (default 0.05). Features with a KS p-value below alpha are flagged as drifted.
- **Drift summary metrics** — total features tested, number with drift detected, and drift percentage.
- Rows with detected drift are **highlighted in red** in the results table.
- **Mean Shift %** shows the relative change in the feature mean between reference and current datasets.
        """)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.theme import page_header
from core.state import log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df
from core.data_manager import sanitize_csv as _sanitize_csv

_CACHE_KEY = "_result_feat_sel"


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


def render():
    page_header("Feature Selection", "Find the best features using correlation filters, mutual information, variance threshold, and RFE.", "🎯")

    _guard()
    df = st.session_state["df"].copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if len(all_cols) < 2:
        st.warning("Need at least 2 columns (1 target + 1 feature).")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Select target column", all_cols)
    with c2:
        target_is_categorical = not pd.api.types.is_numeric_dtype(df[target_col])
        is_clf = target_is_categorical or df[target_col].nunique() <= 20
        if target_is_categorical:
            task_options = ["Classification"]
        else:
            task_options = ["Auto-detect", "Classification", "Regression"]
        task = st.selectbox("Task type", task_options, key="fs_task")
        if task == "Auto-detect":
            task = "Classification" if is_clf else "Regression"
        n_classes = df[target_col].nunique()
        is_binary = task == "Classification" and n_classes == 2
        label = f"**{task}** — {'binary' if is_binary else f'{n_classes} classes' if task == 'Classification' else 'continuous'}"
        st.write(label)

    feature_cols = [c for c in num_cols if c != target_col]

    if not feature_cols:
        st.warning("No numeric feature columns available. Encode categorical features first (Smart Cleaning page).")
        st.stop()

    X = df[feature_cols].fillna(0)

    # Encode target for all methods
    if task == "Classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(df[target_col]), name=target_col)
    else:
        y = df[target_col].fillna(0)

    # ── Cache invalidation ────────────────────────────────────────────────
    _fingerprint = (target_col, task, tuple(feature_cols))
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached.get("inputs") != _fingerprint:
        del st.session_state[_CACHE_KEY]
        cached = None

    tab_corr, tab_mi, tab_var, tab_rfe, tab_summary = st.tabs(
        ["Correlation Filter", "Mutual Information", "Variance Threshold", "RFE", "Summary"]
    )

    results = {}

    # ── Correlation Filter ─────────────────────────────────────────────────────
    with tab_corr:
        st.subheader("Correlation with Target")
        method = st.selectbox("Method", ["pearson", "spearman"], key="corr_method")
        # Drop zero-variance columns — correlation is undefined for constants
        _X_corr = X.loc[:, X.std() > 0]
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            corr_scores = _X_corr.corrwith(y, method=method).abs().sort_values(ascending=False)
        # Add back constant columns with correlation = 0
        _const_cols = [c for c in X.columns if c not in _X_corr.columns]
        for _cc in _const_cols:
            corr_scores[_cc] = 0.0
        corr_scores = corr_scores.sort_values(ascending=False)
        corr_df = pd.DataFrame({"Feature": corr_scores.index, "Abs Correlation": corr_scores.values})
        results["Correlation"] = corr_df.set_index("Feature")["Abs Correlation"]

        fig = px.bar(corr_df, x="Abs Correlation", y="Feature", orientation="h",
                     color="Abs Correlation", color_continuous_scale="Viridis")
        fig.update_layout(height=max(400, len(feature_cols) * 25), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

        threshold = st.slider("Drop features with |corr| below", 0.0, 1.0, 0.05, 0.01)
        kept = corr_scores[corr_scores >= threshold].index.tolist()
        st.write(f"**{len(kept)}** / {len(feature_cols)} features kept (threshold = {threshold})")

    # ── Mutual Information ─────────────────────────────────────────────────────
    with tab_mi:
        st.subheader("Mutual Information Scores")
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

        mi_func = mutual_info_classif if task == "Classification" else mutual_info_regression
        st.info(f"Task: **{task}** ({n_classes} unique target values)")

        mi_scores = mi_func(X, y, random_state=42)
        mi_df = pd.DataFrame({"Feature": feature_cols, "MI Score": mi_scores})
        mi_df = mi_df.sort_values("MI Score", ascending=False)
        results["Mutual Info"] = mi_df.set_index("Feature")["MI Score"]

        fig = px.bar(mi_df, x="MI Score", y="Feature", orientation="h",
                     color="MI Score", color_continuous_scale="Plasma")
        fig.update_layout(height=max(400, len(feature_cols) * 25), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

    # ── Variance Threshold ────────────────────────────────────────────────────
    with tab_var:
        st.subheader("Variance Threshold")
        from sklearn.feature_selection import VarianceThreshold

        variances = X.var().sort_values(ascending=False)
        var_df = pd.DataFrame({"Feature": variances.index, "Variance": variances.values})
        results["Variance"] = var_df.set_index("Feature")["Variance"]

        fig = px.bar(var_df, x="Variance", y="Feature", orientation="h",
                     color="Variance", color_continuous_scale="Cividis")
        fig.update_layout(height=max(400, len(feature_cols) * 25), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

        thresh = st.number_input("Variance threshold", 0.0, float(variances.max()), 0.01, 0.01)
        low_var = variances[variances < thresh].index.tolist()
        st.write(f"**{len(low_var)}** features below threshold: {low_var[:10]}")

    # ── RFE (Recursive Feature Elimination) ───────────────────────────────────
    with tab_rfe:
        st.subheader("Recursive Feature Elimination")
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        n_select = st.slider("Features to select", 1, len(feature_cols), max(1, len(feature_cols) // 2))

        _rfe_clicked = st.button("Run RFE")
        if _rfe_clicked:
            with st.spinner("Running RFE..."):
                if task == "Classification":
                    estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                else:
                    estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

                rfe = RFE(estimator, n_features_to_select=n_select, step=1)
                rfe.fit(X, y)

                rfe_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Ranking": rfe.ranking_,
                    "Selected": rfe.support_,
                }).sort_values("Ranking")

                rfe_ranking = rfe_df.set_index("Feature")["Ranking"]
                selected = rfe_df[rfe_df["Selected"]]["Feature"].tolist()

                # Store RFE results in cache
                st.session_state[_CACHE_KEY] = {
                    "inputs": _fingerprint,
                    "rfe_ranking": rfe_ranking,
                    "rfe_df": rfe_df,
                    "rfe_selected": selected,
                }

                results["RFE Rank"] = rfe_ranking

                st.dataframe(rfe_df, width="stretch", hide_index=True)
                st.success(f"Selected features: {selected}")

        # Restore RFE results from cache on rerun (when button was not just clicked)
        if not _rfe_clicked:
            cached = st.session_state.get(_CACHE_KEY)
            if cached and cached.get("inputs") == _fingerprint and "rfe_ranking" in cached:
                results["RFE Rank"] = cached["rfe_ranking"]
                rfe_df = cached["rfe_df"]
                selected = cached["rfe_selected"]
                st.dataframe(rfe_df, width="stretch", hide_index=True)
                st.success(f"Selected features: {selected}")

    # ── Summary ────────────────────────────────────────────────────────────────
    with tab_summary:
        st.subheader("Feature Ranking Summary")
        if results:
            summary = pd.DataFrame(results)
            # Normalize each method to 0-1 for comparison
            for col in summary.columns:
                mn, mx = summary[col].min(), summary[col].max()
                if mx > mn:
                    summary[f"{col} (norm)"] = (summary[col] - mn) / (mx - mn)
                else:
                    summary[f"{col} (norm)"] = 0

            norm_cols = [c for c in summary.columns if "(norm)" in c]
            if norm_cols:
                summary["Avg Rank Score"] = summary[norm_cols].mean(axis=1)
                summary = summary.sort_values("Avg Rank Score", ascending=False)

            st.dataframe(summary, width="stretch")

            csv_data = _sanitize_csv(summary).to_csv().encode("utf-8")
            st.download_button(
                "Download Rankings CSV",
                data=csv_data,
                file_name="feature_rankings.csv",
                mime="text/csv",
            )

            # Allow user to select and apply
            st.divider()
            top_n = st.slider("Keep top N features", 1, len(feature_cols), max(1, len(feature_cols) // 2),
                              key="top_n_summary")
            top_features = summary.index[:top_n].tolist()
            st.write(f"Top {top_n} features: {top_features}")

            dl_col, apply_col = st.columns(2)
            with dl_col:
                selected_df = df[top_features + [target_col]]
                sel_csv = _sanitize_csv(selected_df).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Selected Features CSV",
                    data=sel_csv,
                    file_name="selected_features.csv",
                    mime="text/csv",
                    width="stretch",
                )
            with apply_col:
                if st.button("Apply: Keep only selected features + target", width="stretch"):
                    keep = top_features + [target_col]
                    st.session_state["df"] = df[keep]
                    st.success(f"Reduced to {len(keep)} columns.")
            # ── PDF Export ─────────────────────────────────────────
            st.divider()
            _top = summary.index[:5].tolist() if len(summary) >= 5 else summary.index.tolist()
            _log_entry = build_log_entry(
                entry_type="feature_selection",
                title=f"Feature Selection: {target_col} ({task})",
                result={
                    "target": target_col,
                    "task": task,
                    "top_features": _top,
                    "n_features": len(feature_cols),
                },
                tables=[_serialize_df(summary, "Feature Rankings")],
                variables={"target": target_col, "task": task},
                dataset_name=st.session_state.get("file_name", ""),
            )
            _include_chart = st.checkbox("Include charts in PDF", value=True, key="fs_pdf_chart")
            if _include_chart:
                _figures = []
                if "Correlation" in results:
                    _corr_df = pd.DataFrame({"Feature": results["Correlation"].index,
                                             "Abs Correlation": results["Correlation"].values})
                    _cfig = px.bar(_corr_df, x="Abs Correlation", y="Feature", orientation="h",
                                   color="Abs Correlation", color_continuous_scale="Viridis")
                    _cfig.update_layout(height=max(400, len(feature_cols) * 25),
                                        yaxis=dict(autorange="reversed"))
                    _figures.append({"label": "Correlation Filter", "fig_dict": _cfig.to_dict()})
                if "Mutual Info" in results:
                    _mi_df = pd.DataFrame({"Feature": results["Mutual Info"].index,
                                           "MI Score": results["Mutual Info"].values})
                    _mfig = px.bar(_mi_df, x="MI Score", y="Feature", orientation="h",
                                   color="MI Score", color_continuous_scale="Plasma")
                    _mfig.update_layout(height=max(400, len(feature_cols) * 25),
                                        yaxis=dict(autorange="reversed"))
                    _figures.append({"label": "Mutual Information", "fig_dict": _mfig.to_dict()})
                if "Variance" in results:
                    _var_df = pd.DataFrame({"Feature": results["Variance"].index,
                                            "Variance": results["Variance"].values})
                    _vfig = px.bar(_var_df, x="Variance", y="Feature", orientation="h",
                                   color="Variance", color_continuous_scale="Cividis")
                    _vfig.update_layout(height=max(400, len(feature_cols) * 25),
                                        yaxis=dict(autorange="reversed"))
                    _figures.append({"label": "Variance Threshold", "fig_dict": _vfig.to_dict()})
                _log_entry["figures"] = _figures
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                if st.button("Add to Report", key="fs_add_report"):
                    if log_result(_log_entry):
                        st.success("Added to report log.")
                    else:
                        st.error("Report log is full (100 entries). Clear it first.")
            with exp_col2:
                st.download_button(
                    "Export PDF",
                    data=generate_single_report(_log_entry, include_charts=_include_chart),
                    file_name="feature_selection.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("Run the methods above to populate this summary.")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Feature Selection — Identify the Most Important Features

This page helps you rank and filter features using four complementary methods, then combine them into a unified ranking.

---

#### Configuration
- **Target column** — the variable you want to predict.
- **Task type** — auto-detected as Classification (categorical or <= 20 unique values) or Regression (continuous). You can override the auto-detection.
- Only **numeric** feature columns are evaluated. Encode categorical features first using the Smart Cleaning page.

#### Correlation Filter Tab
- Computes the **absolute correlation** between each feature and the target variable.
- Choose **Pearson** (linear relationship) or **Spearman** (monotonic/rank-based relationship).
- A horizontal bar chart ranks features by correlation strength.
- Set a **threshold** to drop features with weak correlation (e.g., |r| < 0.05).

#### Mutual Information Tab
- **Mutual Information (MI)** measures the **non-linear dependency** between each feature and the target.
- Uses `mutual_info_classif` for classification tasks and `mutual_info_regression` for regression tasks.
- MI is always >= 0. Higher values indicate stronger dependency. Unlike correlation, MI captures any kind of statistical relationship, not just linear ones.
- Results are displayed as a ranked bar chart.

#### Variance Threshold Tab
- Computes the **variance** of each feature column.
- Features with **very low variance** (near-constant) carry little predictive information and can be safely removed.
- Set a **variance threshold** to identify features below that cutoff.
- Common use case: removing features where almost all values are the same.

#### RFE (Recursive Feature Elimination) Tab
- Uses a **Random Forest** estimator to iteratively remove the least important features.
- At each step, the model is trained, the least important feature is removed, and the process repeats until the desired number of features remains.
- Configure the **number of features to select** with the slider.
- Results show the ranking (1 = selected) and which features were kept.

#### Summary Tab
- Aggregates results from all methods run so far.
- Each method's scores are **normalized to a 0-1 scale** for fair comparison.
- An **Average Rank Score** is computed across all methods.
- Use the **"Keep top N features"** slider to select the best features, then click to apply the selection and reduce your dataset to only those features plus the target column.
        """)

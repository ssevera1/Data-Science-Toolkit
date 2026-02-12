import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.theme import page_header


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

    tab_corr, tab_mi, tab_var, tab_rfe, tab_summary = st.tabs(
        ["Correlation Filter", "Mutual Information", "Variance Threshold", "RFE", "Summary"]
    )

    results = {}

    # ── Correlation Filter ─────────────────────────────────────────────────────
    with tab_corr:
        st.subheader("Correlation with Target")
        method = st.selectbox("Method", ["pearson", "spearman"], key="corr_method")
        corr_scores = X.corrwith(y, method=method).abs().sort_values(ascending=False)
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

        if st.button("Run RFE"):
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

                results["RFE Rank"] = rfe_df.set_index("Feature")["Ranking"]

                st.dataframe(rfe_df, width="stretch", hide_index=True)
                selected = rfe_df[rfe_df["Selected"]]["Feature"].tolist()
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

            csv_data = summary.to_csv().encode("utf-8")
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
                sel_csv = selected_df.to_csv(index=False).encode("utf-8")
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

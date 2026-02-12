import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.theme import page_header


def _guard():
    if "df" not in st.session_state:
        st.warning("Upload a dataset on the **Home** page first.")
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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

                st.dataframe(rfe_df, use_container_width=True, hide_index=True)
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

            st.dataframe(summary, use_container_width=True)

            # Allow user to select and apply
            st.divider()
            top_n = st.slider("Keep top N features", 1, len(feature_cols), max(1, len(feature_cols) // 2),
                              key="top_n_summary")
            top_features = summary.index[:top_n].tolist()
            st.write(f"Top {top_n} features: {top_features}")

            if st.button("Apply: Keep only selected features + target"):
                keep = top_features + [target_col]
                st.session_state["df"] = df[keep]
                st.success(f"Reduced to {len(keep)} columns.")
        else:
            st.info("Run the methods above to populate this summary.")

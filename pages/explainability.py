import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.theme import page_header, set_matplotlib_theme


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


@st.cache_resource
def train_model(_X, _y, model_name, _task, feature_hash):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(_X)

    if model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if _task == "Classification" else RandomForestRegressor
        model = cls(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        cls = XGBClassifier if _task == "Classification" else XGBRegressor
        model = cls(n_estimators=100, random_state=42, verbosity=0)
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier, LGBMRegressor
        cls = LGBMClassifier if _task == "Classification" else LGBMRegressor
        model = cls(n_estimators=100, random_state=42, verbose=-1)
    elif model_name == "Gradient Boosting":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        cls = GradientBoostingClassifier if _task == "Classification" else GradientBoostingRegressor
        model = cls(n_estimators=100, random_state=42)

    model.fit(X_scaled, _y)
    return model, scaler, X_scaled


def render():
    page_header("Model Explainability", "SHAP values, feature importance, and partial dependence — understand any model's decisions.", "🔍")
    set_matplotlib_theme()

    _guard()
    df = st.session_state["df"].copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if len(all_cols) < 2:
        st.warning("Need at least 2 columns (1 target + 1 feature).")
        st.stop()

    # ── Setup ──────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("Target column", all_cols, key="exp_target")
    with c2:
        target_is_categorical = not pd.api.types.is_numeric_dtype(df[target])
        is_clf = target_is_categorical or df[target].nunique() <= 20
        if target_is_categorical:
            task_options = ["Classification"]
        else:
            task_options = ["Auto-detect", "Classification", "Regression"]
        task = st.selectbox("Task type", task_options, key="exp_task")
        if task == "Auto-detect":
            task = "Classification" if is_clf else "Regression"
        n_classes = df[target].nunique()
        is_binary = task == "Classification" and n_classes == 2
        label = f"**{task}** — {'binary' if is_binary else f'{n_classes} classes' if task == 'Classification' else 'continuous'}"
        st.write(label)

    feature_cols = [c for c in num_cols if c != target]
    if not feature_cols:
        st.warning("No numeric feature columns available. Encode categorical features first (Smart Cleaning page).")
        st.stop()

    X = df[feature_cols].fillna(0)
    y = df[target]

    if task == "Classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.fillna(0).astype(float).values

    model_choice = st.selectbox("Model", [
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Gradient Boosting",
    ])

    tab_importance, tab_shap, tab_pdp = st.tabs(
        ["Feature Importance", "SHAP Analysis", "Partial Dependence"]
    )

    # ── Train the model ────────────────────────────────────────────────────────
    import hashlib
    data_bytes = pd.util.hash_pandas_object(pd.DataFrame(X, columns=feature_cols)).values.tobytes()
    feature_hash = hashlib.md5(data_bytes).hexdigest()

    with st.spinner("Training model..."):
        model, scaler, X_scaled = train_model(X, y, model_choice, task, feature_hash)

    # ── Feature Importance ─────────────────────────────────────────────────────
    with tab_importance:
        st.subheader("Feature Importance (Built-in)")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            imp_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importances,
            }).sort_values("Importance", ascending=False)

            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(height=max(400, len(feature_cols) * 25),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")

            st.dataframe(imp_df, width="stretch", hide_index=True)

            csv_data = imp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Importance CSV",
                data=csv_data,
                file_name="feature_importance.csv",
                mime="text/csv",
            )
        else:
            st.info("This model does not have built-in feature importances. Check SHAP tab.")

        # Permutation importance
        st.subheader("Permutation Importance")
        if st.button("Compute Permutation Importance"):
            from sklearn.inspection import permutation_importance
            with st.spinner("Computing..."):
                if task == "Classification" and is_binary:
                    scoring = "roc_auc"
                elif task == "Classification":
                    scoring = "accuracy"
                else:
                    scoring = "r2"
                result = permutation_importance(model, X_scaled, y, n_repeats=10,
                                                random_state=42, scoring=scoring, n_jobs=-1)
                perm_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance Mean": result.importances_mean,
                    "Importance Std": result.importances_std,
                }).sort_values("Importance Mean", ascending=False)

                fig = px.bar(perm_df, x="Importance Mean", y="Feature", orientation="h",
                             error_x="Importance Std",
                             color="Importance Mean", color_continuous_scale="Plasma")
                fig.update_layout(height=max(400, len(feature_cols) * 25),
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")

    # ── SHAP ───────────────────────────────────────────────────────────────────
    with tab_shap:
        st.subheader("SHAP Analysis")
        st.info("SHAP (SHapley Additive exPlanations) provides both global and local model interpretability.")

        max_samples = st.slider("Background samples (lower = faster)", 50, 500, 100)

        if st.button("Compute SHAP Values"):
            import shap

            with st.spinner("Computing SHAP values (this may take a moment)..."):
                # Use appropriate explainer
                sample_idx = np.random.RandomState(42).choice(len(X_scaled), min(max_samples, len(X_scaled)), replace=False)
                X_sample = X_scaled[sample_idx]

                if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    explainer = shap.KernelExplainer(model.predict, X_sample[:50])
                    shap_values = explainer.shap_values(X_sample)

                # Handle multi-class
                if isinstance(shap_values, list):
                    shap_vals = np.abs(np.array(shap_values)).mean(axis=0)
                else:
                    shap_vals = shap_values

                # Summary bar plot
                st.markdown("#### Global Feature Importance (SHAP)")
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                shap_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Mean |SHAP|": mean_abs_shap,
                }).sort_values("Mean |SHAP|", ascending=False)

                fig = px.bar(shap_df, x="Mean |SHAP|", y="Feature", orientation="h",
                             color="Mean |SHAP|", color_continuous_scale="Reds")
                fig.update_layout(height=max(400, len(feature_cols) * 25),
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")

                shap_csv = shap_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download SHAP Summary CSV",
                    data=shap_csv,
                    file_name="shap_importance.csv",
                    mime="text/csv",
                )

                # Beeswarm / scatter
                st.markdown("#### SHAP Beeswarm Plot")
                fig, ax = plt.subplots(figsize=(10, max(6, len(feature_cols) * 0.4)))
                shap.summary_plot(shap_vals, X_sample, feature_names=feature_cols, show=False)
                st.pyplot(fig)
                plt.close()

                # Individual prediction explanation
                st.markdown("#### Explain a Single Prediction")
                pred_idx = st.number_input("Sample index", 0, len(X_sample) - 1, 0)
                fig, ax = plt.subplots(figsize=(10, 4))
                if isinstance(shap_values, list):
                    shap.force_plot(explainer.expected_value[0], shap_values[0][pred_idx],
                                    features=X_sample[pred_idx], feature_names=feature_cols,
                                    matplotlib=True, show=False)
                else:
                    shap.force_plot(explainer.expected_value, shap_values[pred_idx],
                                    features=X_sample[pred_idx], feature_names=feature_cols,
                                    matplotlib=True, show=False)
                st.pyplot(plt.gcf())
                plt.close()

    # ── Partial Dependence ─────────────────────────────────────────────────────
    with tab_pdp:
        st.subheader("Partial Dependence Plots")
        st.info("Shows how each feature affects the model's prediction, holding other features constant.")

        sel_features = st.multiselect("Features to plot", feature_cols,
                                       default=feature_cols[:3], key="pdp_feats")

        if sel_features and st.button("Generate PDP"):
            from sklearn.inspection import PartialDependenceDisplay

            with st.spinner("Computing partial dependence..."):
                feature_indices = [feature_cols.index(f) for f in sel_features]

                fig, axes = plt.subplots(1, len(sel_features),
                                         figsize=(5 * len(sel_features), 4))
                if len(sel_features) == 1:
                    axes = [axes]

                PartialDependenceDisplay.from_estimator(
                    model, X_scaled, feature_indices,
                    feature_names=feature_cols,
                    ax=axes,
                    kind="average",
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # 2D interaction
        st.markdown("#### 2D Feature Interaction")
        if len(feature_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                feat_a = st.selectbox("Feature A", feature_cols, key="pdp_a")
            with c2:
                remaining = [f for f in feature_cols if f != feat_a]
                feat_b = st.selectbox("Feature B", remaining, key="pdp_b")

            if st.button("Generate 2D PDP"):
                from sklearn.inspection import PartialDependenceDisplay

                with st.spinner("Computing 2D partial dependence..."):
                    idx_a = feature_cols.index(feat_a)
                    idx_b = feature_cols.index(feat_b)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    PartialDependenceDisplay.from_estimator(
                        model, X_scaled, [(idx_a, idx_b)],
                        feature_names=feature_cols,
                        ax=ax,
                        kind="average",
                    )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Model Explainability — Understand Any Model's Decisions

This page provides three complementary approaches to explain how a trained model makes predictions.

---

#### Setup
- **Target column** — the variable being predicted.
- **Task type** — auto-detected as Classification or Regression, with manual override.
- **Model selection** — choose from Random Forest, XGBoost, LightGBM, or Gradient Boosting. The model is trained automatically with a StandardScaler on all numeric features.

#### Feature Importance Tab
- **Built-in Feature Importance** — uses the model's native `feature_importances_` attribute (based on how much each feature reduces impurity across all trees). Displayed as a ranked horizontal bar chart and a data table.
- **Permutation Importance** — a model-agnostic method that measures how much the model's score **drops** when a single feature's values are randomly shuffled. A large drop means the feature is important. This method is more reliable than built-in importance because it accounts for feature interactions and is not biased toward high-cardinality features.

#### SHAP Analysis Tab
- **SHAP (SHapley Additive exPlanations)** — based on cooperative game theory (Shapley values). Each feature's contribution to a prediction is computed fairly, considering all possible feature combinations.
- **Global Bar Chart (Mean |SHAP|)** — shows the average absolute SHAP value per feature across all samples. Higher values indicate features with more influence on predictions overall.
- **Beeswarm Plot** — each dot represents a single sample. The x-axis shows the SHAP value (impact on prediction), and color indicates the feature's actual value (red = high, blue = low). This reveals both the **direction** and **magnitude** of each feature's effect.
- **Single Prediction Waterfall** — explains one individual prediction by showing how each feature pushes the prediction away from the baseline (expected value). Useful for understanding specific decisions.
- **Background samples** slider controls the sample size used for SHAP computation (lower = faster, higher = more accurate).

#### Partial Dependence Tab
- **1D Partial Dependence Plots (PDP)** — show the **marginal effect** of a single feature on the model's prediction, averaged over all other features. The x-axis is the feature value, and the y-axis is the predicted outcome. Useful for understanding non-linear relationships.
- **2D Feature Interaction PDP** — shows how **two features jointly** affect the prediction. Displayed as a contour/heatmap plot. Useful for identifying interaction effects between feature pairs.

#### Notes
- All models are trained with a **StandardScaler** to normalize features before fitting.
- SHAP uses **TreeExplainer** for tree-based models (fast, exact) and falls back to **KernelExplainer** for others (slower, approximate).
        """)

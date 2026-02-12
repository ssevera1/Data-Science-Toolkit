import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from time import time
from utils.theme import page_header


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


def render():
    page_header("Model Arena", "Benchmark 10+ algorithms side-by-side with proper cross-validation. Find your best model in one click.", "🏟️")

    _guard()
    df = st.session_state["df"].copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if len(all_cols) < 2:
        st.warning("Need at least 2 columns (1 target + 1 feature).")
        st.stop()

    # ── Setup ──────────────────────────────────────────────────────────────────
    st.subheader("Configuration")
    c1, c2, c3 = st.columns(3)

    with c1:
        target = st.selectbox("Target column", all_cols)
    with c2:
        target_is_categorical = not pd.api.types.is_numeric_dtype(df[target])
        is_clf = target_is_categorical or df[target].nunique() <= 20
        if target_is_categorical:
            task_options = ["Classification"]
        else:
            task_options = ["Auto-detect", "Classification", "Regression"]
        task = st.selectbox("Task type", task_options)
        if task == "Auto-detect":
            task = "Classification" if is_clf else "Regression"
        n_classes = df[target].nunique()
        is_binary = task == "Classification" and n_classes == 2
        label = f"**{task}** — {'binary' if is_binary else f'{n_classes} classes' if task == 'Classification' else 'continuous'}"
        st.write(label)
    with c3:
        cv_folds = st.slider("CV Folds", 2, 10, 5)

    feature_cols = [c for c in num_cols if c != target]
    if not feature_cols:
        st.warning("No numeric feature columns available. Encode categorical features first (Smart Cleaning page).")
        st.stop()

    X = df[feature_cols].fillna(0).values
    y = df[target]

    # ── Model Selection ────────────────────────────────────────────────────────
    st.subheader("Select Models")

    if task == "Classification":
        available_models = {
            "Logistic Regression": True,
            "Random Forest": True,
            "Gradient Boosting": True,
            "XGBoost": True,
            "LightGBM": True,
            "SVM (RBF)": True,
            "K-Nearest Neighbors": True,
            "Decision Tree": True,
            "Naive Bayes": True,
            "AdaBoost": True,
            "Extra Trees": True,
        }
    else:
        available_models = {
            "Linear Regression": True,
            "Ridge Regression": True,
            "Lasso Regression": True,
            "ElasticNet": True,
            "Random Forest": True,
            "Gradient Boosting": True,
            "XGBoost": True,
            "LightGBM": True,
            "SVR (RBF)": True,
            "K-Nearest Neighbors": True,
            "Decision Tree": True,
            "AdaBoost": True,
            "Extra Trees": True,
        }

    selected_models = st.multiselect(
        "Models to benchmark",
        list(available_models.keys()),
        default=list(available_models.keys())[:7],
    )

    # ── Run Benchmark ──────────────────────────────────────────────────────────
    if st.button("Run Benchmark", type="primary"):
        from sklearn.model_selection import cross_validate
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.pipeline import Pipeline

        # Encode target
        if task == "Classification":
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
        else:
            y_enc = y.fillna(0).astype(float).values

        # Metrics
        if task == "Classification" and is_binary:
            scoring = {
                "accuracy": "accuracy",
                "f1": "f1",
                "precision": "precision",
                "recall": "recall",
                "roc_auc": "roc_auc",
            }
        elif task == "Classification":
            scoring = {
                "accuracy": "accuracy",
                "f1_weighted": "f1_weighted",
                "precision_weighted": "precision_weighted",
                "recall_weighted": "recall_weighted",
            }
        else:
            scoring = {
                "r2": "r2",
                "neg_mse": "neg_mean_squared_error",
                "neg_mae": "neg_mean_absolute_error",
            }

        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, model_name in enumerate(selected_models):
            status.write(f"Training **{model_name}**...")
            progress.progress((i) / len(selected_models))

            try:
                model = _get_model(model_name, task) if False else None  # placeholder

                # Build model
                if task == "Classification":
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                                  AdaBoostClassifier, ExtraTreesClassifier)
                    from sklearn.svm import SVC
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.naive_bayes import GaussianNB

                    model_map = {
                        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                        "SVM (RBF)": SVC(random_state=42),
                        "K-Nearest Neighbors": KNeighborsClassifier(),
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                        "Naive Bayes": GaussianNB(),
                        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
                        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    }
                    try:
                        from xgboost import XGBClassifier
                        model_map["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42,
                                                              use_label_encoder=False, eval_metric="mlogloss",
                                                              verbosity=0)
                    except ImportError:
                        pass
                    try:
                        from lightgbm import LGBMClassifier
                        model_map["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                    except ImportError:
                        pass

                else:
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
                    from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                                                  AdaBoostRegressor, ExtraTreesRegressor)
                    from sklearn.svm import SVR
                    from sklearn.neighbors import KNeighborsRegressor
                    from sklearn.tree import DecisionTreeRegressor

                    model_map = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(random_state=42),
                        "Lasso Regression": Lasso(random_state=42),
                        "ElasticNet": ElasticNet(random_state=42),
                        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                        "SVR (RBF)": SVR(),
                        "K-Nearest Neighbors": KNeighborsRegressor(),
                        "Decision Tree": DecisionTreeRegressor(random_state=42),
                        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
                        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                    }
                    try:
                        from xgboost import XGBRegressor
                        model_map["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                    except ImportError:
                        pass
                    try:
                        from lightgbm import LGBMRegressor
                        model_map["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
                    except ImportError:
                        pass

                if model_name not in model_map:
                    continue

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", model_map[model_name]),
                ])

                t0 = time()
                cv_results = cross_validate(pipe, X, y_enc, cv=cv_folds, scoring=scoring,
                                            return_train_score=False, n_jobs=-1)
                elapsed = time() - t0

                row = {"Model": model_name, "Time (s)": round(elapsed, 2)}
                for key in scoring:
                    scores = cv_results[f"test_{key}"]
                    display_key = key.replace("neg_", "").replace("_", " ").title()
                    val = scores.mean()
                    if "neg_" in key:
                        val = -val
                        display_key = display_key
                    row[display_key] = round(val, 4)
                    row[f"{display_key} Std"] = round(scores.std(), 4)

                results.append(row)

            except Exception:
                results.append({"Model": model_name, "Error": "Training failed"})

        progress.progress(1.0)
        status.write("Benchmark complete!")

        # ── Results ────────────────────────────────────────────────────────────
        if results:
            res_df = pd.DataFrame(results)
            st.session_state["arena_results"] = res_df

            # Highlight best
            st.subheader("Results")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            # Primary metric chart
            if task == "Classification" and is_binary:
                primary = "Roc Auc"
            elif task == "Classification":
                primary = "Accuracy"
            else:
                primary = "R2"

            if primary in res_df.columns:
                valid = res_df.dropna(subset=[primary]).sort_values(primary, ascending=False)
                fig = px.bar(valid, x="Model", y=primary, color=primary,
                             color_continuous_scale="Viridis", text_auto=".4f",
                             title=f"Model Comparison — {primary}")
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

                best = valid.iloc[0]
                st.success(f"Best model: **{best['Model']}** with {primary} = {best[primary]:.4f}")

            # Radar chart for classification
            if task == "Classification" and primary in res_df.columns and len(valid) > 1:
                if is_binary:
                    metrics = ["Accuracy", "F1", "Precision", "Recall", "Roc Auc"]
                else:
                    metrics = ["Accuracy", "F1 Weighted", "Precision Weighted", "Recall Weighted"]
                avail_metrics = [m for m in metrics if m in valid.columns]
                if len(avail_metrics) >= 3:
                    fig = go.Figure()
                    for _, row in valid.head(5).iterrows():
                        values = [row.get(m, 0) for m in avail_metrics]
                        values.append(values[0])  # close the polygon
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=avail_metrics + [avail_metrics[0]],
                            name=row["Model"],
                        ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        height=500,
                        title="Top 5 Models — Radar Chart",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Store best model info
            if primary in res_df.columns:
                best_name = valid.iloc[0]["Model"]
                st.session_state["best_model_name"] = best_name
                st.session_state["model_map_task"] = task
                st.session_state["model_X"] = X
                st.session_state["model_y"] = y_enc
                st.session_state["feature_cols"] = feature_cols

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Model Arena — Benchmark Multiple Algorithms Side-by-Side

This page trains and compares 10+ machine learning models using proper cross-validation to find the best one for your data.

---

#### Configuration
- **Target column** — the variable to predict.
- **Task type** — auto-detected as **Classification** (categorical target or <= 20 unique values) or **Regression** (continuous target). Override available.
- **CV Folds** — number of cross-validation folds (2-10, default 5). Higher values give more robust estimates but take longer.

#### Model Selection
- **Classification models:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM (RBF), K-Nearest Neighbors, Decision Tree, Naive Bayes, AdaBoost, Extra Trees.
- **Regression models:** Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVR (RBF), K-Nearest Neighbors, Decision Tree, AdaBoost, Extra Trees.
- Select or deselect individual models before benchmarking.

#### Metrics
- **Binary Classification:** Accuracy, F1 Score, Precision, Recall, ROC AUC.
- **Multiclass Classification:** Accuracy, F1 Weighted, Precision Weighted, Recall Weighted.
- **Regression:** R-squared (R2), Mean Squared Error (MSE), Mean Absolute Error (MAE).
- Both the mean and standard deviation across CV folds are reported for each metric.

#### Results
- **Comparison Table** — all models ranked with their metric scores and training time.
- **Primary Metric Bar Chart** — visual comparison using the most important metric (ROC AUC for binary classification, Accuracy for multiclass, R2 for regression).
- **Radar Chart** (classification only) — overlays the top 5 models across all metrics on a polar chart for easy multi-metric comparison.
- **Best Model** — highlighted with its primary metric score.

#### Pipeline Details
- All models use a **StandardScaler** preprocessing step to normalize features before training.
- **Cross-validation** ensures results are not biased by a particular train/test split.
- The best model info is stored in the session for use in other tools (e.g., Hyperparameter Tuning).
        """)

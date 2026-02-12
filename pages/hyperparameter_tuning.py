import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.theme import page_header


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


def render():
    page_header("Hyperparameter Tuning", "Bayesian optimization via Optuna — find optimal hyperparameters with live trial visualization.", "🎛️")

    _guard()
    df = st.session_state["df"].copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if len(all_cols) < 2:
        st.warning("Need at least 2 columns (1 target + 1 feature).")
        st.stop()

    # ── Setup ──────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        target = st.selectbox("Target column", all_cols, key="hp_target")
    with c2:
        target_is_categorical = not pd.api.types.is_numeric_dtype(df[target])
        is_clf = target_is_categorical or df[target].nunique() <= 20
        if target_is_categorical:
            task_options = ["Classification"]
        else:
            task_options = ["Auto-detect", "Classification", "Regression"]
        task = st.selectbox("Task type", task_options, key="hp_task")
        if task == "Auto-detect":
            task = "Classification" if is_clf else "Regression"
        n_classes = df[target].nunique()
        is_binary = task == "Classification" and n_classes == 2
        label = f"**{task}** — {'binary' if is_binary else f'{n_classes} classes' if task == 'Classification' else 'continuous'}"
        st.write(label)
    with c3:
        if task == "Classification" and is_binary:
            metric_options = ["roc_auc", "accuracy", "f1", "precision", "recall"]
        elif task == "Classification":
            metric_options = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
        else:
            metric_options = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
        primary_metric = st.selectbox("Optimization metric", metric_options, key="hp_metric")

    feature_cols = [c for c in num_cols if c != target]
    if not feature_cols:
        st.warning("No numeric feature columns available. Encode categorical features first (Smart Cleaning page).")
        st.stop()

    X = df[feature_cols].fillna(0).values
    y = df[target]

    if task == "Classification":
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.fillna(0).astype(float).values

    model_choice = st.selectbox("Model to tune", [
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Gradient Boosting",
        "SVM / SVR",
        "K-Nearest Neighbors",
    ])

    n_trials = st.slider("Number of Optuna trials", 10, 200, 50)
    cv_folds = st.slider("CV folds", 2, 10, 5, key="hp_cv")

    # ── Run Tuning ─────────────────────────────────────────────────────────────
    if st.button("Start Tuning", type="primary"):
        import optuna
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        trial_history = []

        def objective(trial):
            if model_choice == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                }
                cls = RandomForestClassifier if task == "Classification" else RandomForestRegressor
                model = cls(**params, random_state=42, n_jobs=-1)

            elif model_choice == "XGBoost":
                from xgboost import XGBClassifier, XGBRegressor
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }
                cls = XGBClassifier if task == "Classification" else XGBRegressor
                model = cls(**params, random_state=42, verbosity=0, use_label_encoder=False)

            elif model_choice == "LightGBM":
                from lightgbm import LGBMClassifier, LGBMRegressor
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 10, 200),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                }
                cls = LGBMClassifier if task == "Classification" else LGBMRegressor
                model = cls(**params, random_state=42, verbose=-1)

            elif model_choice == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                }
                cls = GradientBoostingClassifier if task == "Classification" else GradientBoostingRegressor
                model = cls(**params, random_state=42)

            elif model_choice == "SVM / SVR":
                from sklearn.svm import SVC, SVR
                params = {
                    "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                    "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                }
                model = SVC(**params, random_state=42) if task == "Classification" else SVR(**params)

            elif model_choice == "K-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
                params = {
                    "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
                    "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                    "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"]),
                }
                cls = KNeighborsClassifier if task == "Classification" else KNeighborsRegressor
                model = cls(**params)

            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            scores = cross_val_score(pipe, X, y, cv=cv_folds, scoring=primary_metric, n_jobs=-1)
            score = scores.mean()

            trial_history.append({
                "trial": trial.number,
                "score": score,
                "params": trial.params.copy(),
            })

            return score

        progress = st.progress(0)
        status = st.empty()

        study = optuna.create_study(direction="maximize")

        # Callback to update progress
        def callback(study, trial):
            progress.progress(min(1.0, (trial.number + 1) / n_trials))
            status.write(f"Trial {trial.number + 1}/{n_trials} — Best so far: {study.best_value:.4f}")

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
        progress.progress(1.0)

        # ── Results ────────────────────────────────────────────────────────────
        st.subheader("Results")
        display_metric = primary_metric.replace("neg_", "").replace("_", " ").title()
        best_display = -study.best_value if primary_metric.startswith("neg_") else study.best_value
        st.success(f"Best {display_metric}: **{best_display:.4f}**")

        st.markdown("#### Best Hyperparameters")
        best_params = study.best_params
        st.json(best_params)

        # Optimization history
        st.markdown("#### Optimization History")
        hist_df = pd.DataFrame(trial_history)
        hist_df["best_so_far"] = hist_df["score"].cummax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_df["trial"], y=hist_df["score"],
                                 mode="markers", name="Trial Score", opacity=0.5))
        fig.add_trace(go.Scatter(x=hist_df["trial"], y=hist_df["best_so_far"],
                                 mode="lines", name="Best So Far",
                                 line=dict(color="red", width=2)))
        fig.update_layout(height=400, xaxis_title="Trial", yaxis_title=primary_metric.title())
        st.plotly_chart(fig, width="stretch")

        # Parameter importance
        st.markdown("#### Parameter Importance")
        try:
            importance = optuna.importance.get_param_importances(study)
            imp_df = pd.DataFrame({
                "Parameter": list(importance.keys()),
                "Importance": list(importance.values()),
            })
            fig = px.bar(imp_df, x="Importance", y="Parameter", orientation="h",
                         color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
        except Exception:
            st.info("Parameter importance not available for this study.")

        # Parallel coordinate plot
        st.markdown("#### Parallel Coordinate Plot")
        try:
            params_df = pd.DataFrame([t["params"] for t in trial_history])
            params_df["score"] = [t["score"] for t in trial_history]

            numeric_params = params_df.select_dtypes(include="number").columns.tolist()
            if len(numeric_params) >= 2:
                fig = px.parallel_coordinates(
                    params_df[numeric_params],
                    color="score",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, width="stretch")
        except Exception:
            pass

        # Download trial history
        st.markdown("#### Export")
        dl1, dl2 = st.columns(2)
        with dl1:
            hist_csv = hist_df.drop(columns=["params"], errors="ignore").to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Trial History CSV",
                data=hist_csv,
                file_name="tuning_trial_history.csv",
                mime="text/csv",
                width="stretch",
            )
        with dl2:
            import json
            params_json = json.dumps(best_params, indent=2).encode("utf-8")
            st.download_button(
                "Download Best Params JSON",
                data=params_json,
                file_name="best_hyperparameters.json",
                mime="application/json",
                width="stretch",
            )

        # Store best params
        st.session_state["best_params"] = best_params
        st.session_state["tuned_model"] = model_choice

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Hyperparameter Tuning — Bayesian Optimization with Optuna

This page uses Optuna's Bayesian optimization to efficiently search for the best hyperparameters for a chosen model.

---

#### Configuration
- **Target column** — the variable to predict.
- **Task type** — auto-detected (Classification or Regression), with manual override.
- **Optimization metric** — the metric to maximize during tuning:
  - Binary Classification: ROC AUC, Accuracy, F1, Precision, Recall.
  - Multiclass Classification: Accuracy, F1 Weighted, Precision Weighted, Recall Weighted.
  - Regression: R2, Neg. MSE, Neg. MAE.
- **Model to tune** — choose from Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM/SVR, or K-Nearest Neighbors.
- **Number of trials** — how many hyperparameter combinations to evaluate (10-200, default 50). More trials generally find better parameters but take longer.
- **CV folds** — cross-validation folds per trial (2-10, default 5).

#### Optuna Bayesian Optimization
- Uses the **Tree-structured Parzen Estimator (TPE)** algorithm, which builds a probabilistic model of the objective function to intelligently select the next hyperparameters to try.
- Unlike grid search (exhaustive) or random search (blind), TPE **focuses on promising regions** of the hyperparameter space, finding good parameters faster.

#### Supported Models and Tunable Parameters
- **Random Forest** — n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.
- **XGBoost** — n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda.
- **LightGBM** — n_estimators, max_depth, learning_rate, num_leaves, subsample, colsample_bytree, reg_alpha, reg_lambda.
- **Gradient Boosting** — n_estimators, max_depth, learning_rate, subsample, min_samples_split.
- **SVM / SVR** — C (regularization), kernel (rbf/poly/sigmoid), gamma (scale/auto).
- **K-Nearest Neighbors** — n_neighbors, weights (uniform/distance), metric (euclidean/manhattan/minkowski).

#### Results
- **Best Hyperparameters** — displayed as JSON, showing the optimal values found.
- **Optimization History** — scatter plot of each trial's score with a cumulative best line (red), showing how the search converges.
- **Parameter Importance** — bar chart showing which hyperparameters had the most impact on performance.
- **Parallel Coordinate Plot** — visualizes relationships between numeric hyperparameters and the objective score across all trials.

#### Pipeline Details
- Each trial trains the model inside a **StandardScaler + Model pipeline** with cross-validation.
- Best parameters are stored in the session for reference.
        """)

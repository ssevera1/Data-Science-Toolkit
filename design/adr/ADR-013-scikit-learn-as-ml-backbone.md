# ADR-013: Chosen scikit-learn as the ML Backbone with XGBoost/LightGBM Extensions

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Technology / ML Pipeline |

## Context

The Data Science toolset includes model benchmarking (Model Arena), feature
selection, class imbalance handling, and data drift detection. These capabilities
require a machine learning framework that supports multiple algorithms, consistent
APIs, and integration with the broader Python data science ecosystem.

### Options Considered

| Framework | Pros | Cons |
|---|---|---|
| **scikit-learn only** | Standard API, huge algorithm library, well-documented | Missing gradient boosting state-of-the-art (XGBoost, LightGBM) |
| **scikit-learn + XGBoost + LightGBM** | Best algorithms available, sklearn-compatible API | More dependencies, version management |
| **PyTorch / TensorFlow** | Deep learning, GPU support | Overkill for tabular data, heavy dependencies, complex setup |
| **AutoML (auto-sklearn, H2O)** | Fully automated | Black box, heavy dependencies, less educational |
| **PyCaret** | High-level ML wrapper | Hides complexity, dependency-heavy, less control |

## Decision

**Chosen: scikit-learn as the primary ML framework**, with XGBoost and LightGBM
as sklearn-compatible extensions for gradient boosting.

## Rationale

1. **Estimator API consistency**: scikit-learn's estimator interface (`fit()`,
   `predict()`, `score()`, `get_params()`) is the de facto standard. XGBoost
   and LightGBM provide sklearn-compatible wrappers (`XGBClassifier`,
   `LGBMClassifier`), allowing all 10+ models to be trained with identical code.

2. **Model Arena design**: The Model Arena page benchmarks models in a loop.
   The sklearn API makes this possible with a generic pattern:
   ```python
   for name, model in models.items():
       scores = cross_val_score(model, X, y, cv=5, scoring=metric)
   ```
   Without API consistency, each model would need custom training code.

3. **Preprocessing pipeline**: scikit-learn's `Pipeline` and `ColumnTransformer`
   enable composable preprocessing (scaling, encoding, imputation) that is
   applied consistently during cross-validation, preventing data leakage.

4. **SHAP compatibility**: The SHAP library integrates directly with
   sklearn-compatible models via `shap.Explainer()`. This enables the
   Explainability page to work with any model from Model Arena without
   model-specific SHAP configuration.

5. **No GPU requirement**: All selected algorithms run on CPU, consistent with
   the local-only architecture. Users don't need CUDA or specialized hardware.

## Supported Algorithms

### Classifiers (Model Arena)
| Algorithm | Library | Key Hyperparameters |
|---|---|---|
| Logistic Regression | sklearn | C, penalty, solver |
| Random Forest | sklearn | n_estimators, max_depth, min_samples_split |
| Gradient Boosting | sklearn | n_estimators, learning_rate, max_depth |
| XGBoost | xgboost | n_estimators, learning_rate, max_depth, subsample |
| LightGBM | lightgbm | n_estimators, learning_rate, num_leaves |
| SVM | sklearn | C, kernel, gamma |
| KNN | sklearn | n_neighbors, weights, metric |
| Decision Tree | sklearn | max_depth, min_samples_split, criterion |
| Naive Bayes | sklearn | var_smoothing |
| AdaBoost | sklearn | n_estimators, learning_rate |
| Extra Trees | sklearn | n_estimators, max_depth |

### Regressors (Model Arena)
| Algorithm | Library | Key Hyperparameters |
|---|---|---|
| Linear Regression | sklearn | (none - baseline) |
| Ridge | sklearn | alpha |
| Lasso | sklearn | alpha |
| ElasticNet | sklearn | alpha, l1_ratio |
| Random Forest | sklearn | n_estimators, max_depth |
| Gradient Boosting | sklearn | n_estimators, learning_rate |
| XGBoost | xgboost | n_estimators, learning_rate, max_depth |
| LightGBM | lightgbm | n_estimators, learning_rate, num_leaves |
| SVR | sklearn | C, kernel, epsilon |
| KNN | sklearn | n_neighbors, weights |
| Decision Tree | sklearn | max_depth, min_samples_split |
| AdaBoost | sklearn | n_estimators, learning_rate |
| Extra Trees | sklearn | n_estimators, max_depth |

## Trade-offs Accepted

- **No deep learning**: Neural networks (MLP, CNNs, transformers) are not
  included. For the application's target use case (tabular data, < 200 MB),
  tree-based methods typically outperform neural networks anyway.

- **CPU-only**: No GPU acceleration. This limits performance for very large
  datasets but eliminates hardware requirements and CUDA dependency management.

- **Dependency weight**: XGBoost (~150 MB) and LightGBM (~50 MB) are substantial
  dependencies. They are justified by consistently providing the best performance
  on tabular data benchmarks.

- **No model persistence**: Trained models are not saved to disk. They exist only
  in session state during the session. Users cannot deploy models directly from
  the application. This keeps the tool focused on analysis rather than deployment.

## Consequences

- All models share a consistent API, enabling the Model Arena's generic benchmarking loop
- SHAP explainability works with any model selected from the Arena
- Hyperparameter tuning (Optuna) can optimize any sklearn-compatible estimator
- Feature selection (RFE) works with any estimator that exposes `feature_importances_` or `coef_`
- Class imbalance handling (imbalanced-learn) integrates via sklearn Pipelines

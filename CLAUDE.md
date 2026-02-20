# CLAUDE.md - DS Power Tools

## Project Overview

DS Power Tools is a combined Data Science & Statistics Toolkit built with Streamlit.
It runs 100% locally with zero external API calls, no cloud dependencies, and no telemetry.
Created by Scott Severance. Licensed under MIT.

## Quick Reference

- **Run**: `streamlit run app.py` (or `launch.bat` / `launch.sh`)
- **Python**: 3.9+
- **Entry point**: `app.py`
- **Dependencies**: `pip install -r requirements.txt`

## Architecture

```
app.py                      Entry point: theme, navigation, state init
pages/       (32 modules)   View layer — one page per tool/test
core/        (4 modules)    State management, data I/O, validation, constants
stats/       (13 modules)   Statistical computation (pure functions, no Streamlit imports)
charts/      (7 modules)    Plotly chart generation
components/  (4 modules)    Reusable UI widgets (variable selectors, result cards)
utils/       (2 modules)    Dual-theme CSS system
design/                     C4 architecture diagrams and ADRs
```

### Key Architectural Rules

- **stats/ modules must NEVER import streamlit, plotly, or components** — they are pure computation
- **pages/ should never call scipy/statsmodels directly** — always go through stats/ modules
- **All statistical functions return a dict** with keys: `test_name`, `statistic`, `p_value`, `effect_size`, `effect_size_label`, `interpretation`, `assumptions`
- **Shared state lives in `st.session_state`** — the DataFrame is at `st.session_state["df"]`, variable types at `st.session_state["stats_var_types"]`
- **Theme is at `st.session_state["app_theme"]`** — either "Light" or "Dark"

### Data Flow

```
User uploads file → core/data_manager.py loads + auto-detects types
  → st.session_state["df"] shared across all 32 pages
    → pages/ call stats/ for computation, charts/ for visualization
      → components/results_display.py renders formatted output
```

## Conventions

### Adding a New Statistical Test

1. Create a function in the appropriate `stats/` module (or new module) returning the standard result dict
2. Create a new page in `pages/` following the existing pattern: guard → header → preview → variable selection → validation → run → tabs (Results | Assumptions | Charts) → guide
3. Register the page in `app.py` under the appropriate navigation section
4. Always include assumption checks and effect sizes — they are mandatory, not optional

### Adding a New Chart Type

1. Create a function in `charts/` that takes `(df, columns, theme)` and returns a `plotly.graph_objects.Figure`
2. Use `charts/theme.py` to get the active Plotly template — never hardcode colors

### Page Module Pattern (Statistics)

Every statistics page follows this structure:
- Guard check (is `df` loaded?)
- Page title and description
- Data preview (collapsible)
- Variable selection via `components/variable_selector.py`
- Input validation via `core/validators.py`
- Three tabs: Results, Assumptions, Charts
- Expandable page guide at bottom

### Theming

- Two themes: Light (cream/red) and Dark (navy/cyan)
- UI CSS: `utils/theme.py` injects via `st.markdown(unsafe_allow_html=True)`
- Chart CSS: `charts/theme.py` registers Plotly templates
- Both must be updated together when changing theme colors
- CSS selectors target Streamlit's internal DOM — fragile across Streamlit upgrades

## Important Files

| File | Purpose |
|---|---|
| `app.py` | Entry point, all page registration, sidebar |
| `core/state.py` | `init_state()`, session state defaults |
| `core/data_manager.py` | File loading, variable type auto-detection, export with formula injection prevention |
| `core/validators.py` | Input validation functions |
| `core/constants.py` | Alpha=0.05, variable types, test categories |
| `stats/assumptions.py` | Shapiro-Wilk, Levene's, Mauchly's, Henze-Zirkler, Box's M |
| `stats/effect_size.py` | Cohen's d, eta-squared, omega-squared, Cramer's V, rank-biserial |
| `components/results_display.py` | Formatted result cards with significance coloring |
| `components/variable_selector.py` | Type-aware column dropdowns |
| `utils/theme.py` | Global CSS injection for dual themes |
| `charts/theme.py` | Plotly template registration per theme |
| `.streamlit/config.toml` | Server config: localhost-only, 200MB upload limit, no telemetry |

## Common Pitfalls

- **DataFrame fragmentation**: Use `.copy()` when subsetting DataFrames to avoid `SettingWithCopyWarning` and fragmentation. Never chain `df[col1][col2]` — use `df.loc[]` instead. When adding many columns, batch-build via `pd.DataFrame(dict)` + `pd.concat` instead of column-by-column assignment.
- **Streamlit reruns**: Every widget interaction triggers a full rerun. Guard heavy computations behind `st.button()` clicks, not bare function calls.
- **Theme CSS selectors**: Streamlit changes internal DOM class names between versions. After upgrading Streamlit, test both themes thoroughly.
- **Variable type detection**: The auto-detection heuristic uses fixed thresholds (e.g., <=2 unique numeric → Nominal, 3-7 sparse → Ordinal). These won't be correct for all datasets.
- **Effect sizes are mandatory**: Every statistical test must report an appropriate effect size. This is a deliberate design decision, not optional.
- **No external network calls**: The app must never make HTTP requests, download models, or phone home. This is a core privacy guarantee.
- **CSV formula injection**: Every CSV export **must** go through `_sanitize_csv()` which prefixes formula-trigger characters (`= + - @ \t \r`) with `'`. Never use bare `.to_csv()` in a download button. This is already enforced in all pages — maintain it when adding new exports.
- **KNN on large datasets**: When using `sklearn.NearestNeighbors`, fit on the smallest applicable subset (e.g., minority class only for SMOTE categoricals, same-class only for Tomek links). Fitting on the full dataset with 20+ dimensions triggers O(n×m) brute-force computation that scales catastrophically. The SMOTE categorical KNN in `class_imbalance.py` uses per-class fitting for this reason.
- **Prefer numpy indexing over pandas `.iloc`**: For bulk index lookups (e.g., `df[col].iloc[indices].values`), use `df[col].values[indices]` instead. Direct numpy fancy indexing avoids pandas overhead and is ~10x faster per call, which compounds over many columns.

## Library Usage

| Library | Used For |
|---|---|
| SciPy | t-tests, Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, chi-squared, binomial, Shapiro-Wilk, Levene's, correlations |
| statsmodels | Linear/logistic regression (OLS, Logit), two-way ANOVA, MANOVA, Tukey HSD |
| pingouin | Repeated/mixed ANOVA, Mauchly's sphericity, Box's M, Henze-Zirkler multivariate normality |
| scikit-learn | Model Arena algorithms, preprocessing, cross-validation, feature selection |
| imbalanced-learn | SMOTE oversampling (class_imbalance.py); Tomek links replaced with custom optimized implementation |
| XGBoost / LightGBM | Gradient boosting models (sklearn-compatible API) |
| Optuna | Bayesian hyperparameter optimization |
| SHAP | Model explainability |
| Plotly | All interactive charts in statistics tools |
| Matplotlib / Seaborn | Only where upstream libraries (SHAP, sklearn) output them directly |

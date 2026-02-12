# DS Power Tools

**Combined Data Science & Statistics Toolkit** — a single Streamlit application that brings together automated data science workflows and a full suite of statistical tests.

*Created by Scott Severance*

---

## Features

### Data Science Tools

| Tool | Description |
|------|-------------|
| **Data Profiler** | Automated EDA — distributions, correlations, missing patterns, outlier detection |
| **Smart Cleaning** | One-click imputation, outlier treatment, encoding, deduplication |
| **Feature Engineering** | Auto-generate polynomial, interaction, datetime & binned features |
| **Feature Selection** | Correlation filters, mutual info, variance threshold, RFE ranking |
| **Class Imbalance** | SMOTE, random over/under-sampling, distribution comparison |
| **Model Arena** | Benchmark 10+ classifiers/regressors with cross-validation |
| **Hyperparameter Tuning** | Bayesian optimization via Optuna with live visualizations |
| **Explainability** | SHAP values, feature importance, partial dependence plots |
| **Data Drift** | KS test, PSI, chi-squared drift detection between datasets |

### Statistics Tools

| Category | Tests |
|----------|-------|
| **Descriptive** | Mean, median, std, skewness, kurtosis |
| **t-Tests** | One-sample, Independent, Paired |
| **ANOVA** | One-way, Two-way, Repeated Measures, Mixed |
| **Non-Parametric** | Mann-Whitney U, Wilcoxon Signed-Rank, Kruskal-Wallis, Friedman |
| **Correlation** | Pearson, Spearman |
| **Regression** | Linear (OLS), Logistic |
| **Other** | Chi-Squared, Binomial |

Each statistical test includes assumption checks, effect size calculations, and interactive Plotly charts.

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/ssevera1/DS_Stats_App.git
cd DS_Stats_App
pip install -r requirements.txt
```

### Launch

**Windows:**
```bash
launch.bat
```

**macOS / Linux:**
```bash
chmod +x launch.sh
./launch.sh
```

**Or directly:**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Customizing the Hero Image

To display a logo or banner on the Home page, place an image file at:

```
assets/hero.png
```

The image will appear centered above the title and automatically resize to fit. Supported formats: PNG, JPG, GIF, SVG.

To change the path, edit the `HERO_IMAGE` variable near the top of the `render()` function in `pages/home.py` (around line 48):

```python
HERO_IMAGE = "assets/hero.png"  # Change this to your image path
```

If no file exists at the specified path, the image section is simply hidden.

---

## How It Works

- **Data Science tools** — Upload a CSV or Excel file on the Home page (up to 200 MB). Your dataset persists across all DS tool pages via the sidebar navigation.
- **Statistics tools** — Navigate to "Data Input" under Statistics Tools. Enter data manually in the table editor, paste from a spreadsheet, or upload a separate file. Data persists across all statistical test pages.
- Both toolsets maintain separate data — uploading for one doesn't affect the other.
- All processing happens locally. Nothing is sent to any external server.

---

## Project Structure

```
DS_Stats_App/
├── app.py                  # Main entry point & navigation
├── .streamlit/config.toml  # Theme & upload settings (200 MB limit)
├── requirements.txt        # All dependencies
├── launch.bat / launch.sh  # Quick launchers
│
├── pages/                  # All page modules
│   ├── home.py             # Home page with upload & tool overview
│   ├── data_profiler.py    # DS: Automated EDA
│   ├── smart_cleaning.py   # DS: Data cleaning
│   ├── feature_engineering.py
│   ├── feature_selection.py
│   ├── class_imbalance.py
│   ├── model_arena.py
│   ├── hyperparameter_tuning.py
│   ├── explainability.py
│   ├── data_drift.py
│   ├── data_input.py       # Stats: Data entry/upload
│   ├── descriptive_stats.py
│   ├── one_sample_ttest.py
│   ├── independent_ttest.py
│   ├── paired_ttest.py
│   ├── oneway_anova.py
│   ├── twoway_anova.py
│   ├── repeated_anova.py
│   ├── mixed_anova.py
│   ├── mann_whitney.py
│   ├── wilcoxon.py
│   ├── kruskal_wallis.py
│   ├── friedman.py
│   ├── pearson_correlation.py
│   ├── spearman_correlation.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── chi_squared_test.py
│   └── binomial_test.py
│
├── utils/                  # DS styling & Plotly/Matplotlib themes
├── core/                   # Stats state management & data I/O
├── stats/                  # Statistical computation modules
├── charts/                 # Plotly chart builders
└── components/             # Reusable UI components
```

---

## Dependencies

Core: Streamlit, Pandas, NumPy, Plotly, SciPy, scikit-learn

DS-specific: imbalanced-learn, SHAP, Optuna, XGBoost, LightGBM, Matplotlib, Seaborn

Stats-specific: statsmodels, pingouin

See `requirements.txt` for full list with version constraints.

---

## License

MIT

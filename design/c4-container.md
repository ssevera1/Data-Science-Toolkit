# C4 Model - Level 2: Container Diagram

## Overview

The Container diagram zooms into the DS Power Tools system boundary, revealing the
major deployable units and their interactions. In this application, "containers" map
to Python package directories that form distinct architectural layers.

## Key Observations

- Single-process architecture: all containers run within one Streamlit Python process
- Session state acts as an in-memory message bus between containers
- The `pages/` container is the thickest layer (32 modules) but delegates all computation
- Clear separation: pages never compute statistics directly; stats never render UI

## Container Diagram

```mermaid
C4Container
    title Container Diagram - DS Power Tools

    Person(user, "User", "Data Scientist / Researcher")

    System_Boundary(app, "DS Power Tools") {
        Container(entry, "app.py", "Python / Streamlit", "Entry point: theme initialization, page registration, navigation routing")

        Container(pages, "Pages Layer", "Python / Streamlit (32 modules)", "View layer: renders UI for each tool, collects user input, orchestrates workflows")

        Container(core, "Core Layer", "Python (4 modules)", "State management, data I/O, input validation, global constants")

        Container(stats, "Statistics Engine", "Python / SciPy / statsmodels (13 modules)", "Pure computation: all statistical tests, assumption checks, effect sizes")

        Container(charts, "Visualization Engine", "Python / Plotly (7 modules)", "Chart generation: box plots, histograms, scatter, Q-Q, regression, bar plots")

        Container(components, "UI Components", "Python / Streamlit (4 modules)", "Reusable widgets: variable selectors, result cards, data tables")

        Container(utils, "Theming System", "Python / CSS (2 modules)", "Dual-theme management: Light/Dark mode with CSS injection")

        ContainerDb(state, "Session State", "Streamlit st.session_state", "In-memory shared state: DataFrame, variable types, tool configuration")
    }

    Rel(user, entry, "Opens in browser", "HTTP localhost:8501")
    Rel(entry, pages, "Routes to active page")
    Rel(pages, core, "Loads data, validates input")
    Rel(pages, stats, "Requests statistical computations")
    Rel(pages, charts, "Requests chart generation")
    Rel(pages, components, "Renders reusable widgets")
    Rel(pages, utils, "Applies theme styling")
    Rel(core, state, "Reads/writes shared state")
    Rel(pages, state, "Reads/writes page state")
    Rel(stats, state, "Reads DataFrame")
    Rel(charts, utils, "Gets theme colors & templates")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

## Dependency Graph

```mermaid
flowchart TD
    APP[app.py] --> PAGES[pages/]
    APP --> UTILS[utils/]
    APP --> CORE[core/]

    PAGES --> CORE
    PAGES --> STATS[stats/]
    PAGES --> CHARTS[charts/]
    PAGES --> COMPONENTS[components/]
    PAGES --> UTILS

    COMPONENTS --> UTILS
    CHARTS --> UTILS

    CORE --> STATE[(Session State)]
    PAGES --> STATE

    subgraph "External Libraries"
        SL[Streamlit]
        PD[Pandas]
        SP[SciPy]
        SM[statsmodels]
        SK[scikit-learn]
        PL[Plotly]
        SHAP_LIB[SHAP]
        OP[Optuna]
        XG[XGBoost]
        LG[LightGBM]
        IL[imbalanced-learn]
        PG[pingouin]
    end

    CORE --> SL & PD
    STATS --> SP & SM & PG & PD
    CHARTS --> PL
    PAGES --> SK & SHAP_LIB & OP & XG & LG & IL

    style APP fill:#1a237e,stroke:#0d47a1,color:#fff
    style PAGES fill:#283593,stroke:#1565c0,color:#fff
    style CORE fill:#4a148c,stroke:#6a1b9a,color:#fff
    style STATS fill:#004d40,stroke:#00695c,color:#fff
    style CHARTS fill:#bf360c,stroke:#d84315,color:#fff
    style COMPONENTS fill:#e65100,stroke:#ef6c00,color:#fff
    style UTILS fill:#827717,stroke:#9e9d24,color:#fff
    style STATE fill:#f9a825,stroke:#f57f17,color:#000
```

## Container Responsibilities

### app.py (Entry Point)
- Initializes Streamlit page config (layout, title, icon)
- Calls `core/state.py` to set up session defaults
- Applies global theme CSS via `utils/theme.py`
- Registers all 32 pages with `st.navigation()`
- Renders sidebar (theme selector, branding, navigation)

### Pages Layer (32 modules)
| Category | Count | Responsibility |
|---|---|---|
| Home | 1 | Data upload, tool card navigation |
| Data Science | 9 | EDA, cleaning, feature engineering, model training, explainability, drift detection |
| Statistics | 20 | Data input, descriptive stats, t-tests, ANOVA, non-parametric, correlation, regression, chi-squared, binomial, MANOVA |
| Navigation | 2 | Sidebar helpers, page routing |

### Core Layer (4 modules)
| Module | Responsibility |
|---|---|
| `state.py` | Initialize session defaults, provide getter/setter API for DataFrame and variable types |
| `data_manager.py` | Load CSV/Excel/clipboard, auto-detect variable types, export with formula injection prevention |
| `validators.py` | Validate column existence, numeric types, group structures, paired data, binary variables |
| `constants.py` | Alpha level (0.05), variable type enums, test category definitions, default dimensions |

### Statistics Engine (13 modules)
| Module | Tests |
|---|---|
| `descriptive.py` | Mean, median, mode, SD, skewness, kurtosis, quartiles, frequencies |
| `assumptions.py` | Shapiro-Wilk, Levene's, Mauchly's, Henze-Zirkler, Box's M |
| `ttest.py` | One-sample, independent (Student's & Welch's), paired |
| `anova.py` | One-way, two-way, repeated measures, mixed |
| `manova.py` | Wilks' lambda, Pillai's trace, Hotelling-Lawley, Roy's GR |
| `multivariate_regression.py` | Multi-DV regression with multivariate tests |
| `nonparametric.py` | Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman |
| `correlation.py` | Pearson, Spearman with Fisher's z CIs |
| `regression.py` | Linear (OLS), logistic with standardized betas, odds ratios |
| `chi_squared.py` | Chi-squared independence test, Cramer's V |
| `binomial.py` | Binomial test with CIs |
| `effect_size.py` | Cohen's d, eta-squared, omega-squared, Cramer's V, rank-biserial |

### Visualization Engine (7 modules)
| Module | Chart Types |
|---|---|
| `boxplot.py` | Grouped, paired, single box plots |
| `histogram.py` | Frequency histograms with normal curve overlay |
| `scatter.py` | Scatter plots with optional regression line |
| `barplot.py` | Grouped bar plots with means and error bars |
| `qq_plot.py` | Q-Q plots for normality diagnosis |
| `regression_plot.py` | Regression line with confidence bands |
| `theme.py` | Plotly template registration, color palettes per theme |

### UI Components (4 modules)
| Module | Components |
|---|---|
| `variable_selector.py` | Type-aware dropdown pickers (metric, nominal, ordinal, multi-select) |
| `results_display.py` | Significance cards, assumption badges, effect size gauges |
| `data_table.py` | Compact data preview with variable type annotations |
| `sidebar_nav.py` | Navigation link generators |

### Theming System (2 modules)
| Module | Responsibility |
|---|---|
| `utils/theme.py` | CSS generation for Light (cream/red) and Dark (blue-purple/cyan) themes, widget styling, metric cards |
| `charts/theme.py` | Plotly template objects per theme, color scales, font choices, grid styling |

## Inter-Container Communication

All communication flows through **Streamlit session state** - there are no direct
function callbacks or event emitters between containers. This creates a simple,
predictable data flow:

```mermaid
sequenceDiagram
    participant U as User
    participant P as Page
    participant C as Core
    participant S as Stats Engine
    participant V as Charts
    participant ST as Session State

    U->>P: Select variables & click "Run"
    P->>C: validate_columns(df, columns)
    C->>ST: get_dataframe()
    ST-->>C: DataFrame
    C-->>P: Validation result

    alt Valid input
        P->>S: run_test(df, columns, alpha)
        S-->>P: Results dict {statistic, p, effect_size, assumptions}
        P->>V: create_chart(df, columns, theme)
        V-->>P: Plotly Figure
        P->>U: Render results + chart
    else Invalid input
        P->>U: Show error message
    end
```

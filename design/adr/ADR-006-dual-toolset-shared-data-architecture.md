# ADR-006: Dual-Toolset Architecture with Shared Data Layer

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Architecture / Product |

## Context

The application serves two distinct workflows:

1. **Data Science Tools**: An automated ML pipeline (EDA, cleaning, feature
   engineering, model selection, hyperparameter tuning, explainability)
2. **Statistics Tools**: Classical statistical tests with assumption checking
   and effect size reporting

These workflows share common infrastructure (data loading, variable types,
visualization) but have fundamentally different user interactions and
computational patterns. A decision was needed on whether to build two separate
applications or combine them.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **Two separate apps** | Clear separation, independent deployment, no cross-contamination | Duplicated infrastructure, users must upload data twice, two codebases to maintain |
| **Single app, single workflow** | Simplest architecture | Forces users into one paradigm, limits utility |
| **Single app, dual toolsets** | Shared data, one upload, unified experience, shared infrastructure | Navigation complexity, larger codebase, potential confusion |
| **Plugin architecture** | Extensible, clean boundaries | Overengineered for two toolsets, complex plugin API |

## Decision

**Chosen: Single application with dual independent toolsets** sharing a common
data layer through session state.

## Rationale

1. **Upload once, analyze everywhere**: A user uploads a dataset on the home page
   and immediately has access to both exploratory ML workflows and formal
   statistical testing. This eliminates the friction of exporting from one tool
   and importing into another.

2. **Complementary workflows**: Data scientists often need both. After running
   a model in Model Arena, they may want to run a formal t-test comparing group
   means. After descriptive statistics, they may want to engineer features. The
   dual toolset supports the full analytical lifecycle.

3. **Shared infrastructure amortization**: Data loading, variable type detection,
   export, theming, and chart components are shared. This reduces code duplication
   by approximately 40% compared to two separate applications.

4. **Unified navigation**: A single sidebar with clear section headers
   ("Data Science Tools" and "Statistics Tools") provides discoverability without
   forcing users into a linear workflow.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   app.py                         │
│              (Navigation Router)                 │
├──────────────────────┬──────────────────────────┤
│  Data Science Tools  │    Statistics Tools       │
│  (9 pages)           │    (20 pages)             │
│                      │                           │
│  - Data Profiler     │    - Descriptive Stats    │
│  - Smart Cleaning    │    - t-tests (3)          │
│  - Feature Eng.      │    - ANOVA (4)            │
│  - Feature Sel.      │    - MANOVA               │
│  - Class Imbalance   │    - Non-parametric (4)   │
│  - Model Arena       │    - Correlation (2)      │
│  - Hyperparameter    │    - Regression (3)       │
│  - Explainability    │    - Chi-squared          │
│  - Data Drift        │    - Binomial             │
├──────────────────────┴──────────────────────────┤
│              Shared Infrastructure               │
│  core/ | stats/ | charts/ | components/ | utils/ │
├─────────────────────────────────────────────────┤
│          st.session_state["df"]                  │
│          (Shared DataFrame)                      │
└─────────────────────────────────────────────────┘
```

## Trade-offs Accepted

- **Navigation density**: 32 pages in a single sidebar is dense. Mitigated by
  grouping into collapsible sections and providing tool cards on the home page
  with descriptions.

- **Loading overhead**: Importing all 32 page modules at startup adds to initial
  load time. Mitigated by Streamlit's lazy page loading (only the active page
  renders its heavy imports).

- **Conceptual mixing**: Users who only need statistics tools see Data Science
  tools in the navigation (and vice versa). This is acceptable because the tools
  are clearly labeled and grouped, and the added discoverability often helps users
  find capabilities they didn't know existed.

- **Shared state side effects**: A Data Science tool (e.g., Smart Cleaning) that
  modifies the DataFrame affects subsequent statistical tests. This is intentional
  (users clean data before analyzing it) but means users must be aware of the
  pipeline order.

## Consequences

- Single installation and launch for the complete toolkit
- Data transformation in DS tools flows naturally into statistical analysis
- The codebase is larger but more cohesive than two separate projects
- New tools from either domain can be added by creating a new page module
- The shared data layer means variable type assignments persist across tools

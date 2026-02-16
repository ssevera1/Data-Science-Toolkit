# ADR-011: Strict Separation of Computation and Presentation

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Architecture / Code Organization |

## Context

The application contains both computational logic (statistical tests, data
transformations) and presentation logic (Streamlit widgets, result formatting,
chart rendering). As the codebase grew to 32 pages and 13 statistical modules,
a clear boundary was needed to prevent computational code from being tangled
with UI code.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **Everything in pages** | Simple, no abstraction | Untestable statistics, duplicated logic across pages, monolithic modules |
| **MVC pattern** | Well-understood, testable | Over-formal for Streamlit's rerun model, controllers add overhead |
| **Compute/Present split** | Testable statistics, reusable computations, clear contracts | Requires discipline to maintain boundary, dict-based interface |
| **Service layer pattern** | Enterprise-grade separation | Too much ceremony for the project's scale |

## Decision

**Chosen: Strict compute/present split** where `stats/` modules are pure
computation (no Streamlit imports) and `pages/` modules handle all UI rendering.

## Rationale

1. **Testability**: The `stats/` modules can be unit-tested without Streamlit.
   Pass a DataFrame and parameters in, get a results dict out. No UI mocking,
   no session state setup, no widget simulation.

2. **Reusability**: Statistical functions can be called from any page. The
   `assumptions.shapiro_wilk()` function is used by t-tests, ANOVAs, correlations,
   and regressions without each page reimplementing normality checking.

3. **Clear contracts**: Every statistical function returns a dict with well-known
   keys (`test_name`, `statistic`, `p_value`, `effect_size`, `assumptions`). The
   `components/results_display.py` module knows how to render any dict that
   follows this contract, regardless of which test produced it.

4. **Independent evolution**: The statistics engine can add new tests or fix
   calculations without touching any UI code. The presentation layer can redesign
   result cards or add new chart types without modifying statistical logic.

## Boundary Rules

| Layer | May Import | Must NOT Import |
|---|---|---|
| `stats/` | pandas, numpy, scipy, statsmodels, pingouin | streamlit, plotly, components |
| `charts/` | plotly, pandas, numpy | streamlit (except for theme state) |
| `pages/` | stats, charts, components, core, utils | Raw scipy/statsmodels directly |
| `components/` | streamlit, utils | stats, charts |
| `core/` | streamlit, pandas | stats, charts, pages |

## Interface Contract

```python
# Every stats function follows this pattern:
def some_test(df: pd.DataFrame, **params) -> dict:
    """
    Returns:
        {
            "test_name": str,           # Human-readable test name
            "statistic": float,         # Test statistic value
            "p_value": float,           # p-value
            "effect_size": float,       # Effect size value
            "effect_size_label": str,   # Effect size name (e.g., "Cohen's d")
            "effect_size_interp": str,  # Interpretation (e.g., "medium")
            "interpretation": str,      # Plain-language result
            "assumptions": dict,        # Assumption check results
            "confidence_interval": tuple,  # (lower, upper) if applicable
            "posthoc": DataFrame | None,   # Post-hoc results if applicable
            "additional": dict,         # Test-specific extra data
        }
    """
```

## Trade-offs Accepted

- **Dict-based interface**: Using plain dicts instead of typed dataclasses or
  Pydantic models means there's no compile-time type checking on the interface.
  A misspelled key in the stats module won't be caught until runtime. This is
  acceptable because the page modules render results immediately, making errors
  visible during development.

- **Thin adapter overhead**: Each page module contains ~20-30 lines of "glue code"
  that calls the stats function, unpacks the result dict, and passes values to
  display components. This is mechanical but necessary.

- **No direct Streamlit caching in stats**: Since stats modules don't import
  Streamlit, they cannot use `@st.cache_data`. Caching must be applied at the
  page level, which means the page author must remember to cache. In practice,
  most tests are fast enough that caching is unnecessary.

## Consequences

- Statistical logic is isolated, testable, and reusable
- New statistical tests follow a clear template: create a function in `stats/`, create a page in `pages/`
- The `components/results_display.py` module can render any test's results through the shared dict contract
- Changes to the UI don't risk breaking statistical calculations
- Changes to calculations don't risk breaking the UI

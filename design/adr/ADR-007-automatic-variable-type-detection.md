# ADR-007: Automatic Variable Type Detection with Manual Override

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | UX / Data Handling |

## Context

Statistical tests require knowing whether variables are metric (continuous),
nominal (categorical), or ordinal (ordered categorical). This classification
determines which tests are appropriate, which chart types to use, and how to
validate input.

Users should not need to manually classify every column in their dataset before
running a single test. At the same time, auto-detection is imperfect - a column
of zip codes (numeric but nominal) can be misclassified as metric.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **Manual-only classification** | Always correct (user decides) | Tedious for large datasets, delays time-to-first-analysis |
| **Auto-detect, no override** | Zero friction | Misclassifications with no recourse |
| **Auto-detect with manual override** | Fast start, correctability | Auto-detection logic needs tuning, UI for overrides |
| **Schema file import** | Precise, reusable | Requires users to create schema files, not beginner-friendly |

## Decision

**Chosen: Auto-detect variable types on upload, with manual override via the
Data Input page.**

## Rationale

1. **Time to first analysis**: Users can upload a CSV and immediately run a t-test
   without configuring anything. Auto-detection handles the common case (numeric
   columns are metric, low-cardinality text columns are nominal) correctly ~90%
   of the time.

2. **Correctability**: When auto-detection is wrong (zip codes detected as metric,
   Likert scales detected as metric instead of ordinal), users can reclassify
   columns on the Data Input page. The override persists for the entire session.

3. **Type-aware UI**: The variable selector components use type information to
   filter dropdowns. `select_metric_variable()` only shows metric columns,
   `select_nominal_variable()` only shows nominal columns. This prevents invalid
   test configurations (e.g., running a t-test with a nominal dependent variable).

## Detection Algorithm

```python
def _auto_detect_type(series):
    numeric_count = pd.to_numeric(series, errors='coerce').notna().sum()
    numeric_ratio = numeric_count / len(series)

    if numeric_ratio > 0.5:
        n_unique = series.nunique()
        if n_unique <= 2:
            return "Nominal"      # Binary (e.g., 0/1, Yes/No)
        elif n_unique <= 7 and n_unique / len(series) < 0.05:
            return "Ordinal"      # Few categories, sparse (e.g., Likert 1-5)
        else:
            return "Metric"       # Continuous numeric
    else:
        n_unique = series.nunique()
        if n_unique <= 10:
            return "Nominal"      # Categorical text
        else:
            return "Text"         # Free text, excluded from analysis
```

### Threshold Justification

| Threshold | Value | Rationale |
|---|---|---|
| Numeric ratio | > 50% | Tolerates some missing/non-numeric values |
| Binary cutoff | <= 2 unique | Binary variables are almost always categorical |
| Ordinal range | 3-7 unique, < 5% density | Matches typical Likert scales and rating systems |
| Nominal text cutoff | <= 10 unique | Common number of experimental groups/conditions |

## Trade-offs Accepted

- **Imperfect heuristics**: The algorithm uses fixed thresholds that won't be
  optimal for all datasets. A column with 8 unique numeric values could be ordinal
  or metric depending on context. Users must verify and correct.

- **No type persistence**: Variable types are not saved to a schema file. If the
  user re-uploads the same dataset, types are re-detected. This simplifies the
  system but creates repetitive work for users with recurring datasets.

- **Text exclusion**: Columns classified as "Text" (high-cardinality strings) are
  excluded from statistical analysis entirely. This prevents errors but means
  users must manually reclassify text columns that represent categories.

## Consequences

- Users experience zero-friction onboarding: upload and analyze immediately
- Variable selectors are type-aware, preventing invalid test configurations
- The Data Input page serves as the "fix-up" point for misclassified variables
- Auto-detection covers the common case well enough to be useful, not perfect enough to be trusted blindly

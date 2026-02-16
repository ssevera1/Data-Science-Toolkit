# ADR-014: One Page Per Statistical Test Navigation Model

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | UX / Navigation |

## Context

The application implements 20+ statistical tests. A navigation model was needed
to organize these tests for user access. The key question was whether to group
tests into fewer multi-purpose pages or give each test its own page.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **Single page with test selector** | Compact, fewer modules | Complex page logic, context switching within page, long scrolling |
| **Category pages (e.g., "ANOVA" page with all variants)** | Logical grouping, moderate page count | Still complex per page, variants may confuse users |
| **One page per test** | Focused UX, simple page logic, deep-linkable, each page is self-contained | Many sidebar entries (20+), potential navigation overwhelm |
| **Wizard/flow-based** | Guided experience | Rigid, doesn't support ad-hoc analysis, complex state management |

## Decision

**Chosen: One dedicated page per statistical test**, organized into collapsible
sidebar sections.

## Rationale

1. **Focused context**: Each page has one purpose. When a user navigates to
   "Independent t-test," every element on the page is relevant to that test:
   the variable selectors show appropriate options, the guide explains that
   specific test, and the results are formatted for that test's output.

2. **Simple page modules**: Each page module is 100-200 lines of straightforward
   Python - guard check, header, variable selection, validation, computation,
   display. There are no conditionals for "which test did the user select."

3. **Deep linking**: Each test has a unique URL (e.g., `/independent_ttest`).
   Users can bookmark specific tests or share links with collaborators.

4. **Embedded guides**: Each page includes an expandable "Page Guide & Explanation"
   tailored to that specific test. This provides context-sensitive help without
   a separate documentation system.

5. **Scalability**: Adding a new test means creating a single new page module
   and registering it in `app.py`. No existing pages need modification.

## Navigation Structure

```
Sidebar
├── Home
├── Data Science Tools
│   ├── Data Profiler
│   ├── Smart Cleaning
│   ├── Feature Engineering
│   ├── Feature Selection
│   ├── Class Imbalance
│   ├── Model Arena
│   ├── Hyperparameter Tuning
│   ├── Explainability
│   └── Data Drift
└── Statistics Tools
    ├── Data Input
    ├── Descriptive Statistics
    ├── One-Sample t-test
    ├── Independent t-test
    ├── Paired t-test
    ├── One-Way ANOVA
    ├── Two-Way ANOVA
    ├── Repeated Measures ANOVA
    ├── Mixed ANOVA
    ├── MANOVA
    ├── Mann-Whitney U
    ├── Wilcoxon Signed-Rank
    ├── Kruskal-Wallis
    ├── Friedman
    ├── Pearson Correlation
    ├── Spearman Correlation
    ├── Linear Regression
    ├── Logistic Regression
    ├── Multivariate Regression
    ├── Chi-Squared Test
    └── Binomial Test
```

## Trade-offs Accepted

- **Long sidebar**: 32 entries in the sidebar is visually dense. Mitigated by
  collapsible sections that group tools by category. Users who know which test
  they need can navigate directly; users who don't can browse by category.

- **Code repetition across pages**: The page template (guard, header, variable
  selection, tabs) is repeated in every page module. This is intentional - each
  page is self-contained and can be understood independently. The repetition
  (~50 lines per page) is manageable and avoids the complexity of a dynamic
  page generator.

- **No test recommendation engine**: Users must know which test to use. The
  application doesn't guide users from a research question to the appropriate
  test. Partially mitigated by the Page Guide on each page explaining "when to
  use this test."

## Consequences

- Each statistical test is a self-contained, independently understandable module
- Navigation is flat and predictable - users always know where they are
- Adding new tests is a single-module operation with no side effects
- The sidebar serves as a "menu" of available analyses
- Page guides provide contextual education without a separate help system

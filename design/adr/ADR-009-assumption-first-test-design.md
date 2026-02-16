# ADR-009: Assumption-First Statistical Test Design

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Product / Statistical Methodology |

## Context

Statistical tests have mathematical assumptions (normality, homogeneity of
variances, independence, etc.) that must be satisfied for results to be valid.
Many statistical tools either ignore assumptions entirely (leading to invalid
conclusions) or require users to manually run assumption tests before the main
analysis (creating friction and expertise barriers).

A design decision was needed on how prominently to surface assumption checking
in the user workflow.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **No assumption checking** | Simplest UX, fastest results | Users may draw invalid conclusions, academically irresponsible |
| **Separate assumption page** | Clean separation | Users skip it, breaks flow, not contextual |
| **Warning-only (post-hoc)** | Non-intrusive | Easy to ignore, results already computed |
| **Integrated assumption tab** | Contextual, always visible, educational | More complex page layout, adds to page length |
| **Blocking (fail if violated)** | Prevents invalid analyses | Too restrictive, experts may deliberately violate |

## Decision

**Chosen: Integrated assumption checking as a dedicated tab alongside results
and charts**, with clear pass/fail indicators and interpretive guidance. Assumptions
are checked automatically when the test runs, but violations do not block the
analysis.

## Rationale

1. **Educational design**: Every statistics page has three tabs: Results,
   Assumptions, and Charts. The Assumptions tab is always one click away from
   the results, making it impossible to miss but not intrusive.

2. **Contextual checking**: Assumptions are checked using the same data and
   groups as the main test, not in isolation. For example, when running an
   independent t-test, Shapiro-Wilk is run per group and Levene's test is run
   on the grouping variable. The results directly apply to the current analysis.

3. **Non-blocking with guidance**: Violated assumptions produce colored warning
   badges with interpretation text (e.g., "Normality violated for Group A.
   Consider using Mann-Whitney U as a non-parametric alternative."). This
   respects expert judgment while guiding novices.

4. **Automatic Welch's correction**: The independent t-test automatically applies
   Welch's correction when Levene's test indicates unequal variances. This
   demonstrates how assumption checking can be actionable, not just informational.

## Implementation Pattern

```python
# stats/ttest.py (simplified)
def independent_ttest(df, dv, group, alpha=0.05):
    groups = [g[dv].dropna() for _, g in df.groupby(group)]

    # Assumption checks (run automatically)
    normality = [assumptions.shapiro_wilk(g, alpha) for g in groups]
    homogeneity = assumptions.levene_test(groups, alpha)

    # Adapt test based on assumptions
    if homogeneity["passed"]:
        stat, p = scipy.stats.ttest_ind(groups[0], groups[1])
        test_variant = "Student's t-test"
    else:
        stat, p = scipy.stats.ttest_ind(groups[0], groups[1], equal_var=False)
        test_variant = "Welch's t-test"

    return {
        "test_name": test_variant,
        "statistic": stat,
        "p_value": p,
        "effect_size": effect_size.cohens_d_independent(groups[0], groups[1]),
        "assumptions": {
            "normality": normality,
            "homogeneity": homogeneity,
        }
    }
```

### Assumptions by Test

| Test | Assumptions Checked |
|---|---|
| One-sample t-test | Shapiro-Wilk normality |
| Independent t-test | Per-group normality, Levene's homogeneity |
| Paired t-test | Shapiro-Wilk on differences |
| One-way ANOVA | Per-group normality, Levene's homogeneity |
| Two-way ANOVA | Per-cell normality, Levene's homogeneity |
| Repeated measures ANOVA | Per-level normality, Mauchly's sphericity |
| Mixed ANOVA | Per-cell normality, Mauchly's sphericity |
| MANOVA | Multivariate normality (Henze-Zirkler), Box's M |
| Pearson correlation | Normality of both variables |
| Linear regression | Normality of residuals, homoscedasticity |
| Chi-squared | Expected frequencies >= 5 |
| Non-parametric tests | None (assumption-free by design) |

## Trade-offs Accepted

- **Computation overhead**: Running assumption tests (Shapiro-Wilk per group,
  Levene's, etc.) adds computation time. For most datasets, this is negligible
  (< 100ms), but for very large datasets with many groups, it can add up.

- **Information overload for novices**: The Assumptions tab shows statistical
  test results (W statistic, p-value) that novices may not understand. Mitigated
  by clear pass/fail badges and plain-language interpretation text.

- **Non-blocking risks**: Users can ignore violated assumptions and report
  invalid results. This is a deliberate design choice - the tool educates and
  warns but does not enforce, respecting user autonomy and expert judgment.

## Consequences

- Every statistical test provides contextual assumption checking at no extra user effort
- The three-tab layout (Results | Assumptions | Charts) is consistent across all 20 statistics pages
- Assumption violations produce actionable guidance (e.g., "use Mann-Whitney instead")
- Some tests adapt automatically based on assumption results (Welch's correction)
- Effect sizes are always reported alongside p-values, supporting modern statistical practice

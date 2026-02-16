# ADR-012: Mandatory Effect Sizes Alongside p-Values

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Product / Statistical Methodology |

## Context

Traditional statistical tools often report only p-values, which indicate whether
an effect exists but not how large it is. The American Psychological Association
(APA), the American Statistical Association (ASA), and major journals now require
or strongly recommend reporting effect sizes alongside significance tests.

A design decision was needed on whether effect sizes should be optional, default,
or mandatory in the application's output.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **p-value only** | Simplest output, familiar | Incomplete, poor statistical practice, misleading with large samples |
| **Effect size optional** | User choice, less output clutter | Most users won't enable it, perpetuates p-value-only culture |
| **Effect size default** | Good practice by default | Some users may not understand effect sizes |
| **Effect size mandatory** | Educates users, APA-compliant, complete reporting | Cannot be hidden, adds complexity to output |

## Decision

**Chosen: Effect sizes are always computed and displayed alongside p-values.**
Every statistical test reports an appropriate effect size with magnitude
interpretation (small, medium, large) based on published benchmarks.

## Rationale

1. **Statistical best practice**: The ASA's 2016 statement on p-values explicitly
   warns against relying on p-values alone. Effect sizes provide practical
   significance (how much does it matter?) in addition to statistical significance
   (is it likely real?).

2. **Educational value**: By always showing effect sizes with interpretations,
   users learn to think about practical significance. A t-test with p = 0.001
   but d = 0.05 is statistically significant but practically meaningless. The
   application makes this visible.

3. **Large-sample correction**: With large datasets (n > 10,000), almost any
   difference becomes statistically significant. Effect sizes remain stable
   regardless of sample size, providing a meaningful measure of the finding's
   importance.

4. **Publication readiness**: Results include the information needed for APA-style
   reporting. Users don't need to compute effect sizes separately or in a
   different tool.

## Effect Size Mapping

| Test | Effect Size | Benchmarks (Small / Medium / Large) |
|---|---|---|
| One-sample t-test | Cohen's d | 0.2 / 0.5 / 0.8 |
| Independent t-test | Cohen's d | 0.2 / 0.5 / 0.8 |
| Paired t-test | Cohen's d (paired) | 0.2 / 0.5 / 0.8 |
| One-way ANOVA | Eta-squared (+ Omega-squared) | 0.01 / 0.06 / 0.14 |
| Two-way ANOVA | Eta-squared per effect | 0.01 / 0.06 / 0.14 |
| Repeated measures ANOVA | Eta-squared | 0.01 / 0.06 / 0.14 |
| Mann-Whitney U | Rank-biserial correlation | 0.1 / 0.3 / 0.5 |
| Wilcoxon signed-rank | Rank-biserial correlation | 0.1 / 0.3 / 0.5 |
| Kruskal-Wallis | Epsilon-squared | 0.01 / 0.06 / 0.14 |
| Chi-squared | Cramer's V | varies by df |
| Pearson correlation | r (is itself an effect size) | 0.1 / 0.3 / 0.5 |
| Spearman correlation | rho (is itself an effect size) | 0.1 / 0.3 / 0.5 |
| Linear regression | R-squared | 0.02 / 0.13 / 0.26 |
| Logistic regression | Odds ratio | context-dependent |

## Trade-offs Accepted

- **Output density**: Every test result now includes an additional section for
  effect size reporting. This makes the results view more information-dense.
  Mitigated by clear formatting with colored badges and plain-language
  interpretation.

- **Benchmark debates**: Cohen's benchmarks (small = 0.2, medium = 0.5,
  large = 0.8 for d) are widely used but debated. Some fields have different
  norms. The application uses Cohen's benchmarks as defaults, which is the most
  common convention.

- **Cannot be disabled**: Some users may prefer cleaner output without effect
  sizes. The application does not provide a toggle to hide them, prioritizing
  statistical completeness over UI minimalism.

## Consequences

- Every statistical test reports both statistical and practical significance
- Users receive publication-ready statistics with no additional effort
- The `stats/effect_size.py` module centralizes all effect size calculations
- Interpretation text helps users understand what the numbers mean
- The application promotes modern statistical practice by design

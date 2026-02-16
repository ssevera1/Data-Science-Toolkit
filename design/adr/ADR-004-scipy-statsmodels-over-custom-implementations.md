# ADR-004: Chosen SciPy + statsmodels Over Custom Statistical Implementations

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Technology / Computation |

## Context

The application implements 20+ statistical tests, each with assumption checks and
effect size calculations. A decision was needed on whether to implement statistical
algorithms from scratch, use a single library, or compose from multiple established
libraries.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **Custom implementations** | Full control, no dependencies, educational | Error-prone, untested edge cases, maintenance burden, reinventing the wheel |
| **R via rpy2** | Gold standard for statistics, vast test coverage | Python-R bridge complexity, R installation required, debugging across languages |
| **SciPy alone** | Well-tested, fast, part of scientific Python stack | Missing some tests (repeated measures, MANOVA, logistic regression) |
| **statsmodels alone** | Comprehensive statistical modeling | API inconsistencies, heavier than needed for simple tests |
| **SciPy + statsmodels + pingouin** | Best of each for their strengths | Multiple API styles, more dependencies |

## Decision

**Chosen: SciPy as the primary engine, supplemented by statsmodels for regression
and MANOVA, and pingouin for specialized tests** (sphericity, Box's M, multivariate
normality).

## Rationale

1. **SciPy for core tests**: `scipy.stats` provides battle-tested implementations
   of t-tests, Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, chi-squared,
   binomial, Shapiro-Wilk, Levene's, and Pearson/Spearman correlations. These are
   the most commonly used tests and benefit from SciPy's performance and reliability.

2. **statsmodels for modeling**: Linear regression (OLS), logistic regression (Logit),
   and MANOVA require full model objects with coefficient tables, standard errors,
   confidence intervals, and diagnostic statistics. `statsmodels` provides these with
   a familiar R-like formula interface.

3. **pingouin for niche tests**: Mauchly's sphericity test, Box's M test for
   homogeneity of covariance matrices, and Henze-Zirkler multivariate normality
   test are not available in SciPy or statsmodels. `pingouin` fills these gaps
   with a clean, modern API.

4. **Correctness over originality**: Statistical computation is a domain where
   correctness is paramount. Published, peer-reviewed libraries with thousands of
   tests and users are inherently more reliable than custom implementations. The
   risk of a subtle statistical error (wrong degrees of freedom, incorrect p-value
   calculation) in custom code is unacceptable.

## Library Mapping

| Test/Capability | Library | Module |
|---|---|---|
| t-tests (all variants) | SciPy | `scipy.stats.ttest_1samp`, `ttest_ind`, `ttest_rel` |
| One-way ANOVA | SciPy | `scipy.stats.f_oneway` |
| Two-way ANOVA | statsmodels | `statsmodels.stats.anova.anova_lm` |
| Repeated measures ANOVA | pingouin | `pingouin.rm_anova` |
| Mixed ANOVA | pingouin | `pingouin.mixed_anova` |
| MANOVA | statsmodels | `statsmodels.multivariate.manova.MANOVA` |
| Mann-Whitney U | SciPy | `scipy.stats.mannwhitneyu` |
| Wilcoxon signed-rank | SciPy | `scipy.stats.wilcoxon` |
| Kruskal-Wallis | SciPy | `scipy.stats.kruskal` |
| Friedman | SciPy | `scipy.stats.friedmanchisquare` |
| Pearson correlation | SciPy | `scipy.stats.pearsonr` |
| Spearman correlation | SciPy | `scipy.stats.spearmanr` |
| Linear regression | statsmodels | `statsmodels.api.OLS` |
| Logistic regression | statsmodels | `statsmodels.api.Logit` |
| Chi-squared | SciPy | `scipy.stats.chi2_contingency` |
| Binomial test | SciPy | `scipy.stats.binomtest` |
| Shapiro-Wilk | SciPy | `scipy.stats.shapiro` |
| Levene's test | SciPy | `scipy.stats.levene` |
| Mauchly's sphericity | pingouin | `pingouin.sphericity` |
| Box's M | pingouin | `pingouin.box_m` |
| Henze-Zirkler | pingouin | `pingouin.multivariate_normality` |
| Tukey HSD | statsmodels | `statsmodels.stats.multicomp.pairwise_tukeyhsd` |

## Trade-offs Accepted

- **Multiple API styles**: SciPy returns named tuples, statsmodels returns model
  objects, pingouin returns DataFrames. The `stats/` modules wrap these into a
  consistent `dict` result format, adding a thin adapter layer.

- **Dependency weight**: Three statistical libraries plus their transitive
  dependencies increase install size. This is acceptable for a data science tool
  where users already have most of these installed.

- **Version coupling**: The application depends on specific API behaviors across
  three libraries. Breaking changes in any library could require updates. This is
  mitigated by pinning minimum versions in `requirements.txt`.

## Consequences

- Statistical results are validated against established, peer-reviewed implementations
- The `stats/` layer acts as an adapter that normalizes output formats
- New tests can be added by importing from the appropriate library and wrapping
- Effect sizes are computed manually (using formulas from the literature) since
  no single library provides all needed effect size calculations

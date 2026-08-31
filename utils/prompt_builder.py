"""Pure prompt construction for Gemini AI integration.

This module builds prompt strings from result dicts, DataFrame metadata,
and analysis context.  It has NO Streamlit or API imports — it is a pure
utility following the same discipline as the stats/ modules.
"""

# ---------------------------------------------------------------------------
# System-level preamble injected into every prompt
# ---------------------------------------------------------------------------

_SYSTEM_CONTEXT = (
    "You are a statistics and data-science advisor embedded in DS Power Tools, "
    "an interactive data-science and statistics toolkit. "
    "Provide clear, actionable interpretations in plain language that a "
    "researcher or analyst can immediately use. "
    "Always reference the actual numbers from the results. "
    "Do not use markdown formatting or bullet points in the brief response. "
    "Keep interpretations concise and practical."
)

_DEEP_DIVE_CONTEXT = (
    "You are a statistics and data-science advisor embedded in DS Power Tools. "
    "Provide a thorough, detailed interpretation covering: "
    "(1) statistical significance and what it means in context, "
    "(2) effect size interpretation with practical significance, "
    "(3) assumption check results and whether violations affect the analysis, "
    "(4) confidence interval interpretation if available, "
    "(5) recommendations for alternative tests if assumptions are violated, "
    "(6) follow-up analyses to consider, and "
    "(7) APA-style reporting suggestions. "
    "Reference actual numbers from the results throughout. "
    "Use short paragraphs separated by blank lines."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, decimals=4):
    """Format a numeric value safely."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _assumption_summary(assumptions: dict | None) -> str:
    """Produce a one-line-per-check summary of assumption results."""
    if not assumptions:
        return "No assumption checks available."
    lines = []
    for name, info in assumptions.items():
        if isinstance(info, dict):
            passed = "PASSED" if info.get("passed") else "VIOLATED"
            detail = info.get("detail", "")
            p = _fmt(info.get("p_value"), 4)
            lines.append(f"  - {name}: {passed} (p={p}) {detail}")
    return "\n".join(lines) if lines else "No assumption checks available."


def _var_description(variables: dict | None) -> str:
    """Format variable names into a readable string."""
    if not variables:
        return "not specified"
    parts = []
    for key, val in variables.items():
        label = key.replace("_", " ").title()
        parts.append(f"{label}: {val}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Entry-type specific result extractors
# ---------------------------------------------------------------------------

def _extract_ttest(result: dict) -> str:
    """Extract key values from a t-test result dict."""
    return (
        f"Test: {result.get('test', 'T-Test')}\n"
        f"t-statistic: {_fmt(result.get('t'))}\n"
        f"p-value: {_fmt(result.get('p'))}\n"
        f"Degrees of freedom: {_fmt(result.get('df'), 2)}\n"
        f"Mean difference: {_fmt(result.get('mean_diff'))}\n"
        f"Cohen's d: {_fmt(result.get('cohens_d'))}\n"
        f"95% CI: [{_fmt(result.get('ci_lower'))}, {_fmt(result.get('ci_upper'))}]\n"
        f"Group 1 mean: {_fmt(result.get('mean1'))}, Group 2 mean: {_fmt(result.get('mean2'))}\n"
        f"N1: {result.get('n1', 'N/A')}, N2: {result.get('n2', 'N/A')}"
    )


def _extract_one_sample_ttest(result: dict) -> str:
    return (
        f"Test: {result.get('test', 'One-Sample t-Test')}\n"
        f"t-statistic: {_fmt(result.get('t'))}\n"
        f"p-value: {_fmt(result.get('p'))}\n"
        f"Degrees of freedom: {_fmt(result.get('df'), 2)}\n"
        f"Sample mean: {_fmt(result.get('mean'))}\n"
        f"Hypothesized mean: {_fmt(result.get('mu'))}\n"
        f"Mean difference: {_fmt(result.get('mean_diff'))}\n"
        f"Cohen's d: {_fmt(result.get('cohens_d'))}\n"
        f"95% CI: [{_fmt(result.get('ci_lower'))}, {_fmt(result.get('ci_upper'))}]\n"
        f"N: {result.get('n', 'N/A')}"
    )


def _extract_anova(result: dict) -> str:
    return (
        f"Test: {result.get('test', 'ANOVA')}\n"
        f"F-statistic: {_fmt(result.get('F'))}\n"
        f"p-value: {_fmt(result.get('p'))}\n"
        f"Degrees of freedom: {result.get('df_between', 'N/A')}, {result.get('df_within', 'N/A')}\n"
        f"Effect size (eta-squared): {_fmt(result.get('eta_squared'))}\n"
        f"Omega-squared: {_fmt(result.get('omega_squared'))}\n"
        f"Number of groups: {result.get('n_groups', 'N/A')}\n"
        f"Total N: {result.get('n_total', 'N/A')}"
    )


def _extract_anova_general(result: dict) -> str:
    """For two-way, repeated measures, and mixed ANOVA."""
    lines = [f"Test: {result.get('test', 'ANOVA')}"]
    anova_table = result.get("anova_table")
    if isinstance(anova_table, list):
        for row in anova_table[:10]:
            if isinstance(row, dict):
                src = row.get("Source", row.get("source", "?"))
                f_val = _fmt(row.get("F", row.get("F-stat")))
                p_val = _fmt(row.get("p-unc", row.get("p")))
                lines.append(f"  {src}: F={f_val}, p={p_val}")
    elif isinstance(result.get("F"), (int, float)):
        lines.append(f"F-statistic: {_fmt(result.get('F'))}")
        lines.append(f"p-value: {_fmt(result.get('p'))}")
    lines.append(f"Effect size: {_fmt(result.get('eta_squared', result.get('effect_size')))}")
    return "\n".join(lines)


def _extract_nonparametric(result: dict) -> str:
    stat_name = "U" if "U" in result else "W" if "W" in result else "H" if "H" in result else "statistic"
    stat_val = result.get("U", result.get("W", result.get("H", result.get("statistic"))))
    return (
        f"Test: {result.get('test', 'Non-parametric test')}\n"
        f"{stat_name}-statistic: {_fmt(stat_val)}\n"
        f"p-value: {_fmt(result.get('p'))}\n"
        f"Effect size: {_fmt(result.get('effect_size', result.get('rank_biserial')))}\n"
        f"Effect size label: {result.get('effect_size_label', 'N/A')}\n"
        f"N: {result.get('n', result.get('n1', 'N/A'))}"
    )


def _extract_correlation(result: dict) -> str:
    return (
        f"Test: {result.get('test', 'Correlation')}\n"
        f"Correlation coefficient (r): {_fmt(result.get('r', result.get('rho')))}\n"
        f"p-value: {_fmt(result.get('p'))}\n"
        f"95% CI: [{_fmt(result.get('ci_lower'))}, {_fmt(result.get('ci_upper'))}]\n"
        f"N: {result.get('n', 'N/A')}\n"
        f"R-squared: {_fmt(result.get('r_squared'))}"
    )


def _extract_regression(result: dict) -> str:
    lines = [
        f"Test: {result.get('test', 'Regression')}",
        f"R-squared: {_fmt(result.get('r_squared', result.get('pseudo_r_squared')))}",
        f"Adjusted R-squared: {_fmt(result.get('adj_r_squared'))}",
        f"F-statistic: {_fmt(result.get('f_statistic'))}",
        f"p-value (model): {_fmt(result.get('f_pvalue', result.get('p')))}",
        f"N: {result.get('n', 'N/A')}",
    ]
    coefs = result.get("coefficients")
    if isinstance(coefs, list):
        lines.append("Coefficients:")
        for c in coefs[:10]:
            if isinstance(c, dict):
                name = c.get("variable", c.get("name", "?"))
                coef = _fmt(c.get("coefficient", c.get("coef")))
                p = _fmt(c.get("p_value", c.get("p")))
                lines.append(f"  {name}: coef={coef}, p={p}")
    return "\n".join(lines)


def _extract_chi_squared(result: dict) -> str:
    return (
        f"Test: {result.get('test', 'Chi-Squared Test')}\n"
        f"Chi-squared statistic: {_fmt(result.get('chi2'))}\n"
        f"p-value: {_fmt(result.get('p'))}\n"
        f"Degrees of freedom: {result.get('df', 'N/A')}\n"
        f"Cramer's V: {_fmt(result.get('cramers_v'))}\n"
        f"N: {result.get('n', 'N/A')}"
    )


def _extract_binomial(result: dict) -> str:
    return (
        f"Test: {result.get('test', 'Binomial Test')}\n"
        f"p-value: {_fmt(result.get('p'))}\n"
        f"Observed proportion: {_fmt(result.get('observed_proportion', result.get('proportion')))}\n"
        f"Hypothesized proportion: {_fmt(result.get('hypothesized_proportion', result.get('p0')))}\n"
        f"Successes: {result.get('successes', result.get('k', 'N/A'))}\n"
        f"N: {result.get('n', 'N/A')}\n"
        f"95% CI: [{_fmt(result.get('ci_lower'))}, {_fmt(result.get('ci_upper'))}]"
    )


def _extract_manova(result: dict) -> str:
    lines = [
        f"Test: {result.get('test', 'MANOVA')}",
        f"Pillai's trace: {_fmt(result.get('pillai'))}",
        f"Wilks' lambda: {_fmt(result.get('wilks'))}",
        f"F-statistic: {_fmt(result.get('F'))}",
        f"p-value: {_fmt(result.get('p'))}",
    ]
    return "\n".join(lines)


def _extract_descriptive(result: dict) -> str:
    lines = [f"Test: Descriptive Statistics"]
    summary = result.get("summary")
    if isinstance(summary, dict):
        for key, val in list(summary.items())[:20]:
            lines.append(f"  {key}: {_fmt(val) if isinstance(val, (int, float)) else val}")
    elif isinstance(result.get("n"), (int, float)):
        lines.append(f"N: {result.get('n')}")
        lines.append(f"Mean: {_fmt(result.get('mean'))}")
        lines.append(f"Std Dev: {_fmt(result.get('std'))}")
        lines.append(f"Min: {_fmt(result.get('min'))}")
        lines.append(f"Max: {_fmt(result.get('max'))}")
    return "\n".join(lines)


def _extract_model_arena(result: dict) -> str:
    lines = [
        f"Analysis: Model Arena (ML Benchmarking)",
        f"Task: {result.get('task', 'N/A')}",
        f"Best model: {result.get('best_model', 'N/A')}",
        f"Primary metric: {result.get('primary_metric', 'N/A')}",
        f"Best score: {_fmt(result.get('best_score'))}",
        f"Number of models compared: {result.get('n_models', 'N/A')}",
        f"Cross-validation folds: {result.get('cv_folds', 'N/A')}",
    ]
    rankings = result.get("rankings")
    if isinstance(rankings, list):
        lines.append("Top models:")
        for r in rankings[:5]:
            if isinstance(r, dict):
                lines.append(f"  {r.get('model', '?')}: {_fmt(r.get('score'))}")
    return "\n".join(lines)


def _extract_feature_selection(result: dict) -> str:
    lines = [
        f"Analysis: Feature Selection",
        f"Method: {result.get('method', 'Multiple methods')}",
        f"Number of features evaluated: {result.get('n_features', 'N/A')}",
    ]
    selected = result.get("selected_features")
    if isinstance(selected, list):
        lines.append(f"Selected features ({len(selected)}): {', '.join(str(f) for f in selected[:15])}")
    return "\n".join(lines)


def _extract_generic(result: dict) -> str:
    """Fallback extractor for DS tool pages or unknown entry types."""
    lines = [f"Analysis: {result.get('test', result.get('title', 'Data Analysis'))}"]
    for key in ("summary", "best_model", "n_features", "n_samples", "accuracy",
                "precision", "recall", "f1", "auc", "r_squared", "mse"):
        if key in result:
            label = key.replace("_", " ").title()
            val = result[key]
            lines.append(f"{label}: {_fmt(val) if isinstance(val, (int, float)) else val}")
    return "\n".join(lines)


# Dispatcher: entry_type -> extractor function
_EXTRACTORS = {
    "independent_ttest": _extract_ttest,
    "paired_ttest": _extract_ttest,
    "one_sample_ttest": _extract_one_sample_ttest,
    "oneway_anova": _extract_anova,
    "twoway_anova": _extract_anova_general,
    "repeated_anova": _extract_anova_general,
    "mixed_anova": _extract_anova_general,
    "manova": _extract_manova,
    "mann_whitney": _extract_nonparametric,
    "wilcoxon": _extract_nonparametric,
    "kruskal_wallis": _extract_nonparametric,
    "friedman": _extract_nonparametric,
    "pearson_correlation": _extract_correlation,
    "spearman_correlation": _extract_correlation,
    "linear_regression": _extract_regression,
    "logistic_regression": _extract_regression,
    "multivariate_regression": _extract_regression,
    "chi_squared": _extract_chi_squared,
    "binomial": _extract_binomial,
    "descriptive_stats": _extract_descriptive,
    "model_arena": _extract_model_arena,
    "feature_selection": _extract_feature_selection,
    "data_profiler": _extract_generic,
    "smart_cleaning": _extract_generic,
    "feature_engineering": _extract_generic,
    "class_imbalance": _extract_generic,
    "hyperparameter": _extract_generic,
    "explainability": _extract_generic,
    "data_drift": _extract_generic,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_brief_prompt(
    entry_type: str,
    result: dict,
    variables: dict | None = None,
    alpha: float = 0.05,
) -> str:
    """Build a prompt requesting a 2-3 sentence interpretation."""
    extractor = _EXTRACTORS.get(entry_type, _extract_generic)
    result_text = extractor(result)
    var_text = _var_description(variables)

    return (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"Provide a 2-3 sentence plain-language interpretation of these results.\n"
        f"State whether the result is statistically significant at alpha = {alpha}, "
        f"what the effect size means practically, and one key takeaway.\n\n"
        f"Variables: {var_text}\n\n"
        f"Results:\n{result_text}"
    )


def build_deep_dive_prompt(
    entry_type: str,
    result: dict,
    variables: dict | None = None,
    alpha: float = 0.05,
) -> str:
    """Build a prompt for a detailed explanation with recommendations."""
    extractor = _EXTRACTORS.get(entry_type, _extract_generic)
    result_text = extractor(result)
    var_text = _var_description(variables)
    assumptions_text = _assumption_summary(result.get("assumptions"))

    return (
        f"{_DEEP_DIVE_CONTEXT}\n\n"
        f"Variables: {var_text}\n"
        f"Significance level (alpha): {alpha}\n\n"
        f"Results:\n{result_text}\n\n"
        f"Assumption checks:\n{assumptions_text}"
    )


def build_data_plan_prompt(
    columns: list[dict],
    n_rows: int,
    n_cols: int,
    target_col: str | None = None,
) -> str:
    """Build a prompt requesting a recommended analysis workflow after file upload."""
    col_lines = []
    for col in columns:
        parts = [f"  - {col['name']} (dtype: {col.get('dtype', '?')})"]
        extras = []
        if "n_unique" in col:
            extras.append(f"unique={col['n_unique']}")
        if "n_missing" in col and col["n_missing"] > 0:
            extras.append(f"missing={col['n_missing']}")
        if "mean" in col and col["mean"] is not None:
            extras.append(f"mean={_fmt(col['mean'], 2)}")
        if "std" in col and col["std"] is not None:
            extras.append(f"std={_fmt(col['std'], 2)}")
        if "min" in col and col["min"] is not None:
            extras.append(f"range=[{_fmt(col['min'], 2)}, {_fmt(col.get('max'), 2)}]")
        if extras:
            parts.append(f"  ({', '.join(extras)})")
        col_lines.append("".join(parts))

    col_text = "\n".join(col_lines)
    target_text = f"\nLikely target variable: {target_col}" if target_col else ""

    return (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"I have a dataset with {n_rows:,} rows and {n_cols} columns.{target_text}\n\n"
        f"Columns:\n{col_text}\n\n"
        "Based on this dataset, please recommend:\n"
        "1. A step-by-step analysis workflow\n"
        "2. Which statistical tests would be appropriate and why\n"
        "3. Which columns are likely targets vs. features\n"
        "4. Potential data quality issues to investigate\n"
        "5. Suggested visualizations to explore the data\n"
        "6. Any feature engineering opportunities\n\n"
        "Reference specific column names in your recommendations. "
        "Keep the plan actionable and focused."
    )


def build_clipboard_text(prompt: str) -> str:
    """Wrap a prompt for clipboard copy when using the browser fallback."""
    return (
        "--- Copy everything below this line and paste into Gemini ---\n\n"
        f"{prompt}\n\n"
        "--- End of prompt ---"
    )

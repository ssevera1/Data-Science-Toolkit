"""Application constants: variable types, test names, thresholds."""

# Variable types
METRIC = "Metric"
NOMINAL = "Nominal"
ORDINAL = "Ordinal"

VARIABLE_TYPES = [METRIC, NOMINAL, ORDINAL]

# Default number of rows/columns for new data
DEFAULT_ROWS = 20
DEFAULT_COLS = 5

# Significance level
ALPHA = 0.05

# Test categories for navigation
TEST_CATEGORIES = {
    "Data": ["Data Input"],
    "Descriptive": ["Descriptive Statistics"],
    "t-Tests": [
        "One-Sample t-Test",
        "Independent t-Test",
        "Paired t-Test",
    ],
    "ANOVA": [
        "One-Way ANOVA",
        "Two-Way ANOVA",
        "Repeated Measures ANOVA",
        "Mixed ANOVA",
    ],
    "Non-Parametric": [
        "Mann-Whitney U",
        "Wilcoxon Signed-Rank",
        "Kruskal-Wallis",
        "Friedman Test",
    ],
    "Correlation": [
        "Pearson Correlation",
        "Spearman Correlation",
    ],
    "Regression": [
        "Linear Regression",
        "Logistic Regression",
    ],
    "Other Tests": [
        "Chi-Squared Test",
        "Binomial Test",
    ],
    "Survival": [
        "Survival Analysis",
    ],
}

# P-value interpretation thresholds
P_THRESHOLDS = {
    "significant": 0.05,
    "marginal": 0.10,
}

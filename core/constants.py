"""Application constants: colors, variable types, test names."""

# Brand colors (dark theme)
ACCENT_PRIMARY = "#667eea"
ACCENT_SECONDARY = "#764ba2"
DARK_BG = "#0e1117"
CARD_BG = "#1a1a2e"
WHITE = "#ffffff"
TEXT_BODY = "#e0e0e0"
TEXT_MUTED = "#a0a0b8"
BORDER = "#2a2a4a"
GREEN = "#2ecc71"
RED = "#e74c3c"
YELLOW = "#f39c12"

# Variable types
VARIABLE_TYPES = ["Metric", "Nominal", "Ordinal"]

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
}

# P-value interpretation thresholds
P_THRESHOLDS = {
    "significant": 0.05,
    "marginal": 0.10,
}

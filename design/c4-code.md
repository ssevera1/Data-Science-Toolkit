# C4 Model - Level 4: Code Diagram

## Overview

The Code level diagram reveals the internal structure of key modules: function
signatures, data structures, and the fine-grained relationships between code
elements. This level is most useful for onboarding developers and understanding
implementation contracts.

## State Management (core/state.py)

```mermaid
classDiagram
    class SessionState {
        <<Streamlit st.session_state>>
        +DataFrame df
        +Dict~str,str~ stats_var_types
        +str current_theme
        +str active_page
        +Dict tool_state
    }

    class StateManager {
        <<module: core/state.py>>
        +init_session_state() void
        +get_dataframe() DataFrame | None
        +set_dataframe(df: DataFrame) void
        +get_variable_types() Dict
        +set_variable_type(col: str, vtype: str) void
        +clear_state() void
    }

    class DataManager {
        <<module: core/data_manager.py>>
        +load_csv(file: UploadedFile) DataFrame
        +load_excel(file: UploadedFile) DataFrame
        +load_clipboard(text: str) DataFrame
        +export_csv(df: DataFrame) bytes
        +export_excel(df: DataFrame) bytes
        -_auto_detect_types(df: DataFrame) Dict
        -_sanitize_for_export(df: DataFrame) DataFrame
        -_is_formula_injection(val: str) bool
    }

    class Validators {
        <<module: core/validators.py>>
        +validate_columns(df, cols: List) Result
        +validate_numeric(df, col: str) Result
        +validate_groups(df, col: str, min_groups: int) Result
        +validate_paired(df, col1: str, col2: str) Result
        +validate_binary(df, col: str) Result
        +validate_sample_size(n: int, minimum: int) Result
    }

    class Constants {
        <<module: core/constants.py>>
        +float ALPHA = 0.05
        +Dict VAR_TYPES
        +Dict TEST_CATEGORIES
        +Tuple DEFAULT_DIMS
    }

    StateManager --> SessionState : reads/writes
    DataManager --> StateManager : updates state
    DataManager --> Constants : uses defaults
    Validators --> Constants : uses thresholds
```

## Statistical Test Contract

All statistical test functions follow a common contract. This diagram shows the
shared interface pattern:

```mermaid
classDiagram
    class TestResult {
        <<TypedDict>>
        +str test_name
        +float statistic
        +float p_value
        +float effect_size
        +str effect_size_label
        +str interpretation
        +Dict assumptions
        +DataFrame | None posthoc
        +Dict confidence_interval
    }

    class TTestModule {
        <<module: stats/ttest.py>>
        +one_sample_ttest(df, col, test_value, alpha) TestResult
        +independent_ttest(df, dv, group, alpha) TestResult
        +paired_ttest(df, col1, col2, alpha) TestResult
        -_welch_correction(group1, group2) float
        -_cohens_d_independent(g1, g2) float
        -_cohens_d_paired(d) float
    }

    class ANOVAModule {
        <<module: stats/anova.py>>
        +oneway_anova(df, dv, group, alpha) TestResult
        +twoway_anova(df, dv, factor1, factor2, alpha) TestResult
        +repeated_anova(df, dv, subject, within, alpha) TestResult
        +mixed_anova(df, dv, subject, within, between, alpha) TestResult
        -_tukey_posthoc(df, dv, group) DataFrame
        -_calculate_eta_squared(ss_effect, ss_total) float
    }

    class MANOVAModule {
        <<module: stats/manova.py>>
        +run_manova(df, dvs, group, alpha) MANOVAResult
        -_multivariate_tests(manova_obj) DataFrame
        -_univariate_followups(df, dvs, group) List~TestResult~
    }

    class NonParametricModule {
        <<module: stats/nonparametric.py>>
        +mann_whitney(df, dv, group, alpha) TestResult
        +wilcoxon(df, col1, col2, alpha) TestResult
        +kruskal_wallis(df, dv, group, alpha) TestResult
        +friedman(df, measures, alpha) TestResult
        -_rank_biserial(U, n1, n2) float
    }

    class CorrelationModule {
        <<module: stats/correlation.py>>
        +pearson_correlation(df, var1, var2, alpha) TestResult
        +spearman_correlation(df, var1, var2, alpha) TestResult
        -_fisher_z_ci(r, n, alpha) Tuple
    }

    class RegressionModule {
        <<module: stats/regression.py>>
        +linear_regression(df, dv, ivs, alpha) RegressionResult
        +logistic_regression(df, dv, ivs, alpha) RegressionResult
        -_standardized_betas(model) Series
        -_odds_ratios(model) DataFrame
    }

    class AssumptionsModule {
        <<module: stats/assumptions.py>>
        +shapiro_wilk(data, alpha) AssumptionResult
        +levene_test(groups, alpha) AssumptionResult
        +mauchly_test(df, dv, subject, within) AssumptionResult
        +henze_zirkler(df, dvs) AssumptionResult
        +box_m_test(df, dvs, group) AssumptionResult
        -_per_group_normality(df, dv, group, alpha) List
    }

    class EffectSizeModule {
        <<module: stats/effect_size.py>>
        +cohens_d_one_sample(data, test_value) EffectResult
        +cohens_d_independent(group1, group2) EffectResult
        +cohens_d_paired(diff) EffectResult
        +eta_squared(ss_effect, ss_total) EffectResult
        +omega_squared(ss_effect, ss_error, df_effect, ms_error) EffectResult
        +cramers_v(chi2, n, min_dim) EffectResult
        +rank_biserial(U, n1, n2) EffectResult
        -_interpret_d(d) str
        -_interpret_eta(eta) str
    }

    TTestModule ..> TestResult : returns
    ANOVAModule ..> TestResult : returns
    MANOVAModule ..> TestResult : returns
    NonParametricModule ..> TestResult : returns
    CorrelationModule ..> TestResult : returns
    RegressionModule ..> TestResult : returns

    TTestModule --> AssumptionsModule : checks assumptions
    TTestModule --> EffectSizeModule : computes effect size
    ANOVAModule --> AssumptionsModule : checks assumptions
    ANOVAModule --> EffectSizeModule : computes effect size
    MANOVAModule --> AssumptionsModule : checks assumptions
    NonParametricModule --> EffectSizeModule : computes effect size
```

## Variable Type Auto-Detection Logic

```mermaid
flowchart TD
    START[Column from DataFrame] --> NUMERIC{Can convert >50%<br/>to numeric?}

    NUMERIC -->|Yes| UNIQUE_NUM{Count unique<br/>values}
    NUMERIC -->|No| UNIQUE_TEXT{Count unique<br/>values}

    UNIQUE_NUM -->|"<= 2"| NOMINAL_N[Nominal]
    UNIQUE_NUM -->|"3-7 & sparse"| ORDINAL[Ordinal]
    UNIQUE_NUM -->|"> 7"| METRIC[Metric]

    UNIQUE_TEXT -->|"<= 10"| NOMINAL_T[Nominal]
    UNIQUE_TEXT -->|"> 10"| TEXT[Text / Excluded]

    style METRIC fill:#4caf50,stroke:#2e7d32,color:#fff
    style NOMINAL_N fill:#ff9800,stroke:#e65100,color:#fff
    style NOMINAL_T fill:#ff9800,stroke:#e65100,color:#fff
    style ORDINAL fill:#2196f3,stroke:#1565c0,color:#fff
    style TEXT fill:#9e9e9e,stroke:#616161,color:#fff
```

## Chart Generation Pattern

```mermaid
classDiagram
    class ChartTheme {
        <<module: charts/theme.py>>
        +register_plotly_template(theme: str) void
        +get_color_palette(theme: str) List~str~
        +get_plotly_template(theme: str) Template
        -_light_template() Template
        -_dark_template() Template
    }

    class BoxPlot {
        <<module: charts/boxplot.py>>
        +grouped_boxplot(df, dv, group, theme) Figure
        +paired_boxplot(df, col1, col2, theme) Figure
        +single_boxplot(df, col, theme) Figure
        -_add_jitter_points(fig, data) void
    }

    class Histogram {
        <<module: charts/histogram.py>>
        +histogram(df, col, bins, theme) Figure
        +histogram_with_normal(df, col, theme) Figure
        -_overlay_normal_curve(fig, data) void
    }

    class ScatterPlot {
        <<module: charts/scatter.py>>
        +scatter_plot(df, x, y, theme) Figure
        +scatter_with_regression(df, x, y, theme) Figure
        -_add_regression_line(fig, x, y) void
    }

    class QQPlot {
        <<module: charts/qq_plot.py>>
        +qq_plot(data, theme) Figure
        -_theoretical_quantiles(n) ndarray
    }

    BoxPlot --> ChartTheme : uses template
    Histogram --> ChartTheme : uses template
    ScatterPlot --> ChartTheme : uses template
    QQPlot --> ChartTheme : uses template
```

## UI Component Rendering Pattern

```mermaid
classDiagram
    class VariableSelector {
        <<module: components/variable_selector.py>>
        +select_metric_variable(df, label, key) str | None
        +select_nominal_variable(df, label, key) str | None
        +select_ordinal_variable(df, label, key) str | None
        +select_any_variable(df, label, key) str | None
        +select_multiple_variables(df, label, key) List~str~
        -_filter_by_type(df, var_types, target_type) List~str~
        -_format_option(col, dtype) str
    }

    class ResultsDisplay {
        <<module: components/results_display.py>>
        +render_significance_result(test_name, stat, p, alpha) void
        +render_assumption_check(name, passed, stat, p) void
        +render_effect_size(label, value, interpretation) void
        +render_posthoc_table(posthoc_df) void
        +render_interpretation(text) void
        -_significance_color(p, alpha) str
        -_assumption_badge(passed) str
        -_effect_gauge(value) str
    }

    class DataTable {
        <<module: components/data_table.py>>
        +render_data_preview(df, var_types, max_rows) void
        +render_type_badges(var_types) void
        -_truncate_display(df, max_rows) DataFrame
        -_type_badge_color(vtype) str
    }

    VariableSelector ..> SessionState : reads var_types
    ResultsDisplay ..> ThemeUtils : uses colors
    DataTable ..> SessionState : reads df

    class ThemeUtils {
        <<module: utils/theme.py>>
        +apply_theme(theme: str) void
        +get_css(theme: str) str
        +get_metric_card_css(theme: str) str
        -_light_css() str
        -_dark_css() str
    }
```

## Page Module Pattern (Statistics)

Every statistics page follows the same structural template:

```mermaid
flowchart TD
    subgraph PAGE["Statistics Page Module"]
        direction TB

        GUARD["Guard Check
        ─────────────
        if df is None:
            show_warning()
            return"]

        HEADER["Page Header
        ────────────
        st.title(test_name)
        st.markdown(description)"]

        PREVIEW["Data Preview
        ─────────────
        data_table.render_data_preview()"]

        CONFIG["Configuration
        ──────────────
        variable_selector.select_*()
        alpha = st.slider()"]

        VALIDATE["Validation
        ──────────
        validators.validate_*()"]

        RUN["Run Button
        ──────────
        if st.button('Run Test'):"]

        subgraph TABS["Result Tabs"]
            TAB_R["Results Tab
            ────────────
            results_display.render_*()"]
            TAB_A["Assumptions Tab
            ────────────────
            assumption checks"]
            TAB_C["Charts Tab
            ───────────
            charts.*.create_*()"]
        end

        GUIDE["Page Guide
        ───────────
        st.expander('Guide')
        - When to use
        - Assumptions
        - Interpretation"]
    end

    GUARD --> HEADER --> PREVIEW --> CONFIG --> VALIDATE
    VALIDATE -->|Valid| RUN --> TABS
    VALIDATE -->|Invalid| ERROR[Error Message]
    TABS --> GUIDE

    style PAGE fill:#e3f2fd,stroke:#1565c0
    style TABS fill:#bbdefb,stroke:#1976d2
```

## ML Pipeline Module Pattern (Data Science)

```mermaid
flowchart TD
    subgraph DS_PAGE["Data Science Page Module"]
        direction TB

        GUARD_DS["Guard Check
        if df is None: return"]

        HEADER_DS["Page Header
        emoji + title + tagline"]

        CONFIG_DS["Configuration Panel
        ──────────────────────
        Target variable selection
        Feature selection
        Algorithm parameters
        Cross-validation settings"]

        EXECUTE["Execute Pipeline
        ────────────────
        scikit-learn / XGBoost / etc.
        Progress bars
        Live metric updates"]

        subgraph RESULTS["Multi-Tab Results"]
            OVERVIEW["Overview
            Summary metrics"]
            DETAILS["Details
            Per-model/feature tables"]
            VISUALS["Visualizations
            Plotly / Matplotlib charts"]
        end

        EXPORT_DS["Export Options
        CSV / Excel download"]
    end

    GUARD_DS --> HEADER_DS --> CONFIG_DS --> EXECUTE --> RESULTS --> EXPORT_DS

    style DS_PAGE fill:#e8f5e9,stroke:#2e7d32
    style RESULTS fill:#c8e6c9,stroke:#388e3c
```

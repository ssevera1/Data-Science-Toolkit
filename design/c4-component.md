# C4 Model - Level 3: Component Diagram

## Overview

The Component diagram zooms into each container to reveal the internal modules
and their interactions. This level shows how responsibilities are distributed
within each architectural layer.

## Core Layer Components

```mermaid
flowchart TB
    subgraph CORE["Core Layer"]
        direction TB

        STATE["state.py
        ───────────────
        init_session_state()
        get_dataframe()
        set_dataframe()
        get_variable_types()
        set_variable_type()"]

        DM["data_manager.py
        ───────────────────
        load_csv(file)
        load_excel(file)
        load_clipboard(text)
        export_csv(df)
        export_excel(df)
        auto_detect_types(df)
        sanitize_export(df)"]

        VAL["validators.py
        ────────────────
        validate_columns(df, cols)
        validate_numeric(df, col)
        validate_groups(df, col, min)
        validate_paired(df, col1, col2)
        validate_binary(df, col)
        validate_sample_size(n, min)"]

        CONST["constants.py
        ───────────────
        ALPHA = 0.05
        VAR_TYPES = {Metric, Nominal, Ordinal}
        TEST_CATEGORIES
        DEFAULT_DIMS"]
    end

    STATE --> DM
    DM --> CONST
    VAL --> CONST

    SS[(Session State)]
    STATE <--> SS
    DM --> SS

    style CORE fill:#f3e5f5,stroke:#7b1fa2
    style STATE fill:#ce93d8,stroke:#7b1fa2,color:#000
    style DM fill:#ce93d8,stroke:#7b1fa2,color:#000
    style VAL fill:#ce93d8,stroke:#7b1fa2,color:#000
    style CONST fill:#e1bee7,stroke:#7b1fa2,color:#000
    style SS fill:#f9a825,stroke:#f57f17,color:#000
```

## Statistics Engine Components

```mermaid
flowchart TB
    subgraph STATS["Statistics Engine"]
        direction TB

        subgraph PARAMETRIC["Parametric Tests"]
            TTEST["ttest.py
            ─────────
            one_sample_ttest()
            independent_ttest()
            paired_ttest()"]

            ANOVA["anova.py
            ────────
            oneway_anova()
            twoway_anova()
            repeated_anova()
            mixed_anova()
            tukey_posthoc()"]

            REG["regression.py
            ──────────────
            linear_regression()
            logistic_regression()"]
        end

        subgraph MULTIVARIATE["Multivariate Tests"]
            MANOVA["manova.py
            ─────────
            run_manova()
            multivariate_tests()
            univariate_followups()"]

            MVREG["multivariate_regression.py
            ──────────────────────────
            multivariate_regression()
            per_dv_ols_models()"]
        end

        subgraph NONPARAM["Non-Parametric Tests"]
            NP["nonparametric.py
            ─────────────────
            mann_whitney()
            wilcoxon()
            kruskal_wallis()
            friedman()"]
        end

        subgraph CATEGORICAL["Categorical Tests"]
            CHI["chi_squared.py
            ───────────────
            chi_squared_test()
            contingency_table()"]

            BIN["binomial.py
            ────────────
            binomial_test()
            proportion_ci()"]
        end

        subgraph FOUNDATION["Foundation"]
            DESC["descriptive.py
            ───────────────
            describe_numeric()
            describe_categorical()
            frequency_table()"]

            ASSUME["assumptions.py
            ────────────────
            shapiro_wilk()
            levene_test()
            mauchly_test()
            henze_zirkler()
            box_m_test()"]

            EFFECT["effect_size.py
            ──────────────
            cohens_d()
            eta_squared()
            omega_squared()
            cramers_v()
            rank_biserial()"]

            CORR["correlation.py
            ───────────────
            pearson_correlation()
            spearman_correlation()
            fisher_z_ci()"]
        end
    end

    TTEST --> ASSUME & EFFECT
    ANOVA --> ASSUME & EFFECT
    MANOVA --> ASSUME
    NP --> EFFECT
    CHI --> EFFECT
    REG --> ASSUME

    style STATS fill:#e0f2f1,stroke:#004d40
    style PARAMETRIC fill:#b2dfdb,stroke:#00695c
    style MULTIVARIATE fill:#b2dfdb,stroke:#00695c
    style NONPARAM fill:#b2dfdb,stroke:#00695c
    style CATEGORICAL fill:#b2dfdb,stroke:#00695c
    style FOUNDATION fill:#80cbc4,stroke:#00695c
```

## Pages Layer Components

```mermaid
flowchart TB
    subgraph PAGES["Pages Layer (32 modules)"]
        direction TB

        HOME["home.py
        ────────
        Hero image
        Data upload widget
        Tool card grid
        Navigation links"]

        subgraph DS_TOOLS["Data Science Tools (9 pages)"]
            PROF["data_profiler.py
            Automated EDA"]
            CLEAN["smart_cleaning.py
            Data preprocessing"]
            FEAT_ENG["feature_engineering.py
            Auto-feature creation"]
            FEAT_SEL["feature_selection.py
            Feature importance"]
            CLASS_IMB["class_imbalance.py
            SMOTE / sampling"]
            ARENA["model_arena.py
            Model benchmarking"]
            HYPER["hyperparameter_tuning.py
            Bayesian optimization"]
            EXPLAIN["explainability.py
            SHAP analysis"]
            DRIFT["data_drift.py
            Distribution shifts"]
        end

        subgraph STATS_TOOLS["Statistics Tools (20 pages)"]
            DI["data_input.py"]
            DS["descriptive_stats.py"]
            T1["one_sample_ttest.py"]
            T2["independent_ttest.py"]
            T3["paired_ttest.py"]
            A1["oneway_anova.py"]
            A2["twoway_anova.py"]
            A3["repeated_anova.py"]
            A4["mixed_anova.py"]
            MAN["manova.py"]
            MW["mann_whitney.py"]
            WX["wilcoxon.py"]
            KW["kruskal_wallis.py"]
            FR["friedman.py"]
            PC["pearson_correlation.py"]
            SC["spearman_correlation.py"]
            LR["linear_regression.py"]
            LOR["logistic_regression.py"]
            MVR["multivariate_regression.py"]
            CSQ["chi_squared_test.py"]
        end
    end

    style PAGES fill:#e3f2fd,stroke:#1565c0
    style DS_TOOLS fill:#bbdefb,stroke:#1976d2
    style STATS_TOOLS fill:#bbdefb,stroke:#1976d2
    style HOME fill:#90caf9,stroke:#1565c0,color:#000
```

## Visualization Engine Components

```mermaid
flowchart TB
    subgraph VIZ["Visualization Engine"]
        direction LR

        subgraph CHART_TYPES["Chart Generators"]
            BOX["boxplot.py
            ─────────
            grouped_boxplot()
            paired_boxplot()
            single_boxplot()"]

            HIST["histogram.py
            ────────────
            histogram()
            histogram_with_normal()"]

            SCAT["scatter.py
            ──────────
            scatter_plot()
            scatter_with_regression()"]

            BAR["barplot.py
            ─────────
            grouped_barplot()
            mean_barplot()"]

            QQ["qq_plot.py
            ─────────
            qq_plot()"]

            REGP["regression_plot.py
            ─────────────────
            regression_line()
            confidence_band()"]
        end

        CTHEME["theme.py
        ────────
        register_template()
        get_color_palette()
        get_plotly_template()"]
    end

    BOX & HIST & SCAT & BAR & QQ & REGP --> CTHEME

    style VIZ fill:#fbe9e7,stroke:#bf360c
    style CHART_TYPES fill:#ffccbc,stroke:#d84315
    style CTHEME fill:#ff8a65,stroke:#bf360c,color:#000
```

## UI Components Detail

```mermaid
flowchart TB
    subgraph COMP["UI Components"]
        direction LR

        VS["variable_selector.py
        ──────────────────────
        select_metric_variable(df)
        select_nominal_variable(df)
        select_ordinal_variable(df)
        select_any_variable(df)
        select_multiple_variables(df)"]

        RD["results_display.py
        ────────────────────
        render_significance_result()
        render_assumption_check()
        render_effect_size()
        render_posthoc_table()
        render_interpretation()"]

        DT["data_table.py
        ───────────────
        render_data_preview()
        render_type_badges()"]

        SN["sidebar_nav.py
        ───────────────
        render_nav_links()
        render_section_header()"]
    end

    style COMP fill:#fff3e0,stroke:#e65100
    style VS fill:#ffcc80,stroke:#ef6c00,color:#000
    style RD fill:#ffcc80,stroke:#ef6c00,color:#000
    style DT fill:#ffcc80,stroke:#ef6c00,color:#000
    style SN fill:#ffcc80,stroke:#ef6c00,color:#000
```

## Cross-Container Component Interactions

```mermaid
flowchart TD
    subgraph PAGE["Statistics Page (e.g., oneway_anova.py)"]
        RENDER["render()"]
    end

    subgraph CORE["Core"]
        GET_DF["state.get_dataframe()"]
        VALIDATE["validators.validate_groups()"]
    end

    subgraph COMP["Components"]
        VAR_SEL["variable_selector.select_metric_variable()"]
        GRP_SEL["variable_selector.select_nominal_variable()"]
        RESULT["results_display.render_significance_result()"]
        ASSUME_R["results_display.render_assumption_check()"]
    end

    subgraph STATS["Statistics Engine"]
        ANOVA_FN["anova.oneway_anova()"]
        ASSUME_FN["assumptions.shapiro_wilk()"]
        EFFECT_FN["effect_size.eta_squared()"]
    end

    subgraph VIZ["Charts"]
        BOX_FN["boxplot.grouped_boxplot()"]
    end

    subgraph THEME["Utils"]
        STYLE["theme.apply_theme()"]
    end

    RENDER --> GET_DF
    RENDER --> VAR_SEL & GRP_SEL
    RENDER --> VALIDATE
    RENDER --> ANOVA_FN
    ANOVA_FN --> ASSUME_FN & EFFECT_FN
    RENDER --> BOX_FN
    RENDER --> RESULT & ASSUME_R
    RENDER --> STYLE

    style PAGE fill:#bbdefb,stroke:#1565c0
    style CORE fill:#ce93d8,stroke:#7b1fa2
    style COMP fill:#ffcc80,stroke:#ef6c00
    style STATS fill:#80cbc4,stroke:#00695c
    style VIZ fill:#ffab91,stroke:#bf360c
    style THEME fill:#dce775,stroke:#827717
```

## Data Science Pipeline Flow

```mermaid
flowchart LR
    subgraph PIPELINE["DS Tools Pipeline (sequential, user-driven)"]
        direction LR
        P1["1. Data Profiler
        Automated EDA"]
        P2["2. Smart Cleaning
        Impute, encode, deduplicate"]
        P3["3. Feature Engineering
        Polynomial, interaction, bins"]
        P4["4. Feature Selection
        Correlation, MI, RFE"]
        P5["5. Class Imbalance
        SMOTE, sampling"]
        P6["6. Model Arena
        Benchmark 10+ models"]
        P7["7. Hyperparameter Tuning
        Bayesian optimization"]
        P8["8. Explainability
        SHAP values"]
        P9["9. Data Drift
        Distribution monitoring"]
    end

    P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8
    P6 -.->|"Compare with"| P9

    DF[(Shared DataFrame)]
    DF --> P1
    P2 --> DF
    P3 --> DF
    P5 --> DF

    style PIPELINE fill:#e3f2fd,stroke:#1565c0
    style DF fill:#f9a825,stroke:#f57f17,color:#000
```

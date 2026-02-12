"""One-Way ANOVA page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import (
    render_significance_result, render_assumption_check,
    render_effect_size, interpret_eta_squared,
)
from stats.anova import oneway_anova
from core.validators import validate_groups
from charts.boxplot import grouped_boxplot
from charts.barplot import group_means_bar
from core.state import get_df


def render():
    st.title("One-Way ANOVA")
    st.markdown("Compare means across three or more independent groups.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        dv = select_metric_variable("Dependent variable", key="ow_dv")
    with col2:
        group = select_nominal_variable("Factor (grouping variable)", key="ow_group")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="ow_alpha")

    if dv and group and st.button("Calculate", type="primary"):
        valid, msg = validate_groups(dv, group, min_groups=2)
        if not valid:
            st.error(msg)
            return

        clean = df[[dv, group]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        result = oneway_anova(clean, dv, group, alpha=alpha)

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "F", result["F"], result["p"],
                (result["df_between"], result["df_within"]), alpha
            )
            st.markdown("---")

            st.markdown("**ANOVA Table**")
            st.dataframe(result["anova_table"], width="stretch", hide_index=True)

            st.markdown("**Group Descriptives**")
            st.dataframe(result["group_desc"], width="stretch", hide_index=True)

            render_effect_size("η² (Eta-squared)", result["eta_squared"], interpret_eta_squared(result["eta_squared"]))
            render_effect_size("ω² (Omega-squared)", result["omega_squared"], interpret_eta_squared(result["omega_squared"]))

            if result["posthoc"] is not None:
                st.markdown("### Post-Hoc Tests (Tukey HSD)")
                st.dataframe(result["posthoc"], width="stretch", hide_index=True)

        with tab_assume:
            st.markdown("**Normality (Shapiro-Wilk per group)**")
            for gname, norm in result["assumptions"]["normality"].items():
                render_assumption_check(
                    f"Group: {gname}", norm["statistic"], norm["p_value"],
                    norm["passed"], norm["detail"]
                )
            st.markdown("---")
            homo = result["assumptions"]["homogeneity"]
            render_assumption_check(
                "Levene's Test (Homogeneity of Variances)", homo["statistic"],
                homo["p_value"], homo["passed"], homo["detail"]
            )

        with tab_chart:
            c1, c2 = st.columns(2)
            with c1:
                fig = grouped_boxplot(clean, dv, group)
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = group_means_bar(clean, dv, group)
                st.plotly_chart(fig, width="stretch")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
One-way ANOVA tests whether the **means differ across three or more independent groups**. It is an extension of the independent samples t-test to more than two groups.

#### When to Use
- One **continuous dependent variable** (DV).
- One **categorical independent variable** (factor / grouping variable) with **3 or more groups**.
- Groups are **independent** (different participants in each group).

#### Results Tab
- **F-statistic** and **p-value** -- test the overall null hypothesis that all group means are equal.
- **Degrees of freedom** -- df between groups and df within groups.
- **ANOVA table** -- displays Sum of Squares (SS), degrees of freedom (df), Mean Square (MS), F-statistic, and p-value for between-groups and within-groups sources.
- **Group descriptives** -- N, Mean, and SD for each group.
- **Effect sizes**:
    - **Eta-squared** -- proportion of total variance explained by the factor. Interpretation: **small ~ 0.01**, **medium ~ 0.06**, **large ~ 0.14**.
    - **Omega-squared** -- a less biased estimate of effect size, especially useful for small samples.
- **Post-hoc tests (Tukey HSD)** -- pairwise comparisons that identify **which specific groups differ** from each other. Only displayed when the overall F-test is significant.

#### Assumptions Tab
- **Shapiro-Wilk per group** -- tests normality within each group. ANOVA is robust to moderate normality violations, especially with balanced designs and larger samples.
- **Levene's test** -- tests **homogeneity of variances** across groups. If significant (p < 0.05), consider Welch's ANOVA or a non-parametric alternative (Kruskal-Wallis).

#### Charts Tab
- **Grouped box plot** -- compares distributions across all groups.
- **Group means bar chart with error bars** -- visualizes mean differences between groups.
        """)


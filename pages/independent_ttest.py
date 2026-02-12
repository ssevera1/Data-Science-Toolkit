"""Independent Samples t-Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import (
    render_significance_result, render_assumption_check,
    render_effect_size, interpret_cohens_d,
)
from stats.ttest import independent_ttest
from core.validators import validate_groups
from charts.boxplot import grouped_boxplot
from charts.qq_plot import qq_plot
from core.state import get_df


def render():
    st.title("Independent Samples t-Test")
    st.markdown("Compare means between two independent groups.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        dv = select_metric_variable("Dependent variable (metric)", key="ind_dv")
    with col2:
        group = select_nominal_variable("Grouping variable (2 groups)", key="ind_group")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="ind_alpha")
        equal_var = st.checkbox("Assume equal variances (Student's t)", value=True, key="ind_eqvar")

    if dv and group and st.button("Calculate", type="primary"):
        valid, msg = validate_groups(dv, group, min_groups=2, max_groups=2)
        if not valid:
            st.error(msg)
            return

        clean = df[[dv, group]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        groups = clean[group].unique()
        g1 = clean[clean[group] == groups[0]][dv].values
        g2 = clean[clean[group] == groups[1]][dv].values

        result = independent_ttest(g1, g2, equal_var=equal_var, alpha=alpha)

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "t", result["t"], result["p"], result["df"], alpha
            )
            st.markdown("---")

            st.markdown("**Group Statistics**")
            group_stats = pd.DataFrame({
                "Group": [str(groups[0]), str(groups[1])],
                "N": [result["n1"], result["n2"]],
                "Mean": [result["mean1"], result["mean2"]],
                "SD": [result["sd1"], result["sd2"]],
            })
            st.dataframe(group_stats, use_container_width=True, hide_index=True)

            st.markdown(f"**Mean Difference:** {result['mean_diff']:.4f} (SE = {result['se_diff']:.4f})")
            st.markdown(f"**95% CI:** [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

            render_effect_size("Cohen's d", result["cohens_d"], interpret_cohens_d(result["cohens_d"]))

        with tab_assume:
            for i, gname in enumerate(groups):
                key = f"normality_group{i+1}"
                norm = result["assumptions"][key]
                render_assumption_check(
                    f"Shapiro-Wilk: {gname}", norm["statistic"], norm["p_value"],
                    norm["passed"], norm["detail"]
                )
            homo = result["assumptions"]["homogeneity"]
            render_assumption_check(
                "Levene's Test (Homogeneity)", homo["statistic"], homo["p_value"],
                homo["passed"], homo["detail"]
            )
            if not homo["passed"]:
                st.info("Consider using Welch's t-test (uncheck 'Assume equal variances').")

        with tab_chart:
            fig = grouped_boxplot(clean, dv, group)
            st.plotly_chart(fig, use_container_width=True)

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
The independent samples t-test compares the **means of two independent (unrelated) groups** on a continuous variable to determine whether they differ significantly.

#### When to Use
- Two **separate groups** of participants are each measured on the **same continuous variable** (e.g., treatment vs. control, male vs. female).
- The groups are **independent** -- different individuals in each group.

#### Input
- **Dependent variable** -- must be metric (continuous).
- **Grouping variable** -- must be nominal with **exactly 2 groups**.

#### Options
- **Assume equal variances (Student's t)** -- uses pooled variance. Appropriate when group variances are similar.
- **Unequal variances (Welch's t)** -- does not assume equal variances. Use this when Levene's test is significant (p < 0.05), indicating unequal group variances.

#### Results Tab
- **t-statistic** and **p-value** -- test whether the group means differ significantly.
- **Degrees of freedom (df)** -- depends on sample sizes (and is adjusted for Welch's t).
- **Group statistics** -- N, Mean, and SD for each group.
- **Mean difference** with standard error and **95% confidence interval**.
- **Cohen's d** -- standardized effect size: **small ~ 0.2**, **medium ~ 0.5**, **large ~ 0.8**.

#### Assumptions Tab
- **Shapiro-Wilk per group** -- tests normality within each group. The test is robust to violations with larger samples (N > 30 per group).
- **Levene's test** -- tests **homogeneity of variances**. If significant (p < 0.05), the group variances are unequal and Welch's t-test should be used.

#### Charts Tab
- **Grouped box plot** -- compares the distributions of the two groups side by side.
        """)


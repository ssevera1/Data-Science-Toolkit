"""MANOVA (Multivariate Analysis of Variance) page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_multiple_variables, select_nominal_variable
from components.results_display import (
    render_significance_result, render_assumption_check,
)
from stats.manova import manova
from core.validators import validate_groups
from charts.boxplot import grouped_boxplot
from core.state import get_df


def render():
    st.title("MANOVA")
    st.markdown(
        "Test whether group means differ across **multiple dependent variables** simultaneously."
    )

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    # ── Variable selection ────────────────────────────────────────────────
    dv_cols = select_multiple_variables(
        "Dependent variables (2 or more metric)", key="man_dv", var_type="Metric"
    )
    group = select_nominal_variable("Factor (grouping variable)", key="man_group")

    with st.expander("Options"):
        alpha = st.slider(
            "Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="man_alpha"
        )

    # ── Validation & computation ──────────────────────────────────────────
    if dv_cols and len(dv_cols) >= 2 and group and st.button("Calculate", type="primary"):
        # Check group_col is not among DVs
        if group in dv_cols:
            st.error("The grouping variable cannot also be a dependent variable.")
            return

        # Validate groups using the first DV (they all share the same rows after dropna)
        valid, msg = validate_groups(dv_cols[0], group, min_groups=2)
        if not valid:
            st.error(msg)
            return

        result = manova(df, dv_cols, group, alpha=alpha)

        # ── Tabs ──────────────────────────────────────────────────────────
        tab_res, tab_follow, tab_assume, tab_chart = st.tabs(
            ["Results", "Univariate Follow-ups", "Assumptions", "Charts"]
        )

        # ── Results tab ───────────────────────────────────────────────────
        with tab_res:
            # Significance banner from Wilks' lambda
            mt = result["manova_table"]
            if len(mt) > 0:
                wilks = mt.iloc[0]
                render_significance_result(
                    result["test"],
                    "Wilks' Λ F",
                    wilks["F"],
                    wilks["p"],
                    (int(wilks["Num DF"]), int(wilks["Den DF"])),
                    alpha,
                )
            st.markdown("---")

            st.markdown("**Multivariate Tests**")
            st.dataframe(result["manova_table"], width="stretch", hide_index=True)

            st.markdown("**Group Descriptives**")
            st.dataframe(result["group_desc"], width="stretch", hide_index=True)

            st.metric("N (complete cases)", result["n"])

        # ── Univariate Follow-ups tab ─────────────────────────────────────
        with tab_follow:
            for uv in result["univariate_anovas"]:
                st.markdown(f"### {uv['dv']}")
                st.dataframe(uv["anova_table"], width="stretch", hide_index=True)

            if result["posthoc"] is not None:
                st.markdown("### Post-Hoc Tests (Tukey HSD)")
                st.dataframe(result["posthoc"], width="stretch", hide_index=True)
            elif result["overall_p"] >= alpha:
                st.info(
                    "Post-hoc tests are not displayed because the overall "
                    "MANOVA was not significant."
                )

        # ── Assumptions tab ───────────────────────────────────────────────
        with tab_assume:
            st.markdown("**Multivariate Normality (Henze-Zirkler)**")
            mvn = result["assumptions"]["multivariate_normality"]
            render_assumption_check(
                "Henze-Zirkler test",
                mvn["statistic"],
                mvn["p_value"],
                mvn["passed"],
                mvn["detail"],
            )

            st.markdown("---")
            st.markdown("**Box's M (Homogeneity of Covariance Matrices)**")
            bm = result["assumptions"]["box_m"]
            render_assumption_check(
                "Box's M",
                bm["statistic"],
                bm["p_value"],
                bm["passed"],
                bm["detail"],
            )

            st.markdown("---")
            st.markdown("**Levene's Test (Homogeneity of Variances per DV)**")
            for dv, lev in result["assumptions"]["homogeneity"].items():
                render_assumption_check(
                    f"Levene — {dv}",
                    lev["statistic"],
                    lev["p_value"],
                    lev["passed"],
                    lev["detail"],
                )

        # ── Charts tab ────────────────────────────────────────────────────
        with tab_chart:
            clean = df[dv_cols + [group]].dropna()
            for c in dv_cols:
                clean[c] = pd.to_numeric(clean[c], errors="coerce")
            clean = clean.dropna()

            for dv in dv_cols:
                fig = grouped_boxplot(clean, dv, group)
                st.plotly_chart(fig, width="stretch")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
MANOVA (Multivariate Analysis of Variance) tests whether the **means of two or more dependent variables differ across groups** defined by a categorical factor.  It is the multivariate extension of one-way ANOVA.

#### When to Use
- Two or more **continuous dependent variables** (DVs).
- One **categorical independent variable** (factor) with **2 or more groups**.
- Groups are **independent** (different participants in each group).
- You want to test group differences on **multiple outcomes simultaneously** rather than running separate ANOVAs, which inflates the Type I error rate.

#### Results Tab
- **Multivariate test table** with four test statistics:
    - **Wilks' Lambda** — most commonly reported; smaller values indicate greater group separation.
    - **Pillai's Trace** — more robust when assumptions are violated.
    - **Hotelling-Lawley Trace** — powerful when group differences are concentrated on one dimension.
    - **Roy's Greatest Root** — most powerful when there is a single discriminant dimension, but most sensitive to violations.
- Each row shows the test **Value**, degrees of freedom (**Num DF**, **Den DF**), **F-statistic**, and **p-value**.
- **Group descriptives** — N, Mean, and SD for each DV within each group.

#### Univariate Follow-ups Tab
- Individual **one-way ANOVAs** for each DV, showing which specific outcomes differ across groups.
- **Post-hoc Tukey HSD** tests per DV (only if the overall MANOVA is significant) to identify which groups differ.

#### Assumptions Tab
- **Multivariate Normality (Henze-Zirkler)** — tests whether the DVs jointly follow a multivariate normal distribution.  MANOVA is moderately robust to violations with balanced designs and large samples.
- **Box's M** — tests homogeneity of covariance matrices across groups.  This test is very sensitive; a conservative α (e.g., .001) is often used.
- **Levene's test per DV** — checks univariate homogeneity of variances for each DV separately.

#### Charts Tab
- **Grouped box plots** for each DV, showing distributions across groups.
        """)

"""Kruskal-Wallis H Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import render_significance_result, render_effect_size
from stats.nonparametric import kruskal_wallis
from core.validators import validate_groups
from charts.boxplot import grouped_boxplot
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df


def render():
    st.title("Kruskal-Wallis H Test")
    st.markdown("Non-parametric alternative to one-way ANOVA.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        dv = select_metric_variable("Test variable", key="kw_dv")
    with col2:
        group = select_nominal_variable("Grouping variable", key="kw_group")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="kw_alpha")

    if dv and group and st.button("Calculate", type="primary"):
        valid, msg = validate_groups(dv, group, min_groups=2)
        if not valid:
            st.error(msg)
            return

        clean = df[[dv, group]].dropna()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        result = kruskal_wallis(clean, dv, group, alpha=alpha)

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "H", result["H"], result["p"], result["df"], alpha
            )
            st.markdown("---")

            st.markdown("**Group Descriptives**")
            st.dataframe(result["group_desc"], width="stretch", hide_index=True)

            render_effect_size("ε² (Epsilon-squared)", result["epsilon_squared"])

            if result["posthoc"] is not None:
                st.markdown("### Post-Hoc Pairwise Comparisons (Bonferroni)")
                st.dataframe(result["posthoc"], width="stretch", hide_index=True)

        with tab_chart:
            fig = grouped_boxplot(clean, dv, group)
            st.plotly_chart(fig, width="stretch")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = [_serialize_df(result["group_desc"], "Group Descriptives")]
        if result.get("posthoc") is not None:
            _tables.append(_serialize_df(result["posthoc"], "Post-Hoc Comparisons"))
        _log_entry = build_log_entry(
            entry_type="kruskal_wallis",
            title=f"Kruskal-Wallis H: {dv} by {group}",
            result=result,
            tables=_tables,
            variables={"dv": dv, "group": group},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="kw_pdf_chart")
        if _include_chart:
            _fig = grouped_boxplot(clean, dv, group)
            _log_entry["figures"] = [{"label": "Grouped Boxplot", "fig_dict": _fig.to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="kw_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="kruskal_wallis.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
The Kruskal-Wallis H test is a **non-parametric alternative to one-way ANOVA**. It tests whether the distributions of a variable differ across three or more independent groups.

#### When to Use
- One variable measured across **3 or more independent groups**.
- The data are **non-normally distributed**, **ordinal**, or have **unequal variances** across groups.
- As a robust alternative when ANOVA assumptions are not met.

#### How It Works
All observations from every group are **pooled and ranked** together. The test compares the **mean ranks** between groups. The H statistic follows a **chi-squared distribution** approximately.

#### Input
- **Test variable** -- metric or ordinal variable.
- **Grouping variable** -- nominal variable with 3 or more groups.

#### Results Tab
- **H statistic** -- the Kruskal-Wallis test statistic (chi-squared distributed).
- **p-value** -- probability of observing this result if all group distributions are identical.
- **Degrees of freedom** -- number of groups minus 1.
- **Group descriptives** -- N, Median, and Mean Rank for each group.
- **Epsilon-squared effect size** -- proportion of variance in ranks explained by group membership. Ranges from 0 (no effect) to 1 (complete separation).

#### Post-Hoc Comparisons
When the overall test is significant, **Bonferroni-corrected pairwise comparisons** are performed to identify which specific pairs of groups differ.

#### Charts Tab
- **Grouped box plot** -- compares distributions across all groups.
        """)


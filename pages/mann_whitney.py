"""Mann-Whitney U Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import render_significance_result, render_effect_size
from stats.nonparametric import mann_whitney
from core.validators import validate_groups
from charts.boxplot import grouped_boxplot
from core.state import get_df


def render():
    st.title("Mann-Whitney U Test")
    st.markdown("Non-parametric alternative to the independent samples t-test.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        dv = select_metric_variable("Test variable", key="mw_dv")
    with col2:
        group = select_nominal_variable("Grouping variable (2 groups)", key="mw_group")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="mw_alpha")
        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "greater", "less"], key="mw_alt")

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

        result = mann_whitney(g1, g2, alternative=alternative, alpha=alpha)

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "U", result["U"], result["p"], alpha=alpha
            )
            st.markdown("---")

            group_stats = pd.DataFrame({
                "Group": [str(groups[0]), str(groups[1])],
                "N": [result["n1"], result["n2"]],
                "Median": [result["median1"], result["median2"]],
            })
            st.dataframe(group_stats, use_container_width=True, hide_index=True)

            render_effect_size("Rank-biserial correlation", result["rank_biserial"])

        with tab_chart:
            fig = grouped_boxplot(clean, dv, group)
            st.plotly_chart(fig, use_container_width=True)

"""Mann-Whitney U Test page."""

import streamlit as st
import pandas as pd
from components.data_table import render_data_preview
from components.variable_selector import select_metric_variable, select_nominal_variable
from components.results_display import render_significance_result, render_effect_size
from stats.nonparametric import mann_whitney
from core.validators import validate_groups
from charts.boxplot import grouped_boxplot
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df

_CACHE_KEY = "_result_mann_whitney"


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

        clean = df[[dv, group]].dropna().copy()
        clean[dv] = pd.to_numeric(clean[dv], errors="coerce")
        clean = clean.dropna()

        groups = clean[group].unique()
        g1 = clean[clean[group] == groups[0]][dv].values
        g2 = clean[clean[group] == groups[1]][dv].values

        result = mann_whitney(g1, g2, alternative=alternative, alpha=alpha)

        group_stats = pd.DataFrame({
            "Group": [str(groups[0]), str(groups[1])],
            "N": [result["n1"], result["n2"]],
            "Median": [result["median1"], result["median2"]],
        })

        st.session_state[_CACHE_KEY] = {
            "inputs": (dv, group, alpha, alternative),
            "result": result,
            "group_stats": group_stats,
            "clean": clean,
        }

    # ── Cache invalidation ────────────────────────────────────────────
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached["inputs"] != (dv, group, alpha, alternative):
        del st.session_state[_CACHE_KEY]
        cached = None

    if cached:
        result = cached["result"]
        group_stats = cached["group_stats"]
        clean = cached["clean"]

        tab_res, tab_chart = st.tabs(["Results", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "U", result["U"], result["p"], alpha=alpha
            )
            st.markdown("---")

            st.dataframe(group_stats, width="stretch", hide_index=True)

            render_effect_size("Rank-biserial correlation", result["rank_biserial"])

        with tab_chart:
            fig = grouped_boxplot(clean, dv, group)
            st.plotly_chart(fig, width="stretch")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = [_serialize_df(group_stats, "Group Statistics")]
        _log_entry = build_log_entry(
            entry_type="mann_whitney",
            title=f"Mann-Whitney U: {dv} by {group}",
            result=result,
            tables=_tables,
            variables={"dv": dv, "group": group},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="mw_pdf_chart")
        if _include_chart:
            _fig = grouped_boxplot(clean, dv, group)
            _log_entry["figures"] = [{"label": "Grouped Boxplot", "fig_dict": _fig.to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="mw_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="mann_whitney.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
The Mann-Whitney U test is a **non-parametric alternative to the independent samples t-test**. It compares the distributions (and typically the medians) of two independent groups without assuming normality.

#### When to Use
- Two independent groups measured on the same variable.
- The data **violate the normality assumption** (e.g., heavily skewed distributions).
- The data are **ordinal** rather than truly continuous.
- **Small sample sizes** where the normality assumption cannot be verified.

#### How It Works
All observations from both groups are **pooled and ranked** together. The test compares the **sum of ranks** between the two groups. If one group consistently has higher values, its rank sum will be disproportionately large.

#### Input
- **Test variable** -- metric or ordinal variable.
- **Grouping variable** -- nominal variable with exactly **2 groups**.
- **Alternative hypothesis** -- choose between **two-sided** (groups differ), **greater** (group 1 > group 2), or **less** (group 1 < group 2).

#### Results Tab
- **U statistic** -- the test statistic based on rank sums.
- **p-value** -- probability of observing this result under the null hypothesis of equal distributions.
- **Group medians** and sample sizes.
- **Rank-biserial correlation** -- effect size ranging from **-1 to +1**. Values near 0 indicate no difference; values near -1 or +1 indicate a large difference.

#### Charts Tab
- **Grouped box plot** -- compares the distributions of the two groups visually.
        """)

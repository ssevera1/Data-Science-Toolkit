"""Chi-Squared Test page."""

import streamlit as st
import pandas as pd
import plotly.express as px
from components.data_table import render_data_preview
from components.variable_selector import select_nominal_variable
from components.results_display import render_significance_result, render_assumption_check, render_effect_size
from stats.chi_squared import chi_squared_test
from core.state import get_df, log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df
from charts.theme import apply_theme, get_chart_colors


def render():
    st.title("Chi-Squared Test")
    st.markdown("Test the association between two categorical variables.")

    if not render_data_preview():
        return

    df = get_df().dropna(how="all")

    col1, col2 = st.columns(2)
    with col1:
        var1 = select_nominal_variable("Variable 1", key="chi_v1")
    with col2:
        var2 = select_nominal_variable("Variable 2", key="chi_v2")

    with st.expander("Options"):
        alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01, key="chi_alpha")

    if var1 and var2 and st.button("Calculate", type="primary"):
        if var1 == var2:
            st.error("Please select two different variables.")
            return

        result = chi_squared_test(df, var1, var2, alpha=alpha)

        tab_res, tab_assume, tab_chart = st.tabs(["Results", "Assumptions", "Charts"])

        with tab_res:
            render_significance_result(
                result["test"], "χ²", result["chi2"], result["p"], result["df"], alpha
            )
            st.markdown("---")

            st.markdown("### Observed Frequencies")
            st.dataframe(result["contingency"], width="stretch")

            st.markdown("### Expected Frequencies")
            st.dataframe(
                result["expected"].round(2), width="stretch"
            )

            cols = st.columns(3)
            cols[0].metric("N", result["n"])
            cols[1].metric("Cramér's V", f"{result['cramers_v']:.4f}")
            cols[2].metric("df", result["df"])

        with tab_assume:
            ef = result["assumptions"]["expected_frequencies"]
            render_assumption_check(
                "Expected Frequencies ≥ 5", None, None,
                ef["passed"], ef["detail"]
            )

        with tab_chart:
            # Stacked bar chart
            contingency = result["contingency"]
            fig = px.bar(
                contingency.T,
                barmode="group",
                color_discrete_sequence=get_chart_colors(),
                title=f"Frequencies: {var1} × {var2}",
            )
            fig.update_layout(xaxis_title=var2, yaxis_title="Count")
            st.plotly_chart(apply_theme(fig), width="stretch")

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _tables = [
            _serialize_df(result["contingency"], "Observed Frequencies"),
            _serialize_df(result["expected"].round(2), "Expected Frequencies"),
        ]
        _log_entry = build_log_entry(
            entry_type="chi_squared",
            title=f"Chi-Squared Test: {var1} x {var2}",
            result=result,
            tables=_tables,
            variables={"variable_1": var1, "variable_2": var2},
            alpha=alpha,
            dataset_name=st.session_state.get("file_name", ""),
        )
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="chi_pdf_chart")
        if _include_chart:
            _contingency = result["contingency"]
            _cfig = px.bar(
                _contingency.T,
                barmode="group",
                color_discrete_sequence=get_chart_colors(),
                title=f"Frequencies: {var1} x {var2}",
            )
            _cfig.update_layout(xaxis_title=var2, yaxis_title="Count")
            _log_entry["figures"] = [{"label": "Frequency Chart", "fig_dict": apply_theme(_cfig).to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="chi_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="chi_squared.pdf",
                mime="application/pdf",
            )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Purpose
The chi-squared test of independence tests whether there is a **significant association between two categorical variables**. It compares the observed frequencies in a contingency table to the frequencies that would be expected if the variables were independent.

#### When to Use
- Both variables are **nominal** (categorical) -- e.g., gender and preference, treatment group and outcome, region and product choice.
- You want to determine whether knowing the value of one variable helps predict the other.

#### Results Tab
- **Chi-squared statistic** -- measures the overall discrepancy between observed and expected frequencies.
- **p-value** -- probability of observing this discrepancy (or more extreme) if the variables are truly independent.
- **Degrees of freedom** -- (rows - 1) x (columns - 1).
- **Observed frequency table** -- the actual counts in each combination of categories.
- **Expected frequency table** -- the counts expected under the null hypothesis of independence.
- **N** -- total number of observations.
- **Cramer's V** -- effect size measuring the strength of association. Ranges from **0** (no association) to **1** (perfect association).

#### Assumptions Tab
- **Expected frequencies should be >= 5** in all cells. If this assumption is violated (many cells with expected counts below 5), the chi-squared approximation may be unreliable. Consider Fisher's exact test as an alternative.

#### Charts Tab
- **Grouped bar chart** -- displays observed frequencies broken down by both variables, making it easy to visually compare the distribution of categories.
        """)


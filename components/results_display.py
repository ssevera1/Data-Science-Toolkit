"""Formatted results tables, p-value highlighting, and result cards."""

import html as html_lib
import streamlit as st
import pandas as pd
from core.constants import ALPHA
from utils.theme import get_colors


def format_p_value(p):
    """Format a p-value with significance indicator."""
    if p is None or pd.isna(p):
        return "N/A"
    if p < 0.001:
        return "< .001 ***"
    elif p < 0.01:
        return f"{p:.3f} **"
    elif p < 0.05:
        return f"{p:.3f} *"
    elif p < 0.10:
        return f"{p:.3f} †"
    else:
        return f"{p:.3f}"


def p_value_color(p):
    """Return CSS class for p-value coloring."""
    if p is None or pd.isna(p):
        return ""
    if p < ALPHA:
        return "p-significant"
    elif p < 0.10:
        return "p-marginal"
    return "p-not-significant"


def render_results_table(df, title=None):
    """Render a styled DataFrame as a results table."""
    if title:
        st.markdown(f"**{title}**")

    # Format numeric columns
    styled = df.copy()
    for col in styled.columns:
        if styled[col].dtype in ["float64", "float32"]:
            styled[col] = styled[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )

    st.dataframe(styled, use_container_width=True, hide_index=False)


def render_stat_card(label, value, description=None):
    """Render a statistic as a metric card."""
    if isinstance(value, float):
        value = f"{value:.4f}"
    st.metric(label=label, value=value, help=description)


def render_significance_result(test_name, statistic, stat_value, p_value, df_val=None, alpha=ALPHA):
    """Render main test result with significance interpretation."""
    significant = p_value < alpha if p_value is not None else None

    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown(f"### {test_name}")

        parts = [f"**{statistic}** = {stat_value:.4f}"]
        if df_val is not None:
            if isinstance(df_val, tuple):
                parts.append(f"df = ({df_val[0]}, {df_val[1]})")
            else:
                parts.append(f"df = {df_val}")
        parts.append(f"**p** = {format_p_value(p_value)}")

        st.markdown(" &nbsp;|&nbsp; ".join(parts))

    with cols[1]:
        if significant is not None:
            if significant:
                st.success(f"✓ Significant (p < {alpha})")
            else:
                st.error(f"✗ Not significant (p ≥ {alpha})")


def render_assumption_check(name, test_stat, p_value, passed, detail=None):
    """Render an assumption check result."""
    icon = "✓" if passed else "✗"
    css_class = "assumption-pass" if passed else "assumption-fail"
    status = "Passed" if passed else "Violated"

    safe_name = html_lib.escape(str(name))
    safe_detail = html_lib.escape(str(detail)) if detail else None

    markup = f"""
    <div style="margin-bottom: 0.5rem;">
        <span class="{css_class}">{icon} {status}</span>
        <strong style="margin-left: 0.5rem;">{safe_name}</strong>
    """
    if test_stat is not None:
        markup += f" — Statistic = {test_stat:.4f}"
    if p_value is not None:
        markup += f", p = {format_p_value(p_value)}"
    if safe_detail:
        muted = get_colors()["text_muted"]
        markup += f"<br><small style='color:{muted}; margin-left: 2rem;'>{safe_detail}</small>"
    markup += "</div>"

    st.markdown(markup, unsafe_allow_html=True)


def render_effect_size(name, value, interpretation=None):
    """Render an effect size result."""
    text = f"**{name}:** {value:.4f}"
    if interpretation:
        text += f" ({interpretation})"
    st.markdown(text)


def interpret_cohens_d(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    return "large"


def interpret_eta_squared(eta2):
    """Interpret eta-squared magnitude."""
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    return "large"


def interpret_r(r):
    """Interpret correlation coefficient magnitude."""
    r = abs(r)
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    return "large"

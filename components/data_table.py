"""Compact data preview component for analysis pages."""

import streamlit as st
import pandas as pd
from core.state import get_df, get_var_type


def render_data_preview(max_rows=8):
    """Show a compact data preview with variable types."""
    df = get_df()
    clean = df.dropna(how="all")

    if clean.empty:
        st.warning("No data available. Please enter data on the Data Input page.")
        return False

    with st.expander(f"Data Preview ({len(clean)} rows, {len(clean.columns)} variables)", expanded=False):
        # Show variable types
        type_row = {col: get_var_type(col) for col in clean.columns}
        type_df = pd.DataFrame([type_row])
        type_df.index = ["Type"]
        st.dataframe(type_df, width="stretch", height=60)

        # Show data preview
        st.dataframe(clean.head(max_rows), width="stretch", hide_index=True)

    return True

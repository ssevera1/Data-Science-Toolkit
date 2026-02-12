"""Main data input page - default landing page."""

import streamlit as st
import pandas as pd
import numpy as np
from core.state import get_df, set_df, get_var_type, set_var_type, init_state
from core.data_manager import load_csv, load_excel, load_from_paste, add_column, add_rows, export_csv, export_excel
from core.constants import VARIABLE_TYPES


def render():
    st.title("Data Input")
    st.markdown("Enter your data manually, upload a file, or paste from a spreadsheet.")

    # --- File Upload / Paste ---
    with st.expander("Import Data", expanded=False):
        tab_upload, tab_paste = st.tabs(["Upload File", "Paste Data"])

        with tab_upload:
            uploaded = st.file_uploader(
                "Upload CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                key="file_uploader",
            )
            if uploaded is not None:
                if uploaded.name.endswith(".csv"):
                    success, err = load_csv(uploaded)
                else:
                    success, err = load_excel(uploaded)
                if success:
                    st.success(f"Loaded {uploaded.name} successfully!")
                    st.rerun()
                else:
                    st.error(f"Error loading file: {err}")

        with tab_paste:
            pasted = st.text_area(
                "Paste data (tab, comma, or semicolon separated)",
                height=150,
                key="paste_area",
            )
            if st.button("Load Pasted Data", key="load_paste"):
                if pasted.strip():
                    success, err = load_from_paste(pasted)
                    if success:
                        st.success("Data loaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"Error: {err}")

    # --- Variable Types ---
    st.subheader("Variable Types")
    df = get_df()
    cols = st.columns(min(len(df.columns), 6))

    for i, col in enumerate(df.columns):
        with cols[i % len(cols)]:
            current_type = get_var_type(col)
            type_idx = VARIABLE_TYPES.index(current_type) if current_type in VARIABLE_TYPES else 0
            new_type = st.selectbox(
                col,
                VARIABLE_TYPES,
                index=type_idx,
                key=f"vartype_{col}",
            )
            if new_type != current_type:
                set_var_type(col, new_type)

    # --- Data Editor ---
    st.subheader("Data Table")

    col_btns = st.columns(4)
    with col_btns[0]:
        if st.button("+ Add Column"):
            add_column()
            st.rerun()
    with col_btns[1]:
        if st.button("+ Add 10 Rows"):
            add_rows(10)
            st.rerun()
    with col_btns[2]:
        csv_data = export_csv()
        st.download_button("Download CSV", csv_data, "data.csv", "text/csv")
    with col_btns[3]:
        excel_data = export_excel()
        st.download_button("Download Excel", excel_data, "data.xlsx")

    # Editable data table
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor",
    )

    # Update state if edited
    if edited_df is not None and not edited_df.equals(df):
        set_df(edited_df)
        # Set types for any new columns
        for col in edited_df.columns:
            if col not in df.columns:
                set_var_type(col, "Metric")

    # --- Summary ---
    clean = df.dropna(how="all")
    st.caption(f"{len(clean)} rows × {len(df.columns)} variables")

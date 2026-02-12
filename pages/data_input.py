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
        width="stretch",
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

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Import Data
- **Upload File** -- upload a **CSV** or **Excel** (.xlsx / .xls) file directly from your computer.
- **Paste Data** -- paste tab-separated, comma-separated, or semicolon-separated text copied from a spreadsheet or other source. Click **Load Pasted Data** to import it.

#### Variable Types
Each column is assigned one of three types. The type you choose determines which statistical tests and charts are available throughout the app.

- **Metric** -- continuous numeric data (e.g., height, weight, temperature). Required by t-tests, ANOVA, correlation, and regression tools.
- **Nominal** -- unordered categorical data (e.g., gender, treatment group, color). Used as grouping variables and in chi-squared / binomial tests.
- **Ordinal** -- ordered categorical data (e.g., Likert scales, education level). Treated as numeric for descriptive statistics and non-parametric tests.

#### Data Table
- An **editable spreadsheet** -- click any cell to modify its value directly.
- **+ Add Column** adds a new empty column; **+ Add 10 Rows** appends blank rows.
- **Download CSV / Download Excel** exports the current data to your computer.
- Rows and columns can also be added or removed through the built-in data editor controls.

#### Shared Data
Data entered or imported on this page is **shared across all Data Science and Statistics tools** in the application. Any changes you make here are immediately available on every other page.
        """)


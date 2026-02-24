import streamlit as st
import pandas as pd
import numpy as np
from utils.theme import page_header
from core.data_manager import sanitize_csv as _sanitize_csv


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


def render():
    page_header("Smart Cleaning", "One-click missing value imputation, outlier treatment, encoding, and deduplication.", "🧹")

    _guard()
    df = st.session_state["df"].copy()

    tab_missing, tab_outliers, tab_encode, tab_dedup = st.tabs(
        ["Missing Values", "Outlier Treatment", "Encoding", "Deduplication"]
    )

    # ── Missing Values ─────────────────────────────────────────────────────────
    with tab_missing:
        st.subheader("Handle Missing Values")
        miss_cols = df.columns[df.isnull().any()].tolist()
        if not miss_cols:
            st.success("No missing values to handle!")
        else:
            st.info(f"**{len(miss_cols)}** columns have missing values.")

            strategy = st.selectbox(
                "Global strategy",
                ["Per-column (configure below)", "Drop rows with any null",
                 "Drop columns > 50% missing"],
            )

            if strategy == "Drop rows with any null":
                if st.button("Apply: Drop rows"):
                    before = len(df)
                    df = df.dropna()
                    st.session_state["df"] = df
                    st.success(f"Dropped {before - len(df):,} rows. {len(df):,} remaining.")
                    st.rerun()

            elif strategy == "Drop columns > 50% missing":
                threshold = st.slider("Missing threshold %", 10, 90, 50, 5)
                drop_cols = [c for c in miss_cols if df[c].isnull().mean() * 100 > threshold]
                st.write(f"Columns to drop: {drop_cols}")
                if drop_cols and st.button("Apply: Drop columns"):
                    df = df.drop(columns=drop_cols)
                    st.session_state["df"] = df
                    st.success(f"Dropped {len(drop_cols)} columns.")
                    st.rerun()

            else:  # Per-column
                actions = {}
                for col in miss_cols:
                    pct = df[col].isnull().mean() * 100
                    col_type = str(df[col].dtype)
                    c1, c2, c3 = st.columns([2, 2, 1])
                    with c1:
                        st.write(f"**{col}** ({col_type}) — {pct:.1f}% missing")
                    with c2:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            opts = ["Skip", "Mean", "Median", "Mode", "Zero", "Forward fill"]
                        else:
                            opts = ["Skip", "Mode", "Constant (Unknown)", "Forward fill"]
                        actions[col] = st.selectbox(f"Strategy for {col}", opts, key=f"miss_{col}")

                if st.button("Apply Missing Value Fixes"):
                    for col, action in actions.items():
                        if action == "Mean":
                            df[col] = df[col].fillna(df[col].mean())
                        elif action == "Median":
                            df[col] = df[col].fillna(df[col].median())
                        elif action == "Mode":
                            mode_val = df[col].mode()
                            if len(mode_val) > 0:
                                df[col] = df[col].fillna(mode_val.iloc[0])
                        elif action == "Zero":
                            df[col] = df[col].fillna(0)
                        elif action == "Constant (Unknown)":
                            df[col] = df[col].fillna("Unknown")
                        elif action == "Forward fill":
                            df[col] = df[col].ffill()
                    st.session_state["df"] = df
                    remaining = df.isnull().sum().sum()
                    st.success(f"Done! {remaining} missing values remain.")
                    st.rerun()

    # ── Outlier Treatment ──────────────────────────────────────────────────────
    with tab_outliers:
        st.subheader("Outlier Treatment")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            st.info("No numeric columns.")
        else:
            sel_cols = st.multiselect("Columns to treat", num_cols, default=[])
            method = st.selectbox("Method", ["IQR Capping", "Z-Score Capping", "Remove Rows"])
            iqr_mult = st.slider("IQR multiplier / Z threshold", 1.0, 4.0, 1.5, 0.1)

            if sel_cols and st.button("Apply Outlier Treatment"):
                for col in sel_cols:
                    if method == "IQR Capping":
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
                        df[col] = df[col].clip(lower, upper)
                    elif method == "Z-Score Capping":
                        mean, std = df[col].mean(), df[col].std()
                        lower, upper = mean - iqr_mult * std, mean + iqr_mult * std
                        df[col] = df[col].clip(lower, upper)
                    elif method == "Remove Rows":
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
                        df = df[(df[col] >= lower) & (df[col] <= upper)]

                st.session_state["df"] = df
                st.success("Outlier treatment applied.")
                st.rerun()

    # ── Encoding ───────────────────────────────────────────────────────────────
    with tab_encode:
        st.subheader("Categorical Encoding")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not cat_cols:
            st.success("No categorical columns to encode.")
        else:
            enc_method = st.selectbox("Encoding method", [
                "Label Encoding (ordinal)",
                "One-Hot Encoding",
                "Frequency Encoding",
            ])
            sel_enc = st.multiselect("Columns to encode", cat_cols, default=[])

            if sel_enc:
                for col in sel_enc:
                    st.write(f"**{col}** — {df[col].nunique()} unique values")

            if sel_enc and st.button("Apply Encoding"):
                for col in sel_enc:
                    if enc_method == "Label Encoding (ordinal)":
                        df[col] = df[col].astype("category").cat.codes
                    elif enc_method == "One-Hot Encoding":
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    elif enc_method == "Frequency Encoding":
                        freq = df[col].value_counts(normalize=True)
                        df[col] = df[col].map(freq)

                st.session_state["df"] = df
                st.success(f"Encoding applied. Shape: {df.shape}")
                st.rerun()

    # ── Deduplication ──────────────────────────────────────────────────────────
    with tab_dedup:
        st.subheader("Remove Duplicates")
        n_dups = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{n_dups:,}")

        if n_dups > 0:
            st.write("Preview of duplicates:")
            st.dataframe(df[df.duplicated(keep=False)].head(20), width="stretch")
            keep = st.selectbox("Keep", ["first", "last"])
            if st.button("Remove Duplicates"):
                df = df.drop_duplicates(keep=keep)
                st.session_state["df"] = df
                st.success(f"Removed {n_dups:,} duplicates. {len(df):,} rows remaining.")
                st.rerun()
        else:
            st.success("No duplicate rows found.")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Smart Cleaning — One-Click Data Preparation

This page provides four tabs for common data cleaning tasks. All changes update the shared dataset used across the app.

---

#### Missing Values Tab
- **Global Strategies:**
  - **Drop rows with any null** — removes every row that has at least one missing value. Simple but can lose significant data if missingness is widespread.
  - **Drop columns > threshold % missing** — removes entire columns where the percentage of missing values exceeds a configurable threshold (default 50%).
- **Per-Column Strategies** — configure a different fill strategy for each column:
  - **Mean** — fills with the column's arithmetic mean (numeric only). Good for normally distributed data.
  - **Median** — fills with the column's median. More robust to outliers than mean.
  - **Mode** — fills with the most frequent value. Works for both numeric and categorical columns.
  - **Zero** — fills with 0 (numeric only). Appropriate when zero is a meaningful default.
  - **Constant (Unknown)** — fills categorical columns with the string "Unknown".
  - **Forward Fill** — propagates the last valid value forward. Useful for time-series or ordered data.

#### Outlier Treatment Tab
- Select one or more numeric columns and choose a treatment method:
  - **IQR Capping** — clips values to [Q1 - k*IQR, Q3 + k*IQR]. Outliers are replaced with the boundary values rather than removed.
  - **Z-Score Capping** — clips values to [mean - k*std, mean + k*std]. Uses standard deviations instead of quartiles.
  - **Remove Rows** — deletes rows where the value falls outside the IQR boundaries.
- The **multiplier/threshold (k)** is adjustable from 1.0 to 4.0 (default 1.5).

#### Encoding Tab
- Converts categorical (text) columns into numeric representations for machine learning:
  - **Label Encoding** — assigns each unique category an ordinal integer (0, 1, 2, ...). Suitable for ordinal data but can imply false ordering for nominal data.
  - **One-Hot Encoding** — creates binary (0/1) dummy columns for each category (drops the first to avoid multicollinearity). Best for nominal data with few unique values.
  - **Frequency Encoding** — replaces each category with its proportion (frequency) in the dataset. Preserves information about category prevalence.

#### Deduplication Tab
- Detects and displays **exact duplicate rows** in the dataset.
- Choose which duplicate to **keep** — the **first** or **last** occurrence — and remove the rest.

#### Download Section
- After cleaning, download the updated dataset as a **CSV file**. The download applies CSV-injection sanitization to protect against formula-based attacks in spreadsheet applications.
        """)

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Download Cleaned Data")

    csv = _sanitize_csv(df).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "cleaned_data.csv", "text/csv")

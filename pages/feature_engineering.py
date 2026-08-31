import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from math import comb
from utils.theme import page_header
from core.data_manager import sanitize_csv as _sanitize_csv


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


def render():
    page_header("Feature Engineering", "Auto-generate polynomial, interaction, datetime, and binned features in seconds.", "🔧")

    _guard()
    df = st.session_state["df"].copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()

    tab_poly, tab_interact, tab_datetime, tab_bin, tab_math = st.tabs(
        ["Polynomial", "Interactions", "Datetime", "Binning", "Math Transforms"]
    )

    # ── Polynomial Features ───────────────────────────────────────────────────
    with tab_poly:
        st.subheader("Polynomial Features")
        if not num_cols:
            st.info("No numeric columns.")
        else:
            sel = st.multiselect("Select columns", num_cols, default=[], key="poly_cols")
            degree = st.slider("Degree", 2, 4, 2)
            include_bias = st.checkbox("Include bias (constant 1)", False)

            if sel and st.button("Generate Polynomial Features"):
                n_output = comb(len(sel) + degree, degree)
                if n_output > 1000:
                    st.error(
                        f"This would generate ~{n_output:,} features. "
                        "Please reduce the number of columns or the degree (max 1,000 features)."
                    )
                    st.stop()
                from sklearn.preprocessing import PolynomialFeatures
                pf = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=False)
                transformed = pf.fit_transform(df[sel].fillna(0))
                names = pf.get_feature_names_out(sel)
                # Only add new features (exclude originals)
                new_feats = pd.DataFrame(transformed, columns=names, index=df.index)
                new_only = new_feats[[c for c in new_feats.columns if c not in df.columns]]
                if len(new_only.columns) > 0:
                    df = pd.concat([df, new_only], axis=1)
                st.session_state["df"] = df
                added = len(new_feats.columns) - len(sel) - (1 if include_bias else 0)
                st.success(f"Added {added} polynomial features. New shape: {df.shape}")
                st.dataframe(df.head(), width="stretch")

    # ── Interaction Features ──────────────────────────────────────────────────
    with tab_interact:
        st.subheader("Interaction Features")
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            sel = st.multiselect("Select columns", num_cols, default=[], key="interact_cols")
            ops = st.multiselect("Operations", ["Multiply", "Divide", "Add", "Subtract"],
                                 default=["Multiply"])

            if len(sel) >= 2 and st.button("Generate Interactions"):
                n_output = comb(len(sel), 2) * len(ops)
                if n_output > 1000:
                    st.error(
                        f"This would generate {n_output:,} features. "
                        "Please reduce the number of columns or operations (max 1,000 features)."
                    )
                    st.stop()
                new_cols = {}
                for a, b in combinations(sel, 2):
                    if "Multiply" in ops:
                        new_cols[f"{a}_x_{b}"] = df[a] * df[b]
                    if "Divide" in ops:
                        new_cols[f"{a}_div_{b}"] = df[a] / df[b].replace(0, np.nan)
                    if "Add" in ops:
                        new_cols[f"{a}_plus_{b}"] = df[a] + df[b]
                    if "Subtract" in ops:
                        new_cols[f"{a}_minus_{b}"] = df[a] - df[b]
                new_count = len(new_cols)
                if new_cols:
                    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
                st.session_state["df"] = df
                st.success(f"Added {new_count} interaction features. New shape: {df.shape}")
                st.dataframe(df.head(), width="stretch")

    # ── Datetime Features ─────────────────────────────────────────────────────
    with tab_datetime:
        st.subheader("Datetime Feature Extraction")
        # Try to detect datetime columns
        potential_dt = []
        for col in all_cols:
            if df[col].dtype == "datetime64[ns]":
                potential_dt.append(col)
            elif df[col].dtype == object:
                sample = df[col].dropna().head(20)
                try:
                    pd.to_datetime(sample, format="mixed")
                    potential_dt.append(col)
                except (ValueError, TypeError):
                    pass

        if not potential_dt:
            st.info("No datetime columns detected. Select a column to try parsing it.")
            manual_col = st.selectbox("Try parsing column as datetime", all_cols)
            if st.button("Parse as Datetime"):
                try:
                    df[manual_col] = pd.to_datetime(df[manual_col], format="mixed")
                    st.session_state["df"] = df
                    st.success(f"Parsed {manual_col} as datetime.")
                    st.rerun()
                except Exception:
                    st.error("Could not parse column as datetime. Please check the column format.")
        else:
            sel = st.multiselect("Datetime columns", potential_dt, default=potential_dt[:1])
            features = st.multiselect(
                "Extract",
                ["year", "month", "day", "dayofweek", "hour", "minute", "quarter",
                 "is_weekend", "day_of_year", "week_of_year"],
                default=["year", "month", "day", "dayofweek", "is_weekend"],
            )

            if sel and features and st.button("Extract Datetime Features"):
                for col in sel:
                    dt = pd.to_datetime(df[col], errors="coerce", format="mixed")
                    for feat in features:
                        name = f"{col}_{feat}"
                        if feat == "year":
                            df[name] = dt.dt.year
                        elif feat == "month":
                            df[name] = dt.dt.month
                        elif feat == "day":
                            df[name] = dt.dt.day
                        elif feat == "dayofweek":
                            df[name] = dt.dt.dayofweek
                        elif feat == "hour":
                            df[name] = dt.dt.hour
                        elif feat == "minute":
                            df[name] = dt.dt.minute
                        elif feat == "quarter":
                            df[name] = dt.dt.quarter
                        elif feat == "is_weekend":
                            df[name] = (dt.dt.dayofweek >= 5).astype(int)
                        elif feat == "day_of_year":
                            df[name] = dt.dt.dayofyear
                        elif feat == "week_of_year":
                            week = dt.dt.isocalendar().week
                            df[name] = pd.to_numeric(week, errors="coerce").astype("Int64")
                st.session_state["df"] = df
                st.success(f"Extracted {len(sel) * len(features)} datetime features.")
                st.dataframe(df.head(), width="stretch")

    # ── Binning ────────────────────────────────────────────────────────────────
    with tab_bin:
        st.subheader("Numeric Binning")
        if not num_cols:
            st.info("No numeric columns.")
        else:
            sel = st.multiselect("Columns to bin", num_cols, default=[], key="bin_cols")
            n_bins = st.slider("Number of bins", 2, 20, 5)
            bin_method = st.selectbox("Method", ["Equal Width (cut)", "Equal Frequency (qcut)"])

            if sel and st.button("Apply Binning"):
                for col in sel:
                    name = f"{col}_binned"
                    if bin_method == "Equal Width (cut)":
                        df[name] = pd.cut(df[col], bins=n_bins, labels=False)
                    else:
                        df[name] = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
                st.session_state["df"] = df
                st.success(f"Added {len(sel)} binned features.")
                st.dataframe(df.head(), width="stretch")

    # ── Math Transforms ───────────────────────────────────────────────────────
    with tab_math:
        st.subheader("Mathematical Transforms")
        if not num_cols:
            st.info("No numeric columns.")
        else:
            sel = st.multiselect("Columns", num_cols, default=[], key="math_cols")
            transforms = st.multiselect(
                "Transforms",
                ["Log (log1p)", "Square Root", "Reciprocal", "Square", "Standard Scale", "Min-Max Scale"],
                default=["Log (log1p)"],
            )

            if sel and st.button("Apply Transforms"):
                for col in sel:
                    for t in transforms:
                        if t == "Log (log1p)":
                            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
                        elif t == "Square Root":
                            df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))
                        elif t == "Reciprocal":
                            df[f"{col}_recip"] = 1 / df[col].replace(0, np.nan)
                        elif t == "Square":
                            df[f"{col}_sq"] = df[col] ** 2
                        elif t == "Standard Scale":
                            mean, std = df[col].mean(), df[col].std()
                            df[f"{col}_scaled"] = (df[col] - mean) / std if std > 0 else 0
                        elif t == "Min-Max Scale":
                            mn, mx = df[col].min(), df[col].max()
                            df[f"{col}_minmax"] = (df[col] - mn) / (mx - mn) if mx > mn else 0
                st.session_state["df"] = df
                st.success(f"Applied {len(transforms)} transforms to {len(sel)} columns.")
                st.dataframe(df.head(), width="stretch")

    # ── AI Interpretation ──────────────────────────────────────────────────
    from components.ai_advisor import render_ai_interpretation
    ai_texts = render_ai_interpretation(
        entry_type="feature_engineering",
        result={
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "n_numeric": len(num_cols),
            "n_categorical": len(cat_cols),
        },
        variables={},
        page_key="feng",
    )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Feature Engineering — Automated Feature Generation

This page lets you create new features from your existing data across five tabs. All new features are added to the shared dataset.

---

#### Polynomial Features Tab
- Generates **polynomial and interaction terms** using scikit-learn's `PolynomialFeatures`.
- Select numeric columns and a **degree** (2 to 4):
  - Degree 2 produces squared terms (e.g., `x1^2`, `x1*x2`).
  - Degree 3 adds cubed terms, and so on.
- **Include bias** option adds a constant column of 1s (rarely needed if your model has an intercept).
- A safety cap of **1,000 features** prevents accidental memory issues.

#### Interactions Tab
- Creates **pairwise interaction features** between selected numeric columns.
- Four operations available:
  - **Multiply** (`x1 * x2`) — captures multiplicative relationships.
  - **Divide** (`x1 / x2`) — captures ratio relationships. Division by zero produces NaN.
  - **Add** (`x1 + x2`) — captures additive effects.
  - **Subtract** (`x1 - x2`) — captures difference effects.
- Also capped at 1,000 features for safety.

#### Datetime Tab
- **Auto-detects** datetime columns (including string columns that can be parsed as dates).
- Extracts a variety of time-based features:
  - **year, month, day** — calendar components.
  - **dayofweek** — 0 (Monday) through 6 (Sunday).
  - **hour, minute** — time-of-day components.
  - **quarter** — fiscal quarter (1-4).
  - **is_weekend** — binary flag (1 if Saturday/Sunday, 0 otherwise).
  - **day_of_year, week_of_year** — position within the year.
- If no datetime column is detected, you can manually select a column and attempt to parse it.

#### Binning Tab
- Converts continuous numeric columns into **discrete bins** (buckets):
  - **Equal Width (cut)** — divides the value range into N bins of equal width. Good for uniformly distributed data.
  - **Equal Frequency (qcut)** — divides data so each bin has approximately the same number of observations. Better for skewed distributions.
- Configurable **number of bins** from 2 to 20.
- Creates a new column with a `_binned` suffix containing integer bin labels.

#### Math Transforms Tab
- Applies common mathematical transformations to numeric columns:
  - **Log (log1p)** — `log(1 + x)`. Compresses right-skewed distributions. Negative values are clipped to 0.
  - **Square Root** — `sqrt(x)`. Milder compression than log. Negative values are clipped to 0.
  - **Reciprocal** — `1 / x`. Useful for inverse relationships. Zero values produce NaN.
  - **Square** — `x^2`. Emphasizes large values and captures quadratic effects.
  - **Standard Scale** — `(x - mean) / std`. Centers data at 0 with unit variance (z-score normalization).
  - **Min-Max Scale** — `(x - min) / (max - min)`. Scales data to the [0, 1] range.

#### Download Section
- Download the dataset with all newly engineered features as a CSV file.
        """)

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Download Engineered Data")

    csv = _sanitize_csv(df).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "engineered_data.csv", "text/csv")

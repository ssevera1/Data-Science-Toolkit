import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from math import comb
from utils.theme import page_header


def _guard():
    if "df" not in st.session_state:
        st.warning("Upload a dataset on the **Home** page first.")
        st.stop()


def _sanitize_csv(dataframe):
    """Prefix cells that start with formula-trigger characters to prevent CSV injection."""
    _dangerous = ("=", "+", "-", "@", "\t", "\r")
    out = dataframe.copy()
    for col in out.select_dtypes(include=["object", "category"]).columns:
        out[col] = out[col].apply(
            lambda v: "'" + v if isinstance(v, str) and v and v[0] in _dangerous else v
        )
    return out


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
                for c in new_feats.columns:
                    if c not in df.columns:
                        df[c] = new_feats[c]
                st.session_state["df"] = df
                added = len(new_feats.columns) - len(sel) - (1 if include_bias else 0)
                st.success(f"Added {added} polynomial features. New shape: {df.shape}")
                st.dataframe(df.head(), use_container_width=True)

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
                new_count = 0
                for a, b in combinations(sel, 2):
                    if "Multiply" in ops:
                        name = f"{a}_x_{b}"
                        df[name] = df[a] * df[b]
                        new_count += 1
                    if "Divide" in ops:
                        name = f"{a}_div_{b}"
                        df[name] = df[a] / df[b].replace(0, np.nan)
                        new_count += 1
                    if "Add" in ops:
                        name = f"{a}_plus_{b}"
                        df[name] = df[a] + df[b]
                        new_count += 1
                    if "Subtract" in ops:
                        name = f"{a}_minus_{b}"
                        df[name] = df[a] - df[b]
                        new_count += 1
                st.session_state["df"] = df
                st.success(f"Added {new_count} interaction features. New shape: {df.shape}")
                st.dataframe(df.head(), use_container_width=True)

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
                    pd.to_datetime(sample)
                    potential_dt.append(col)
                except (ValueError, TypeError):
                    pass

        if not potential_dt:
            st.info("No datetime columns detected. Select a column to try parsing it.")
            manual_col = st.selectbox("Try parsing column as datetime", all_cols)
            if st.button("Parse as Datetime"):
                try:
                    df[manual_col] = pd.to_datetime(df[manual_col])
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
                    dt = pd.to_datetime(df[col], errors="coerce")
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
                            df[name] = dt.dt.isocalendar().week.astype(int)
                st.session_state["df"] = df
                st.success(f"Extracted {len(sel) * len(features)} datetime features.")
                st.dataframe(df.head(), use_container_width=True)

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
                st.dataframe(df.head(), use_container_width=True)

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
                st.dataframe(df.head(), use_container_width=True)

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Download Engineered Data")

    csv = _sanitize_csv(df).to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "engineered_data.csv", "text/csv")

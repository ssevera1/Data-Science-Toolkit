import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.theme import page_header


def _guard():
    if "df" not in st.session_state:
        st.warning("Upload a dataset on the **Home** page first.")
        st.stop()


def render():
    page_header("Class Imbalance Handler", "Detect class skew and fix it with SMOTE, random oversampling, or undersampling.", "⚖️")

    _guard()
    df = st.session_state["df"].copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    # ── Target Selection ───────────────────────────────────────────────────────
    st.subheader("1. Select Target Column")
    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox("Target (class) column", all_cols)
    with c2:
        n_classes = df[target].nunique()
        is_binary = n_classes == 2
        label = f"**Classification** — {'binary' if is_binary else f'{n_classes} classes'}"
        st.write(label)

    if n_classes > 50:
        st.warning(f"Target has {n_classes} unique values — this tool is designed for classification targets (< 50 classes).")
        st.stop()

    # ── Imbalance Analysis ────────────────────────────────────────────────────
    st.subheader("2. Imbalance Analysis")
    vc = df[target].value_counts()
    vc_df = vc.reset_index()
    vc_df.columns = ["Class", "Count"]
    vc_df["Percentage"] = (vc_df["Count"] / len(df) * 100).round(2)

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(vc_df, use_container_width=True, hide_index=True)
        imbalance_ratio = vc.max() / vc.min()
        st.metric("Imbalance Ratio (max/min)", f"{imbalance_ratio:.1f}x")
        if imbalance_ratio > 3:
            st.error(f"Significant imbalance detected ({imbalance_ratio:.1f}x).")
        elif imbalance_ratio > 1.5:
            st.warning(f"Moderate imbalance ({imbalance_ratio:.1f}x).")
        else:
            st.success(f"Classes are relatively balanced ({imbalance_ratio:.1f}x).")

    with c2:
        fig = px.pie(vc_df, names="Class", values="Count", title="Class Distribution",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── Resampling ─────────────────────────────────────────────────────────────
    st.subheader("3. Apply Resampling")

    feature_cols = [c for c in num_cols if c != target]
    if not feature_cols:
        st.warning("Need numeric feature columns for resampling. Encode categorical features first (Smart Cleaning page).")
        st.stop()

    method = st.selectbox("Resampling method", [
        "SMOTE (Synthetic Minority Oversampling)",
        "Random Oversampling",
        "Random Undersampling",
        "SMOTE + Tomek Links (Combined)",
    ])

    X = df[feature_cols].fillna(0)
    y = df[target]

    if st.button("Apply Resampling"):
        with st.spinner("Resampling..."):
            try:
                if method.startswith("SMOTE ("):
                    from imblearn.over_sampling import SMOTE
                    min_class_count = y.value_counts().min()
                    k = min(5, min_class_count - 1) if min_class_count > 1 else 1
                    sampler = SMOTE(random_state=42, k_neighbors=k)
                elif method == "Random Oversampling":
                    from imblearn.over_sampling import RandomOverSampler
                    sampler = RandomOverSampler(random_state=42)
                elif method == "Random Undersampling":
                    from imblearn.under_sampling import RandomUnderSampler
                    sampler = RandomUnderSampler(random_state=42)
                else:  # SMOTE + Tomek
                    from imblearn.combine import SMOTETomek
                    min_class_count = y.value_counts().min()
                    k = min(5, min_class_count - 1) if min_class_count > 1 else 1
                    from imblearn.over_sampling import SMOTE as SMOTE2
                    sampler = SMOTETomek(random_state=42, smote=SMOTE2(k_neighbors=k))

                X_res, y_res = sampler.fit_resample(X, y)

                new_df = pd.DataFrame(X_res, columns=feature_cols)
                new_df[target] = y_res

                # Show comparison
                st.markdown("#### Before vs After")
                c1, c2 = st.columns(2)

                with c1:
                    st.write("**Before**")
                    before_vc = vc_df.copy()
                    st.dataframe(before_vc, use_container_width=True, hide_index=True)
                    st.write(f"Total: {len(df):,}")

                with c2:
                    st.write("**After**")
                    after_vc = y_res.value_counts().reset_index()
                    after_vc.columns = ["Class", "Count"]
                    after_vc["Percentage"] = (after_vc["Count"] / len(y_res) * 100).round(2)
                    st.dataframe(after_vc, use_container_width=True, hide_index=True)
                    st.write(f"Total: {len(new_df):,}")

                # Comparison chart
                compare = pd.DataFrame({
                    "Class": list(vc.index) + list(y_res.value_counts().index),
                    "Count": list(vc.values) + list(y_res.value_counts().values),
                    "Stage": ["Before"] * len(vc) + ["After"] * len(y_res.value_counts()),
                })
                fig = px.bar(compare, x="Class", y="Count", color="Stage", barmode="group",
                             color_discrete_map={"Before": "#e74c3c", "After": "#2ecc71"})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                if st.button("Accept & Update Dataset"):
                    st.session_state["df"] = new_df
                    st.success("Dataset updated with resampled data.")
                    st.rerun()

            except Exception:
                st.error("Resampling failed. Please check that the target column and features are suitable for the selected method.")

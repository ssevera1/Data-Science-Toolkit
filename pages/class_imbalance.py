import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.theme import page_header, get_colors


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
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
        st.dataframe(vc_df, width="stretch", hide_index=True)
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
        st.plotly_chart(fig, width="stretch")

    # ── Resampling ─────────────────────────────────────────────────────────────
    st.subheader("3. Apply Resampling")

    feature_cols = [c for c in num_cols if c != target]
    cat_cols = [c for c in all_cols if c not in num_cols and c != target]
    all_feature_cols = [c for c in all_cols if c != target]

    if not feature_cols:
        st.warning("Need numeric feature columns for resampling. Encode categorical features first (Smart Cleaning page).")
        st.stop()

    method = st.selectbox("Resampling method", [
        "SMOTE (Synthetic Minority Oversampling)",
        "Random Oversampling",
        "Random Undersampling",
        "SMOTE + Tomek Links (Combined)",
    ])

    y = df[target]
    uses_smote = method.startswith("SMOTE")

    if st.button("Apply Resampling"):
        with st.spinner("Resampling..."):
            try:
                if uses_smote:
                    # SMOTE requires numeric input — encode categoricals temporarily
                    X = df[feature_cols].fillna(0)
                    if method.startswith("SMOTE ("):
                        from imblearn.over_sampling import SMOTE
                        min_class_count = y.value_counts().min()
                        k = min(5, min_class_count - 1) if min_class_count > 1 else 1
                        sampler = SMOTE(random_state=42, k_neighbors=k)
                    else:  # SMOTE + Tomek
                        from imblearn.combine import SMOTETomek
                        min_class_count = y.value_counts().min()
                        k = min(5, min_class_count - 1) if min_class_count > 1 else 1
                        from imblearn.over_sampling import SMOTE as SMOTE2
                        sampler = SMOTETomek(random_state=42, smote=SMOTE2(k_neighbors=k))

                    X_arr = X.values
                    X_res, y_res = sampler.fit_resample(X, y)
                    X_res_arr = X_res.values if isinstance(X_res, pd.DataFrame) else X_res

                    # Build result columns dict, then create DataFrame once
                    result = {col: X_res_arr[:, i] for i, col in enumerate(feature_cols)}
                    result[target] = y_res.values if isinstance(y_res, pd.Series) else y_res

                    # For synthetic rows, fill categoricals from the nearest original row
                    if cat_cols:
                        from sklearn.neighbors import NearestNeighbors
                        nn = NearestNeighbors(n_neighbors=1).fit(X_arr)
                        indices = nn.kneighbors(X_res_arr, return_distance=False).ravel()
                        for col in cat_cols:
                            result[col] = df[col].iloc[indices].values

                    new_df = pd.DataFrame(result)
                    # Reorder columns to match original
                    new_df = new_df[[c for c in df.columns if c in new_df.columns]]
                else:
                    # Random over/undersampling — works on all column types
                    X_all = df[all_feature_cols]
                    if method == "Random Oversampling":
                        from imblearn.over_sampling import RandomOverSampler
                        sampler = RandomOverSampler(random_state=42)
                    else:
                        from imblearn.under_sampling import RandomUnderSampler
                        sampler = RandomUnderSampler(random_state=42)

                    X_res, y_res = sampler.fit_resample(X_all, y)
                    result = pd.DataFrame(X_res, columns=all_feature_cols)
                    result[target] = y_res
                    # Reorder columns to match original and defragment
                    new_df = result[[c for c in df.columns if c in result.columns]].copy()

                # Show comparison
                st.markdown("#### Before vs After")
                c1, c2 = st.columns(2)

                with c1:
                    st.write("**Before**")
                    before_vc = vc_df.copy()
                    st.dataframe(before_vc, width="stretch", hide_index=True)
                    st.write(f"Total: {len(df):,}")

                with c2:
                    st.write("**After**")
                    after_vc = y_res.value_counts().reset_index()
                    after_vc.columns = ["Class", "Count"]
                    after_vc["Percentage"] = (after_vc["Count"] / len(y_res) * 100).round(2)
                    st.dataframe(after_vc, width="stretch", hide_index=True)
                    st.write(f"Total: {len(new_df):,}")

                # Comparison chart
                compare = pd.DataFrame({
                    "Class": list(vc.index) + list(y_res.value_counts().index),
                    "Count": list(vc.values) + list(y_res.value_counts().values),
                    "Stage": ["Before"] * len(vc) + ["After"] * len(y_res.value_counts()),
                })
                _ci_c = get_colors()
                fig = px.bar(compare, x="Class", y="Count", color="Stage", barmode="group",
                             color_discrete_map={"Before": _ci_c["error"], "After": _ci_c["success"]})
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")

                dl_col, accept_col = st.columns(2)
                with dl_col:
                    csv_data = new_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Resampled CSV",
                        data=csv_data,
                        file_name="resampled_data.csv",
                        mime="text/csv",
                        width="stretch",
                    )
                with accept_col:
                    if st.button("Accept & Update Dataset", width="stretch"):
                        st.session_state["df"] = new_df
                        st.success("Dataset updated with resampled data.")
                        st.rerun()

            except Exception:
                st.error("Resampling failed. Please check that the target column and features are suitable for the selected method.")

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Class Imbalance Handler — Fix Skewed Class Distributions

This page detects class imbalance in classification targets and provides resampling methods to correct it.

---

#### Target Selection
- Choose the **target (class) column** — the column you want to predict in classification.
- The tool displays the number of unique classes and whether the task is **binary** (2 classes) or **multiclass**.
- Targets with more than 50 unique values are flagged as likely not classification targets.

#### Imbalance Analysis
- **Value Counts Table** — shows each class, its count, and its percentage of the total dataset.
- **Imbalance Ratio** — computed as `max_class_count / min_class_count`. Interpretation:
  - **< 1.5x** — classes are relatively balanced (green).
  - **1.5x - 3x** — moderate imbalance (yellow warning).
  - **> 3x** — significant imbalance (red alert).
- **Pie Chart** — visual breakdown of class proportions.

#### Resampling Methods
- **SMOTE (Synthetic Minority Oversampling Technique)** — generates **synthetic** minority class samples by interpolating between existing minority samples and their k-nearest neighbors. Produces more diverse samples than simple duplication. The `k_neighbors` parameter is automatically adjusted based on the smallest class size.
- **Random Oversampling** — **duplicates** existing minority class samples at random until all classes are balanced. Simple but can lead to overfitting on duplicated samples.
- **Random Undersampling** — **removes** majority class samples at random until all classes are balanced. Fast but discards potentially useful data.
- **SMOTE + Tomek Links** — a **hybrid** approach that first applies SMOTE to oversample the minority class, then removes **Tomek links** (pairs of nearest neighbors from different classes that are close together). This cleans up the decision boundary for better separation.

#### Before vs After Comparison
- After resampling, a side-by-side comparison shows the class counts and percentages before and after.
- A **grouped bar chart** visualizes the change in class distribution.
- Click **"Accept & Update Dataset"** to replace the current dataset with the resampled version.
        """)

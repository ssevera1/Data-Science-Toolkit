import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.theme import page_header, get_colors
from core.data_manager import sanitize_csv as _sanitize_csv
from core.state import log_result
from utils.pdf_export import build_log_entry, generate_single_report, _serialize_df

_CACHE_KEY = "_result_class_imb"


def _guard():
    if "df" not in st.session_state or st.session_state["df"].dropna(how="all").empty:
        st.warning("Upload a dataset on the **Home** page first, or enter data via **Statistics Tools > Data Input**.")
        st.stop()


def _apply_tomek_links(result_df, target_col, feature_cols, n_original):
    """Remove Tomek links, checking only synthetic rows for ~3-4x speedup.

    Instead of checking all n**2 pairs (imblearn default), only checks
    synthetic rows against opposite-class samples for mutual cross-class
    nearest neighbors.  Majority-class members of Tomek links are removed.
    """
    from sklearn.neighbors import NearestNeighbors

    X = result_df[feature_cols].values
    y = result_df[target_col].values

    if len(X) <= n_original:
        return result_df, result_df[target_col]

    y_synth = y[n_original:]
    tomek_drop = set()

    for cls in np.unique(y_synth):
        s_mask = y_synth == cls
        s_global = np.where(s_mask)[0] + n_original
        X_s = X[s_global]

        if len(X_s) == 0:
            continue

        opp_global = np.where(y != cls)[0]
        same_global = np.where(y == cls)[0]

        # Nearest opposite-class neighbor for each synthetic sample
        nn_opp = NearestNeighbors(n_neighbors=1).fit(X[opp_global])
        idx_a = nn_opp.kneighbors(X_s, return_distance=False).ravel()
        matched_opp = opp_global[idx_a]

        # For unique matched opposites, find nearest same-class neighbor
        unique_opp, inv = np.unique(matched_opp, return_inverse=True)
        nn_same = NearestNeighbors(n_neighbors=1).fit(X[same_global])
        idx_b = nn_same.kneighbors(X[unique_opp], return_distance=False).ravel()
        nearest_back = same_global[idx_b]

        # Mutual cross-class nearest neighbors = Tomek link
        is_mutual = nearest_back[inv] == s_global
        tomek_drop.update(matched_opp[is_mutual])

    if tomek_drop:
        keep = np.ones(len(X), dtype=bool)
        keep[list(tomek_drop)] = False
        result_df = result_df[keep].reset_index(drop=True)

    return result_df, result_df[target_col]


def _gower_undersample(df, target, all_feature_cols, random_state=42):
    """Boundary-preserving undersampling using Gower distance.

    Keeps majority samples closest to the minority class boundary,
    measured by average Gower distance to k-nearest minority neighbors.
    Works natively on mixed numeric + categorical features.
    """
    import gower

    y = df[target]
    class_counts = y.value_counts()
    min_count = class_counts.min()
    minority_class = class_counts.idxmin()

    minority_mask = y == minority_class
    minority_df = df.loc[minority_mask, all_feature_cols]

    keep_indices = list(df.index[minority_mask])

    for cls in class_counts.index:
        if cls == minority_class:
            continue
        cls_mask = y == cls
        cls_count = class_counts[cls]

        if cls_count <= min_count:
            keep_indices.extend(df.index[cls_mask])
            continue

        cls_df = df.loc[cls_mask, all_feature_cols]
        cls_indices = df.index[cls_mask]

        # Compute Gower distance: each majority sample vs all minority samples
        dist_matrix = gower.gower_matrix(cls_df, minority_df)

        # Average distance to k-nearest minority neighbors
        k = min(5, len(minority_df))
        if k < len(minority_df):
            # Partial sort for k smallest distances per row
            partitioned = np.partition(dist_matrix, k, axis=1)[:, :k]
            avg_dist = partitioned.mean(axis=1)
        else:
            avg_dist = dist_matrix.mean(axis=1)

        # Keep the min_count samples closest to the boundary
        rng = np.random.RandomState(random_state)
        # Break ties randomly by adding tiny jitter
        jitter = rng.uniform(0, 1e-10, size=len(avg_dist))
        closest_idx = np.argsort(avg_dist + jitter)[:min_count]
        keep_indices.extend(cls_indices[closest_idx])

    result_df = df.loc[keep_indices].reset_index(drop=True)
    return result_df, result_df[target]


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

    if n_classes < 2:
        st.warning("Target column must have at least 2 classes for resampling.")
        st.stop()

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

    method = st.selectbox("Resampling method", [
        "SMOTE (Synthetic Minority Oversampling)",
        "Random Oversampling",
        "Random Undersampling",
        "Gower Distance Undersampling",
        "SMOTE + Tomek Links (Combined)",
    ])

    y = df[target]
    uses_smote = method.startswith("SMOTE")

    if uses_smote and not feature_cols:
        st.warning("SMOTE requires numeric feature columns. Encode categorical features first (Smart Cleaning page), or use Random/Gower undersampling.")
        st.stop()

    # Warn about large datasets with SMOTE methods
    if uses_smote and len(df) > 50_000:
        st.warning(
            f"Dataset has {len(df):,} rows. SMOTE methods build nearest-neighbor "
            f"models that scale poorly on large datasets. Consider using Random "
            f"Over/Undersampling, or subsample your data first."
        )

    # Warn about large datasets with Gower distance
    if method == "Gower Distance Undersampling":
        majority_count = vc.max()
        if majority_count > 50_000:
            st.warning(
                f"Largest class has {majority_count:,} rows. Gower distance computes an "
                f"O(n×m×p) distance matrix that may be slow. Consider Random "
                f"Undersampling for very large datasets."
            )

    if st.button("Apply Resampling"):
        with st.spinner("Resampling..."):
            try:
                if uses_smote:
                    # SMOTE requires numeric input — encode categoricals temporarily
                    X = df[feature_cols].fillna(0)
                    from imblearn.over_sampling import SMOTE
                    min_class_count = y.value_counts().min()
                    k = min(5, min_class_count - 1) if min_class_count > 1 else 1
                    sampler = SMOTE(random_state=42, k_neighbors=k)
                    X_res, y_res = sampler.fit_resample(X, y)
                    X_res_arr = X_res.values if isinstance(X_res, pd.DataFrame) else X_res

                    # Build result DataFrame directly from the array
                    result_df = pd.DataFrame(X_res_arr, columns=feature_cols)
                    result_df[target] = y_res.values if isinstance(y_res, pd.Series) else y_res

                    # Original rows keep their own categoricals — only synthetic
                    # rows need a KNN lookup to inherit categorical values
                    if cat_cols:
                        n_original = len(X)
                        n_resampled = len(X_res_arr)
                        original_cat_vals = {col: df[col].values for col in cat_cols}

                        if n_resampled > n_original:
                            from sklearn.neighbors import NearestNeighbors
                            X_arr = X.values
                            X_synth = X_res_arr[n_original:]
                            y_res_arr = y_res.values if isinstance(y_res, pd.Series) else y_res
                            y_synth = y_res_arr[n_original:]

                            # Fit KNN per class on same-class originals only.
                            # Synthetic samples are interpolations of minority
                            # class points, so categoricals should come from
                            # the nearest original of the SAME class.
                            synth_indices = np.empty(len(X_synth), dtype=int)
                            for cls in np.unique(y_synth):
                                synth_mask = y_synth == cls
                                orig_positions = np.where(y.values == cls)[0]
                                nn = NearestNeighbors(n_neighbors=1).fit(
                                    X_arr[orig_positions]
                                )
                                local_idx = nn.kneighbors(
                                    X_synth[synth_mask], return_distance=False
                                ).ravel()
                                synth_indices[synth_mask] = orig_positions[local_idx]

                            cat_data = {}
                            for col in cat_cols:
                                synth_vals = df[col].values[synth_indices]
                                cat_data[col] = np.concatenate([original_cat_vals[col], synth_vals])
                            result_df = pd.concat([result_df, pd.DataFrame(cat_data)], axis=1)
                        else:
                            cat_data = {col: original_cat_vals[col][:n_resampled] for col in cat_cols}
                            result_df = pd.concat([result_df, pd.DataFrame(cat_data)], axis=1)

                    # Apply optimized Tomek links after categoricals are assigned
                    if not method.startswith("SMOTE ("):
                        result_df, y_res = _apply_tomek_links(
                            result_df, target, feature_cols, len(X)
                        )

                    new_df = result_df[[c for c in df.columns if c in result_df.columns]]
                elif method == "Gower Distance Undersampling":
                    result_df, y_res = _gower_undersample(
                        df, target, all_feature_cols
                    )
                    new_df = result_df[[c for c in df.columns if c in result_df.columns]].copy()

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

                # Build before/after DataFrames for cache
                before_vc = vc_df.copy()

                after_counts = y_res.value_counts()
                after_vc = after_counts.reset_index()
                after_vc.columns = ["Class", "Count"]
                after_vc["Percentage"] = (after_vc["Count"] / len(y_res) * 100).round(2)

                compare = pd.DataFrame({
                    "Class": list(vc.index) + list(after_counts.index),
                    "Count": list(vc.values) + list(after_counts.values),
                    "Stage": ["Before"] * len(vc) + ["After"] * len(after_counts),
                })

                # Store results in session state cache
                st.session_state[_CACHE_KEY] = {
                    "inputs": (target, method),
                    "new_df": new_df,
                    "before_vc": before_vc,
                    "after_vc": after_vc,
                    "compare": compare,
                    "imbalance_ratio": imbalance_ratio,
                    "rows_before": len(df),
                    "rows_after": len(new_df),
                }

            except Exception as e:
                st.error(f"Resampling failed: {e}")

    # ── Cache invalidation ────────────────────────────────────────────────
    cached = st.session_state.get(_CACHE_KEY)
    if cached and cached.get("inputs") != (target, method):
        del st.session_state[_CACHE_KEY]
        cached = None

    # ── Render cached results ─────────────────────────────────────────────
    if cached:
        new_df = cached["new_df"]
        before_vc = cached["before_vc"]
        after_vc = cached["after_vc"]
        compare = cached["compare"]

        st.markdown("#### Before vs After")
        c1, c2 = st.columns(2)

        with c1:
            st.write("**Before**")
            st.dataframe(before_vc, width="stretch", hide_index=True)
            st.write(f"Total: {cached['rows_before']:,}")

        with c2:
            st.write("**After**")
            st.dataframe(after_vc, width="stretch", hide_index=True)
            st.write(f"Total: {cached['rows_after']:,}")

        # Comparison chart
        _ci_c = get_colors()
        fig = px.bar(compare, x="Class", y="Count", color="Stage", barmode="group",
                     color_discrete_map={"Before": _ci_c["error"], "After": _ci_c["success"]})
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

        dl_col, accept_col = st.columns(2)
        with dl_col:
            csv_data = _sanitize_csv(new_df).to_csv(index=False).encode("utf-8")
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
                del st.session_state[_CACHE_KEY]
                st.success("Dataset updated with resampled data.")
                st.rerun()

        # ── AI Interpretation ──────────────────────────────────────────
        from components.ai_advisor import render_ai_interpretation
        ai_texts = render_ai_interpretation(
            entry_type="class_imbalance",
            result={
                "target": target,
                "method": method,
                "imbalance_ratio": cached["imbalance_ratio"],
                "rows_before": cached["rows_before"],
                "rows_after": cached["rows_after"],
            },
            variables={"target": target, "method": method},
            page_key="cimb",
        )

        # ── PDF Export ─────────────────────────────────────────────────
        st.divider()
        _log_entry = build_log_entry(
            entry_type="class_imbalance",
            title=f"Class Imbalance: {target}",
            result={
                "target": target,
                "method": method,
                "imbalance_ratio": cached["imbalance_ratio"],
                "rows_before": cached["rows_before"],
                "rows_after": cached["rows_after"],
            },
            tables=[
                _serialize_df(before_vc, "Before Resampling"),
                _serialize_df(after_vc, "After Resampling"),
            ],
            variables={"target": target, "method": method},
            dataset_name=st.session_state.get("file_name", ""),
        )
        if ai_texts.get("brief"):
            _log_entry["ai_interpretation"] = ai_texts["brief"]
        if ai_texts.get("deep_dive"):
            _log_entry["ai_deep_dive"] = ai_texts["deep_dive"]
        _include_chart = st.checkbox("Include chart in PDF", value=True, key="ci_pdf_chart")
        if _include_chart:
            _fig = px.bar(compare, x="Class", y="Count", color="Stage", barmode="group",
                          color_discrete_map={"Before": _ci_c["error"], "After": _ci_c["success"]})
            _fig.update_layout(height=400)
            _log_entry["figures"] = [{"label": "Before vs After", "fig_dict": _fig.to_dict()}]
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            if st.button("Add to Report", key="ci_add_report"):
                if log_result(_log_entry):
                    st.success("Added to report log.")
                else:
                    st.error("Report log is full (100 entries). Clear it first.")
        with exp_col2:
            st.download_button(
                "Export PDF",
                data=generate_single_report(_log_entry, include_charts=_include_chart),
                file_name="class_imbalance.pdf",
                mime="application/pdf",
                key="ci_pdf_download",
            )

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
- **Gower Distance Undersampling** — an **informed** undersampling method that keeps majority class samples closest to the minority class boundary based on Gower distance. Unlike random undersampling, it preserves the most informative majority samples — those near the decision boundary — while removing redundant ones far from it. Works natively on **mixed numeric + categorical features** without requiring encoding. Slower than random undersampling on large datasets due to distance matrix computation (O(n×m×p)).
- **SMOTE + Tomek Links** — a **hybrid** approach that first applies SMOTE to oversample the minority class, then removes **Tomek links** (pairs of nearest neighbors from different classes that are close together). This cleans up the decision boundary for better separation.

#### Before vs After Comparison
- After resampling, a side-by-side comparison shows the class counts and percentages before and after.
- A **grouped bar chart** visualizes the change in class distribution.
- Click **"Accept & Update Dataset"** to replace the current dataset with the resampled version.
        """)

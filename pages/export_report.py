"""Report Builder — combine saved analysis results into a single PDF report."""

import streamlit as st
from core.state import get_report_log, clear_report_log
from utils.pdf_export import generate_full_report


def render():
    st.title("Report Builder")
    st.markdown("Review saved analysis results and export them as a combined PDF report.")

    report_log = get_report_log()

    if not report_log:
        st.info(
            "No results saved yet. Run analyses on the statistics or data science pages "
            "and click **Add to Report** to start building your report."
        )
        return

    # ── Entry List ────────────────────────────────────────────────────────────
    st.subheader(f"Saved Results ({len(report_log)})")

    # Select All / Clear
    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        if st.button("Select All"):
            for entry in report_log:
                entry["include_in_report"] = True
            st.rerun()
    with ctrl_col2:
        if st.button("Clear Report Log"):
            clear_report_log()
            st.rerun()

    # Entry checkboxes
    to_remove = None
    for i, entry in enumerate(report_log):
        col_check, col_info, col_remove = st.columns([0.5, 5, 1])
        with col_check:
            included = st.checkbox(
                "Include",
                value=entry.get("include_in_report", True),
                key=f"rpt_inc_{i}",
                label_visibility="collapsed",
            )
            entry["include_in_report"] = included
        with col_info:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            entry_type = entry.get("entry_type", "unknown").replace("_", " ").title()
            st.markdown(f"**{entry.get('title', 'Untitled')}**  \n"
                        f"_{entry_type}_ | {ts} | {entry.get('dataset_name', '')}")
        with col_remove:
            if st.button("Remove", key=f"rpt_rm_{i}"):
                to_remove = i

    if to_remove is not None:
        report_log.pop(to_remove)
        st.session_state["report_log"] = report_log
        st.rerun()

    # ── Export Controls ───────────────────────────────────────────────────────
    st.divider()
    include_charts = st.checkbox("Include charts in PDF", value=True, key="rpt_charts")

    selected = [e for e in report_log if e.get("include_in_report", True)]
    if not selected:
        st.warning("No entries selected. Check at least one entry above to generate a report.")
        return

    st.markdown(f"**{len(selected)}** of {len(report_log)} entries selected for export.")

    # Build dataset info
    dataset_name = selected[0].get("dataset_name", "Unknown") if selected else "Unknown"
    df = st.session_state.get("df")
    dataset_info = {
        "name": dataset_name,
        "rows": len(df) if df is not None else 0,
        "cols": len(df.columns) if df is not None else 0,
    }

    with st.spinner("Generating PDF report..."):
        pdf_bytes = generate_full_report(
            report_log=selected,
            dataset_info=dataset_info,
            include_charts=include_charts,
        )

    st.download_button(
        "Download Full Report PDF",
        data=pdf_bytes,
        file_name="ds_power_tools_report.pdf",
        mime="application/pdf",
        type="primary",
    )

    # ── Page Guide ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("Page Guide & Explanation", expanded=False):
        st.markdown("""
#### Report Builder

This page collects all analysis results you have saved during your session and lets you export them as a single combined PDF report.

#### How It Works
1. Run any analysis on the statistics or data science pages (e.g., t-Test, ANOVA, Correlation, Model Arena).
2. After viewing results, click **Add to Report** to save the result to this page.
3. Come here to review, select/deselect, or remove entries.
4. Toggle **Include charts** to control whether chart images are embedded (charts increase file size).
5. Click **Download Full Report PDF** to generate and download the combined report.

#### Report Contents
- **Cover page** with dataset name, number of analyses, and generation timestamp.
- **One section per analysis** with test statistics, tables, assumptions, and optionally charts.
- Page numbers and headers on every page.

#### Notes
- Results are stored in your browser session only. Refreshing or closing the browser clears the log.
- Charts are exported as PNG images embedded in the PDF.
- Very wide tables (>8 columns) are automatically split into chunks.
- Tables longer than 100 rows are truncated.
        """)

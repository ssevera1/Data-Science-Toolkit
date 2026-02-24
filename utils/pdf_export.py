"""PDF report generation for DS Power Tools.

Pure Python module — no Streamlit imports.
Uses fpdf2 for PDF rendering and kaleido for Plotly chart export.
"""

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
from fpdf import FPDF


# ── Constants ────────────────────────────────────────────────────────────────

_ACCENT = (30, 80, 160)       # Blue accent for headers/bands
_ACCENT_LIGHT = (220, 230, 245)
_SUCCESS = (39, 174, 96)      # Green for significant / pass
_DANGER = (231, 76, 60)       # Red for not significant / fail
_TEXT = (30, 30, 30)
_MUTED = (120, 120, 120)
_TABLE_HEADER_BG = (30, 80, 160)
_TABLE_HEADER_FG = (255, 255, 255)
_TABLE_ALT_ROW = (240, 244, 250)
_MAX_TABLE_COLS = 8
_MAX_TABLE_ROWS = 100


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize_text(text) -> str:
    """Replace Unicode characters that core PDF fonts can't render.

    Accepts any type — coerces to str first, handles None gracefully.
    """
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "\u2014": "--",    # em-dash
        "\u2013": "-",     # en-dash
        "\u2018": "'",     # left single quote
        "\u2019": "'",     # right single quote
        "\u201c": '"',     # left double quote
        "\u201d": '"',     # right double quote
        "\u2026": "...",   # ellipsis
        "\u2265": ">=",    # >=
        "\u2264": "<=",    # <=
        "\u00b2": "^2",    # superscript 2
        "\u03b1": "alpha", # alpha
        "\u03b7": "eta",   # eta
        "\u03c9": "omega", # omega
        "\u03bc": "mu",    # mu
        "\u2080": "0",     # subscript 0
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    # Strip any remaining non-latin-1 characters
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _safe_str(val) -> str:
    """Convert a value to a display string, handling numpy/pandas types."""
    if val is None:
        return "N/A"
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return str(val)
    if isinstance(val, (np.integer,)):
        return str(int(val))
    if isinstance(val, (np.floating,)):
        return f"{float(val):.4f}"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, (np.bool_,)):
        return str(bool(val))
    return str(val)


def _native(val):
    """Convert numpy scalar to native Python type."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _serialize_df(df, label: str) -> dict:
    """Serialize a DataFrame to a dict for the log entry.

    Preserves the index as a column if it carries meaningful labels
    (i.e., is not a default integer RangeIndex).
    """
    out = df.head(_MAX_TABLE_ROWS)
    has_meaningful_index = not isinstance(out.index, pd.RangeIndex)
    if has_meaningful_index:
        idx_name = out.index.name or "Index"
        out = out.reset_index()
        if out.columns[0] != idx_name:
            out = out.rename(columns={out.columns[0]: idx_name})
    return {
        "label": label,
        "data": out.to_dict(orient="records"),
        "columns": list(out.columns),
    }


def _deserialize_df(table_dict: dict) -> pd.DataFrame:
    """Reconstruct a DataFrame from a serialized table dict.

    Returns an empty DataFrame if the dict is malformed.
    """
    try:
        return pd.DataFrame(table_dict["data"], columns=table_dict["columns"])
    except (KeyError, TypeError, ValueError):
        return pd.DataFrame()


# ── Report FPDF Subclass ────────────────────────────────────────────────────

class _DSReport(FPDF):
    """Styled PDF report for DS Power Tools."""

    def __init__(self, dataset_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._dataset_name = _sanitize_text(dataset_name)
        self.set_auto_page_break(auto=True, margin=20)

    # ── Header / Footer ──────────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return  # cover page has its own header
        self.set_fill_color(*_ACCENT)
        self.rect(0, 0, self.w, 8, style="F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_MUTED)
        self.set_y(10)
        self.cell(0, 5, f"DS Power Tools -- {self._dataset_name}", align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_MUTED)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── Text Helpers ─────────────────────────────────────────────────────

    def section_heading(self, text: str):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*_ACCENT)
        self.cell(0, 10, _sanitize_text(text), new_x="LMARGIN", new_y="NEXT")
        # underline
        self.set_draw_color(*_ACCENT)
        self.set_line_width(0.5)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(4)

    def sub_heading(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_TEXT)
        self.cell(0, 8, _sanitize_text(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*_TEXT)
        self.multi_cell(0, 5, _sanitize_text(text))
        self.ln(2)

    def kv_line(self, label: str, value: str):
        label = _sanitize_text(label)
        value = _sanitize_text(value)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_TEXT)
        w_label = self.get_string_width(label + ": ") + 2
        self.cell(w_label, 6, label + ": ")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def sig_badge(self, p_value: float, alpha: float = 0.05):
        if p_value < alpha:
            color = _SUCCESS
            label = f"Significant (p = {p_value:.4f})"
        else:
            color = _DANGER
            label = f"Not Significant (p = {p_value:.4f})"
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*color)
        self.cell(0, 7, label, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*_TEXT)
        self.ln(2)

    # ── Table Rendering ──────────────────────────────────────────────────

    def dataframe_table(self, df: pd.DataFrame, title: str = "",
                        max_cols: int = _MAX_TABLE_COLS,
                        max_rows: int = _MAX_TABLE_ROWS):
        if df is None or df.empty:
            return

        truncated_rows = len(df) > max_rows
        df = df.head(max_rows)
        columns = list(df.columns)

        # Split wide tables into chunks
        chunks = [columns[i:i + max_cols] for i in range(0, len(columns), max_cols)]

        for chunk_idx, col_chunk in enumerate(chunks):
            chunk_df = df[col_chunk]
            chunk_title = title if chunk_idx == 0 else f"{title} (cont.)"

            if chunk_title:
                self.sub_heading(chunk_title)

            n_cols = len(col_chunk)
            usable_w = self.w - self.l_margin - self.r_margin
            col_w = usable_w / n_cols

            # Header row
            self.set_font("Helvetica", "B", 8)
            self.set_fill_color(*_TABLE_HEADER_BG)
            self.set_text_color(*_TABLE_HEADER_FG)
            for col_name in col_chunk:
                self.cell(col_w, 7, _sanitize_text(str(col_name)[:20]), border=1, fill=True, align="C")
            self.ln()

            # Data rows
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_TEXT)
            for row_idx in range(len(chunk_df)):
                if row_idx % 2 == 1:
                    self.set_fill_color(*_TABLE_ALT_ROW)
                    fill = True
                else:
                    fill = False

                for col_name in col_chunk:
                    val = chunk_df.iloc[row_idx][col_name]
                    self.cell(col_w, 6, _sanitize_text(_safe_str(val)[:25]), border=1,
                              fill=fill, align="C")
                self.ln()

            self.ln(3)

        if truncated_rows:
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_MUTED)
            self.cell(0, 5, f"(First {max_rows} rows shown)", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

    # ── Chart Embedding ──────────────────────────────────────────────────

    _MAX_FIG_DICT_SIZE = 2_000_000   # 2 MB serialized figure dict
    _MAX_IMG_SIZE = 5_000_000        # 5 MB rendered PNG

    def embed_chart(self, fig_dict: dict, label: str = ""):
        """Reconstruct a Plotly figure from dict and embed as PNG."""
        try:
            import json
            import plotly.graph_objects as go

            # Guard against oversized figure dicts
            if len(json.dumps(fig_dict, default=str)) > self._MAX_FIG_DICT_SIZE:
                raise ValueError("Figure data too large")

            fig = go.Figure(fig_dict)
            # Force white background for print
            fig.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black"),
                width=700,
                height=400,
            )
            img_bytes = fig.to_image(format="png", scale=2)

            if len(img_bytes) > self._MAX_IMG_SIZE:
                raise ValueError("Rendered chart image too large")

            if label:
                self.sub_heading(label)

            # Check if we need a page break for the image
            if self.get_y() + 80 > self.h - 20:
                self.add_page()

            img_w = self.w - self.l_margin - self.r_margin - 10
            tmp = io.BytesIO(img_bytes)
            self.image(tmp, x=self.l_margin + 5, w=img_w)
            self.ln(5)
        except Exception:
            if label:
                self.sub_heading(label)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*_MUTED)
            self.cell(0, 6, "[Chart unavailable -- image export failed]",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(*_TEXT)
            self.ln(3)

    # ── Cover Page ───────────────────────────────────────────────────────

    def cover_page(self, dataset_name: str, n_entries: int, generated_at: str):
        self.add_page()
        # Blue band
        self.set_fill_color(*_ACCENT)
        self.rect(0, 0, self.w, 60, style="F")
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(255, 255, 255)
        self.set_y(18)
        self.cell(0, 12, "DS Power Tools", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 14)
        self.cell(0, 8, "Statistical Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")

        self.set_y(70)
        self.set_text_color(*_TEXT)
        self.set_font("Helvetica", "", 11)
        self.kv_line("Dataset", dataset_name)
        self.kv_line("Analyses included", str(n_entries))
        self.kv_line("Generated", generated_at)
        self.ln(10)


# ── Type-Specific Renderers ─────────────────────────────────────────────────

def _render_ttest_family(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render independent, paired, or one-sample t-test."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    # Variables used
    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    # Significance
    pdf.sig_badge(result.get("p", 1.0), alpha)

    # Test statistics
    pdf.kv_line("Test statistic (t)", _safe_str(result.get("t")))
    pdf.kv_line("p-value", _safe_str(result.get("p")))
    if "df" in result:
        pdf.kv_line("Degrees of freedom", _safe_str(result.get("df")))

    # Sample info
    if "n" in result:
        pdf.kv_line("N", _safe_str(result.get("n")))
    if "n1" in result:
        pdf.kv_line("N (group 1)", _safe_str(result.get("n1")))
        pdf.kv_line("N (group 2)", _safe_str(result.get("n2")))
    if "mean" in result:
        pdf.kv_line("Mean", _safe_str(result.get("mean")))
    if "mean1" in result:
        pdf.kv_line("Mean (group 1)", _safe_str(result.get("mean1")))
        pdf.kv_line("Mean (group 2)", _safe_str(result.get("mean2")))
    if "mean_diff" in result:
        pdf.kv_line("Mean difference", _safe_str(result.get("mean_diff")))
    if "ci_lower" in result:
        pdf.kv_line("95% CI", f"[{_safe_str(result.get('ci_lower'))}, {_safe_str(result.get('ci_upper'))}]")
    if "cohens_d" in result:
        pdf.kv_line("Cohen's d", _safe_str(result.get("cohens_d")))

    pdf.ln(3)

    # Tables
    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    # Assumptions
    _render_assumptions(pdf, result.get("assumptions", {}))

    # Charts
    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_anova(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render one-way ANOVA result."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    pdf.sig_badge(result.get("p", 1.0), alpha)

    pdf.kv_line("F-statistic", _safe_str(result.get("F")))
    pdf.kv_line("p-value", _safe_str(result.get("p")))
    if "df_between" in result:
        pdf.kv_line("df (between, within)",
                     f"({_safe_str(result.get('df_between'))}, {_safe_str(result.get('df_within'))})")
    if "eta_squared" in result:
        pdf.kv_line("Eta-squared", _safe_str(result.get("eta_squared")))
    if "omega_squared" in result:
        pdf.kv_line("Omega-squared", _safe_str(result.get("omega_squared")))

    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    _render_assumptions(pdf, result.get("assumptions", {}))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_correlation(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render Pearson/Spearman correlation."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    pdf.sig_badge(result.get("p", 1.0), alpha)

    pdf.kv_line("r", _safe_str(result.get("r")))
    if "r_squared" in result:
        pdf.kv_line("R-squared", _safe_str(result["r_squared"]))
    pdf.kv_line("p-value", _safe_str(result.get("p")))
    pdf.kv_line("N", _safe_str(result.get("n")))
    if "ci_lower" in result:
        pdf.kv_line("95% CI for r", f"[{_safe_str(result.get('ci_lower'))}, {_safe_str(result.get('ci_upper'))}]")

    pdf.ln(3)

    _render_assumptions(pdf, result.get("assumptions", {}))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_descriptive(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render descriptive statistics (tables only, no test result)."""
    pdf.section_heading(entry["title"])

    variables = entry.get("variables", {})
    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_model_arena(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render Model Arena benchmark results."""
    result = entry["result"]

    pdf.section_heading(entry["title"])

    pdf.kv_line("Task", str(result.get("task", "N/A")))
    pdf.kv_line("Best model", str(result.get("best_model", "N/A")))
    pdf.kv_line("Primary metric", str(result.get("primary_metric", "N/A")))
    pdf.kv_line("Best score", _safe_str(result.get("best_score")))
    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_feature_selection(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render feature selection summary."""
    result = entry["result"]

    pdf.section_heading(entry["title"])

    pdf.kv_line("Target", str(result.get("target", "N/A")))
    pdf.kv_line("Task", str(result.get("task", "N/A")))
    if result.get("top_features"):
        pdf.kv_line("Top features", ", ".join(str(f) for f in result["top_features"][:10]))
    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_assumptions(pdf: _DSReport, assumptions: dict):
    """Render assumption check results."""
    if not assumptions:
        return

    pdf.sub_heading("Assumption Checks")

    for key, val in assumptions.items():
        if isinstance(val, dict) and "statistic" in val:
            _render_single_assumption(pdf, key, val)
        elif isinstance(val, dict) and "passed" in val:
            # Assumption without a test statistic (e.g., chi-squared expected frequencies)
            _render_single_assumption(pdf, key, val)
        elif isinstance(val, dict):
            # Nested (e.g., normality per group, homogeneity per DV)
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, dict) and ("statistic" in sub_val or "passed" in sub_val):
                    _render_single_assumption(pdf, f"{key}: {sub_key}", sub_val)

    pdf.ln(2)


def _render_single_assumption(pdf: _DSReport, name: str, check: dict):
    """Render one assumption check line."""
    passed = check.get("passed", True)
    color = _SUCCESS if passed else _DANGER
    status = "PASS" if passed else "FAIL"

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*color)
    pdf.cell(12, 5, f"[{status}]")
    pdf.set_text_color(*_TEXT)
    pdf.set_font("Helvetica", "", 9)
    stat_str = _safe_str(check.get("statistic"))
    p_str = _safe_str(check.get("p_value"))
    pdf.cell(0, 5, _sanitize_text(f"  {name}  (stat={stat_str}, p={p_str})"),
             new_x="LMARGIN", new_y="NEXT")
    if check.get("detail"):
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_MUTED)
        pdf.cell(0, 4, _sanitize_text(f"    {check['detail']}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*_TEXT)
    pdf.ln(1)


def _render_nonparametric(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render Mann-Whitney, Wilcoxon, Kruskal-Wallis, or Friedman results."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    pdf.sig_badge(result.get("p", 1.0), alpha)

    # Test statistic (varies by test type)
    for stat_key, stat_label in [("U", "U statistic"), ("W", "W statistic"),
                                  ("H", "H statistic"), ("chi2", "Chi-squared")]:
        if stat_key in result:
            pdf.kv_line(stat_label, _safe_str(result[stat_key]))
    pdf.kv_line("p-value", _safe_str(result.get("p")))
    if "df" in result:
        pdf.kv_line("Degrees of freedom", _safe_str(result["df"]))

    # Sample info
    for key in ["n", "n1", "n2", "n_subjects", "n_conditions"]:
        if key in result:
            pdf.kv_line(key.replace("_", " ").title(), _safe_str(result[key]))
    for key in ["median1", "median2", "median_diff"]:
        if key in result:
            pdf.kv_line(key.replace("_", " ").replace("1", " 1").replace("2", " 2").title(),
                        _safe_str(result[key]))

    # Effect sizes
    for key, label in [("rank_biserial", "Rank-biserial correlation"),
                        ("r_effect", "Effect size (r)"),
                        ("epsilon_squared", "Epsilon-squared"),
                        ("kendalls_w", "Kendall's W")]:
        if key in result:
            pdf.kv_line(label, _safe_str(result[key]))

    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    _render_assumptions(pdf, result.get("assumptions", {}))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_anova_general(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render two-way, repeated measures, or mixed ANOVA."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    # Extract smallest p-value from the ANOVA table for the significance badge
    _anova_tables = [t for t in entry.get("tables", []) if "ANOVA" in t.get("label", "")]
    if _anova_tables:
        _aov_df = _deserialize_df(_anova_tables[0])
        for _pcol in ("p-unc", "p", "p-value", "Pr(>F)"):
            if _pcol in _aov_df.columns:
                _pvals = pd.to_numeric(_aov_df[_pcol], errors="coerce").dropna()
                if len(_pvals) > 0:
                    pdf.sig_badge(float(_pvals.min()), alpha)
                break
    elif "p" in result:
        pdf.sig_badge(result["p"], alpha)

    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    _render_assumptions(pdf, result.get("assumptions", {}))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_manova(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render MANOVA results."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    if "overall_p" in result:
        pdf.sig_badge(result["overall_p"], alpha)

    pdf.kv_line("N (complete cases)", _safe_str(result.get("n")))
    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    _render_assumptions(pdf, result.get("assumptions", {}))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_regression(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render linear or logistic regression results."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    # Model-level significance
    if "f_p" in result:
        pdf.sig_badge(result["f_p"], alpha)
        pdf.kv_line("F-statistic", _safe_str(result.get("f_stat")))
        pdf.kv_line("p-value (F)", _safe_str(result["f_p"]))
    elif "chi2_p" in result:
        pdf.sig_badge(result["chi2_p"], alpha)
        pdf.kv_line("Chi-squared (LR)", _safe_str(result.get("chi2")))
        pdf.kv_line("p-value (LR)", _safe_str(result["chi2_p"]))

    # Fit statistics
    for key, label in [("r_squared", "R-squared"), ("adj_r_squared", "Adj. R-squared"),
                        ("pseudo_r_squared", "Pseudo R-squared (McFadden)"),
                        ("accuracy", "Accuracy"), ("n", "N"),
                        ("aic", "AIC"), ("bic", "BIC")]:
        if key in result:
            pdf.kv_line(label, _safe_str(result[key]))

    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    _render_assumptions(pdf, result.get("assumptions", {}))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_chi_squared(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render chi-squared test results."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    pdf.sig_badge(result.get("p", 1.0), alpha)

    pdf.kv_line("Chi-squared", _safe_str(result.get("chi2")))
    pdf.kv_line("p-value", _safe_str(result.get("p")))
    pdf.kv_line("Degrees of freedom", _safe_str(result.get("df")))
    pdf.kv_line("N", _safe_str(result.get("n")))
    pdf.kv_line("Cramer's V", _safe_str(result.get("cramers_v")))

    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    _render_assumptions(pdf, result.get("assumptions", {}))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_binomial(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render binomial test results."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    pdf.sig_badge(result.get("p", 1.0), alpha)

    pdf.kv_line("N", _safe_str(result.get("n")))
    pdf.kv_line("Observed proportion", _safe_str(result.get("observed_prop")))
    pdf.kv_line("Test proportion", _safe_str(result.get("test_prop")))
    pdf.kv_line("p-value", _safe_str(result.get("p")))
    if "n_success" in result:
        pdf.kv_line(f"'{result.get('success_label', 'Success')}'",
                    _safe_str(result.get("n_success")))
        pdf.kv_line(f"'{result.get('failure_label', 'Failure')}'",
                    _safe_str(result.get("n_failure")))
    if "ci_lower" in result:
        pdf.kv_line("95% CI for proportion",
                    f"[{_safe_str(result.get('ci_lower'))}, {_safe_str(result.get('ci_upper'))}]")

    pdf.ln(3)

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_multivariate_regression(pdf: _DSReport, entry: dict, include_charts: bool):
    """Render multivariate regression results."""
    result = entry["result"]
    alpha = entry.get("alpha", 0.05)
    variables = entry.get("variables", {})

    pdf.section_heading(entry["title"])

    for k, v in variables.items():
        pdf.kv_line(k.replace("_", " ").title(), str(v))
    pdf.ln(2)

    pdf.kv_line("N (complete cases)", _safe_str(result.get("n")))
    pdf.ln(3)

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    _render_assumptions(pdf, result.get("assumptions", {}))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


def _render_fallback(pdf: _DSReport, entry: dict, include_charts: bool):
    """Fallback renderer for unknown entry types."""
    pdf.section_heading(entry.get("title", "Analysis Result"))
    pdf.body_text(f"Export for entry type '{entry.get('entry_type', 'unknown')}' "
                  "is not yet supported in PDF reports.")

    for tbl in entry.get("tables", []):
        df = _deserialize_df(tbl)
        pdf.dataframe_table(df, tbl.get("label", ""))

    if include_charts:
        for fig_entry in entry.get("figures", []):
            pdf.embed_chart(fig_entry["fig_dict"], fig_entry.get("label", ""))


# ── Renderer Dispatch ────────────────────────────────────────────────────────

_RENDERERS = {
    "independent_ttest": _render_ttest_family,
    "paired_ttest": _render_ttest_family,
    "one_sample_ttest": _render_ttest_family,
    "oneway_anova": _render_anova,
    "twoway_anova": _render_anova_general,
    "repeated_anova": _render_anova_general,
    "mixed_anova": _render_anova_general,
    "manova": _render_manova,
    "pearson_correlation": _render_correlation,
    "spearman_correlation": _render_correlation,
    "mann_whitney": _render_nonparametric,
    "wilcoxon": _render_nonparametric,
    "kruskal_wallis": _render_nonparametric,
    "friedman": _render_nonparametric,
    "linear_regression": _render_regression,
    "logistic_regression": _render_regression,
    "multivariate_regression": _render_multivariate_regression,
    "chi_squared": _render_chi_squared,
    "binomial": _render_binomial,
    "descriptive_stats": _render_descriptive,
    "model_arena": _render_model_arena,
    "feature_selection": _render_feature_selection,
}


# ── Public API ───────────────────────────────────────────────────────────────

def build_log_entry(
    entry_type: str,
    title: str,
    result: dict,
    tables: list[dict] | None = None,
    figures: list[dict] | None = None,
    variables: dict | None = None,
    alpha: float = 0.05,
    dataset_name: str = "",
) -> dict:
    """Build a normalized log entry for the report log.

    Parameters
    ----------
    entry_type : str
        Dispatcher key (e.g. "independent_ttest").
    title : str
        Human-readable title.
    result : dict
        Stats result dict. DataFrames are stripped; numpy scalars are converted.
    tables : list of dicts, optional
        Pre-serialized table dicts (via _serialize_df).
    figures : list of dicts, optional
        Each has {"label": str, "fig_dict": dict}.
    variables : dict, optional
        Variable names used in the analysis.
    alpha : float
        Significance level.
    dataset_name : str
        Name of the loaded dataset file.
    """
    # Recursively convert numpy types in result
    clean_result = _deep_native(result)

    return {
        "entry_type": entry_type,
        "title": title,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name or "Unknown",
        "result": clean_result,
        "tables": tables or [],
        "figures": figures or [],
        "variables": variables or {},
        "alpha": alpha,
        "include_in_report": True,
    }


def _deep_native(obj):
    """Recursively convert numpy/pandas types to native Python.

    Strips DataFrames, Series, and non-serializable objects (e.g., statsmodels
    model instances) to keep the log entry lightweight.
    """
    if isinstance(obj, dict):
        return {k: _deep_native(v) for k, v in obj.items()
                if not isinstance(v, (pd.DataFrame, pd.Series))
                and not _is_model_object(v)}
    if isinstance(obj, (list, tuple)):
        return [_deep_native(v) for v in obj]
    return _native(obj)


def _is_model_object(obj) -> bool:
    """Check if obj is a statsmodels or sklearn model (non-serializable)."""
    type_name = type(obj).__module__
    return type_name.startswith(("statsmodels.", "sklearn."))


def generate_single_report(entry: dict, include_charts: bool = True) -> bytes:
    """Generate a compact single-result PDF report.

    Returns raw PDF bytes suitable for st.download_button.
    """
    dataset_name = entry.get("dataset_name", "Unknown")
    pdf = _DSReport(dataset_name=dataset_name, orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.add_page()

    # Compact header (no cover page for single reports)
    pdf.set_fill_color(*_ACCENT)
    pdf.rect(0, 0, pdf.w, 12, style="F")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(3)
    pdf.cell(0, 6, f"DS Power Tools -- {_sanitize_text(dataset_name)}", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(18)
    pdf.set_text_color(*_MUTED)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 4, f"Generated: {entry.get('timestamp', '')}", align="R",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    renderer = _RENDERERS.get(entry["entry_type"], _render_fallback)
    renderer(pdf, entry, include_charts)

    return pdf.output()


def generate_full_report(
    report_log: list[dict],
    dataset_info: dict | None = None,
    include_charts: bool = True,
) -> bytes:
    """Generate a combined multi-section PDF report.

    Parameters
    ----------
    report_log : list of dicts
        Entries to include (those with include_in_report=True).
    dataset_info : dict, optional
        {"name": str, "rows": int, "cols": int}.
    include_charts : bool
        Whether to embed chart images.

    Returns raw PDF bytes.
    """
    dataset_info = dataset_info or {}
    dataset_name = dataset_info.get("name", "Unknown")
    entries = [e for e in report_log if e.get("include_in_report", True)]

    pdf = _DSReport(dataset_name=dataset_name, orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()

    # Cover page
    pdf.cover_page(
        dataset_name=dataset_name,
        n_entries=len(entries),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    # Render each entry
    for i, entry in enumerate(entries):
        pdf.add_page()
        renderer = _RENDERERS.get(entry.get("entry_type", ""), _render_fallback)
        renderer(pdf, entry, include_charts)

    return pdf.output()

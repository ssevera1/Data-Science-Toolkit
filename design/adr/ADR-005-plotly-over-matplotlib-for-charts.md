# ADR-005: Chosen Plotly Over Matplotlib for Interactive Charts

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Technology / Visualization |

## Context

The application needs to render statistical charts (box plots, histograms, scatter
plots, Q-Q plots, bar plots, regression plots) that are embedded within a web UI.
Users should be able to interact with charts (zoom, hover for values, pan) to
explore their data.

### Options Considered

| Library | Pros | Cons |
|---|---|---|
| **Matplotlib** | De facto standard, extensive customization, publication-quality | Static images, no interactivity in web context, slow rendering for large data |
| **Plotly** | Interactive (zoom, hover, pan), native Streamlit integration, web-native | Larger bundle size, less customizable for publication, different API mental model |
| **Altair/Vega** | Declarative grammar, good Streamlit support | Limited chart types, performance ceiling with large datasets |
| **Bokeh** | Interactive, server-side rendering | Heavier setup, less Streamlit integration |

## Decision

**Chosen: Plotly** as the primary charting library for all statistical visualizations
rendered in the UI. Matplotlib and Seaborn are retained as secondary libraries
for specific Data Science tool pages (SHAP plots, feature importance) where
scikit-learn and SHAP output Matplotlib figures natively.

## Rationale

1. **Native Streamlit integration**: `st.plotly_chart()` renders Plotly figures
   with zero configuration. The figures are interactive in the browser with zoom,
   pan, hover tooltips, and screenshot export built in.

2. **Hover data for statistics**: Users can hover over box plot whiskers to see
   exact quartile values, hover over bars to see means and standard deviations,
   and hover over scatter points to see individual data values. This interactivity
   is essential for exploratory data analysis.

3. **Client-side rendering**: Plotly renders charts in the browser using WebGL/SVG,
   offloading rendering from the Python server. This keeps the server responsive
   during visualization of large datasets.

4. **Theming support**: Plotly's template system allows registering custom themes
   (color palettes, fonts, grid styles, background colors) that can be switched
   dynamically. This enables the dual Light/Dark theme system.

5. **Consistent API across chart types**: Box plots, histograms, scatter plots,
   and bar charts all use `plotly.graph_objects` or `plotly.express` with a
   consistent API, making the `charts/` module predictable and maintainable.

## Implementation

```
charts/
├── theme.py          # Register Light/Dark Plotly templates
├── boxplot.py        # go.Box with jitter points
├── histogram.py      # go.Histogram with normal curve overlay
├── scatter.py        # go.Scatter with regression line
├── barplot.py        # go.Bar with error bars
├── qq_plot.py        # go.Scatter (theoretical vs. sample quantiles)
└── regression_plot.py # go.Scatter with confidence bands
```

### Theme Registration Pattern

```python
# charts/theme.py
def get_plotly_template(theme):
    if theme == "Dark":
        return go.layout.Template(
            layout=go.Layout(
                paper_bgcolor="#1a1a2e",
                plot_bgcolor="#16213e",
                font=dict(color="#e0e0e0"),
                # ...
            )
        )
    else:
        return go.layout.Template(
            layout=go.Layout(
                paper_bgcolor="#faf3e8",
                plot_bgcolor="#ffffff",
                font=dict(color="#333333"),
                # ...
            )
        )
```

## Trade-offs Accepted

- **Publication quality**: Matplotlib produces higher-quality static figures for
  academic papers. Users who need publication-ready plots may need to export data
  and re-plot in Matplotlib or R. The application prioritizes interactive
  exploration over print-ready output.

- **Bundle size**: Plotly's JavaScript bundle (~3 MB) is loaded client-side. This
  adds to initial page load time but is cached after the first load.

- **Dual library maintenance**: Some Data Science pages use Matplotlib (via SHAP
  and scikit-learn), creating two visualization stacks. The `charts/` module
  exclusively uses Plotly to maintain consistency in the statistics tools, but
  the DS tools accept Matplotlib output where upstream libraries produce it.

- **Limited statistical chart types**: Plotly doesn't have native Q-Q plot or
  forest plot types. These are constructed manually using `go.Scatter`, which
  requires more code than Matplotlib's `probplot()` but produces interactive
  results.

## Consequences

- All statistical charts are interactive by default
- Chart theming is centralized in `charts/theme.py`
- New chart types follow the established pattern: function takes `(df, columns, theme)` and returns `go.Figure`
- Matplotlib is only used where external libraries (SHAP, sklearn) produce Matplotlib output directly

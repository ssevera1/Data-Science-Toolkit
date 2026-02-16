# ADR-001: Chosen Streamlit as the Application Framework

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Architecture / Framework |

## Context

The application needed a web-based UI framework that would allow rapid development
of an interactive data science and statistics toolkit. The primary user is a data
scientist or researcher who needs to upload datasets, configure analyses, and
interpret results through charts and formatted output.

### Options Considered

| Framework | Pros | Cons |
|---|---|---|
| **Streamlit** | Python-native, zero frontend code, built-in widgets for data, rapid prototyping, session state, automatic reruns | Limited layout control, server-side rendering, no fine-grained reactivity, full rerun on interaction |
| **Dash (Plotly)** | More layout control, callback-based reactivity, Plotly-native | More boilerplate, explicit callback wiring, steeper learning curve |
| **Flask + React** | Full control, SPA experience, REST API separation | Massive development overhead, two codebases, two languages |
| **Jupyter Voila** | Notebook-native, familiar to data scientists | Limited widget set, poor navigation, not a real app |
| **Panel (HoloViz)** | Python-native, flexible layout | Smaller community, less mature ecosystem |

## Decision

**Chosen: Streamlit** (v1.40+)

## Rationale

1. **Python-only stack**: The entire application is written in Python with zero
   JavaScript. This keeps the codebase accessible to data scientists who may not
   have frontend experience, and eliminates the complexity of a polyglot stack.

2. **Widget ecosystem**: Streamlit provides native widgets for file upload,
   sliders, selectboxes, dataframes, and charts - all of which are core to the
   application's functionality. No custom widget development was required.

3. **Multi-page navigation**: Streamlit's `st.navigation()` and `st.Page()` APIs
   provide first-class multi-page support, enabling the 32-page architecture with
   clean URL routing and sidebar navigation.

4. **Session state**: `st.session_state` provides a simple key-value store that
   persists across reruns within a user session. This enables the shared DataFrame
   pattern where data uploaded on any page is available to all other pages.

5. **Deployment simplicity**: A single `streamlit run app.py` command launches
   the entire application. No build step, no bundling, no container required
   (though containers are supported).

## Trade-offs Accepted

- **Full reruns on interaction**: Every widget interaction triggers a full script
  rerun from top to bottom. This means computation-heavy operations (Model Arena,
  SHAP) must be carefully guarded behind button clicks and `st.cache` decorators to
  avoid redundant computation.

- **Limited layout control**: Streamlit's column/container system is less flexible
  than CSS Grid or Flexbox. Complex layouts require workarounds with `st.columns()`
  and custom CSS injection via `st.markdown()`.

- **No fine-grained reactivity**: Unlike React or Dash callbacks, there is no way
  to update only a portion of the page. The `st.fragment` API helps but adds
  complexity.

- **Custom theming requires CSS injection**: Streamlit's built-in theming is
  limited. The dual-theme system (Light/Dark) required extensive CSS injection via
  `st.markdown(unsafe_allow_html=True)`, which is fragile and tightly coupled to
  Streamlit's internal DOM structure.

## Consequences

- All developers must know Python (no frontend specialization needed)
- UI complexity is bounded by Streamlit's widget set
- Performance-sensitive operations require explicit caching strategies
- The application is single-user per session (no shared state between users)
- Theme customization is more complex than it would be in a CSS-first framework

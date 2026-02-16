# ADR-008: Custom Dual-Theme System via CSS Injection

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | UX / Styling |

## Context

The application needed a professional, branded visual identity that supports both
light and dark mode preferences. Streamlit provides a built-in theme system, but
it is limited to a few color variables and does not support the level of
customization needed for a cohesive, polished experience.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **Streamlit built-in theme** | Zero effort, consistent with Streamlit ecosystem | Limited to ~5 color variables, no widget-level control, no gradient support |
| **Custom CSS via st.markdown** | Full control over every element, gradients, animations | Fragile (depends on Streamlit DOM structure), verbose, maintenance overhead |
| **Streamlit Components (React)** | Full frontend control | Requires JavaScript, breaks Python-only philosophy |
| **Third-party theme library** | Pre-built themes | Limited selection, dependency on unmaintained packages |

## Decision

**Chosen: Custom dual-theme system using CSS injection via `st.markdown(unsafe_allow_html=True)`**, with theme-synchronized Plotly templates.

## Rationale

1. **Brand identity**: The application has a distinct visual identity - Dark theme
   uses deep blue/purple backgrounds with cyan accents; Light theme uses warm
   cream backgrounds with red accents. This level of customization is not possible
   with Streamlit's built-in theme variables.

2. **Widget-level styling**: The custom CSS styles specific Streamlit widgets
   (metric cards, expanders, tabs, buttons, data editors, selectboxes) to match
   the theme. The built-in theme only affects global colors, not individual
   component appearances.

3. **Chart consistency**: Plotly charts register custom templates that match the
   surrounding UI theme. Without this, charts would have white backgrounds in
   dark mode, breaking visual cohesion.

4. **Session-based switching**: Users can switch themes via a sidebar toggle.
   The switch triggers a Streamlit rerun that reapplies the appropriate CSS and
   Plotly template.

## Implementation Architecture

```
utils/theme.py                    charts/theme.py
├── apply_theme(theme)            ├── register_plotly_template(theme)
├── get_css(theme)                ├── get_plotly_template(theme)
│   ├── _light_css()              │   ├── _light_template()
│   └── _dark_css()               │   └── _dark_template()
└── get_metric_card_css(theme)    └── get_color_palette(theme)

        │                                   │
        └──────────── app.py ───────────────┘
                 (applies both on each rerun)
```

### Theme Properties

| Property | Dark Theme | Light Theme |
|---|---|---|
| Background | `#1a1a2e` (deep navy) | `#faf3e8` (warm cream) |
| Card background | `#16213e` | `#ffffff` |
| Primary accent | `#00d4ff` (cyan) | `#c0392b` (red) |
| Secondary accent | `#7c3aed` (purple) | `#e74c3c` (light red) |
| Text color | `#e0e0e0` (light gray) | `#333333` (dark gray) |
| Border style | Subtle gradient borders | Solid warm-toned borders |
| Chart palette | Cool blues, purples, cyans | Warm reds, oranges, greens |

## Trade-offs Accepted

- **Fragile CSS selectors**: The CSS targets Streamlit's internal DOM structure
  (e.g., `.stMetric`, `.stExpander`, `[data-testid="stSidebar"]`). Streamlit
  updates can change these selectors, requiring CSS maintenance. This has already
  required fixes in recent commits (emoji squares, spinner visibility).

- **No CSS-in-JS or design tokens**: The CSS is generated as a raw string in
  Python, not managed by a proper styling system. This makes the styling code
  verbose and harder to refactor.

- **`unsafe_allow_html=True`**: This flag is required for CSS injection and
  disables Streamlit's XSS protection for that element. The risk is mitigated
  because the HTML/CSS is generated entirely from application code, not from
  user input.

- **Theme state mismatch during transition**: When the user switches themes,
  the page reruns. During the rerun, there's a brief flash of the previous
  theme before the new CSS applies. This is a Streamlit limitation with no
  clean workaround.

## Consequences

- The application has a distinctive, professional appearance in both modes
- Theme changes are instantaneous (single rerun)
- All UI elements and charts are theme-consistent
- Streamlit version upgrades require CSS compatibility testing
- Adding new UI elements requires considering both theme variants

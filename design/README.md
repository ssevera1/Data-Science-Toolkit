# Design Documentation - DS Power Tools

This directory contains architectural documentation for DS Power Tools, including
C4 model diagrams and Architecture Decision Records (ADRs).

## Contents

### C4 Model Diagrams

The [C4 model](https://c4model.com/) provides a hierarchical set of diagrams
that describe the software architecture at four levels of abstraction. All diagrams
use [Mermaid.js](https://mermaid.js.org/) for rendering and are editable as
plain Markdown.

| Level | Document | Description |
|---|---|---|
| **Level 1** | [c4-context.md](c4-context.md) | **System Context** - Shows DS Power Tools as a single system, its users, and external interactions. Highlights the privacy boundary and latency characteristics. |
| **Level 2** | [c4-container.md](c4-container.md) | **Container** - Zooms into the system boundary to reveal the major architectural layers: pages, core, stats, charts, components, and utils. Includes dependency graph and inter-container communication. |
| **Level 3** | [c4-component.md](c4-component.md) | **Component** - Shows the internal modules within each container, their responsibilities, and cross-container interactions. Includes the DS pipeline flow and statistical test execution sequence. |
| **Level 4** | [c4-code.md](c4-code.md) | **Code** - Reveals function signatures, data structures, class diagrams, and interface contracts. Covers the statistical test contract, variable detection logic, and page module patterns. |

### Architecture Decision Records (ADRs)

The [adr/](adr/) directory contains 14 ADRs documenting the key design decisions
and their rationale. Each ADR follows a structured template with context, options
considered, decision, rationale, trade-offs, and consequences.

See the [ADR index](adr/README.md) for the complete list.

#### Key ADRs by Audience

| Audience | Start With |
|---|---|
| New contributors | ADR-001, ADR-006, ADR-011 |
| Security reviewers | ADR-002 |
| Statisticians | ADR-004, ADR-009, ADR-012 |
| ML engineers | ADR-010, ADR-013 |
| UX/Frontend | ADR-005, ADR-007, ADR-008, ADR-014 |

## Rendering Diagrams

The Mermaid.js diagrams can be rendered in:

- **GitHub**: Renders Mermaid blocks natively in Markdown files
- **VS Code**: Install the [Mermaid Preview](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) extension
- **Mermaid Live Editor**: Paste diagram code at [mermaid.live](https://mermaid.live)
- **Excalidraw**: Import Mermaid syntax via the Mermaid-to-Excalidraw plugin for editable whiteboard diagrams

## Architecture Overview

```
DS Power Tools
├── app.py                    Entry point, navigation, theme
├── pages/ (32 modules)       View layer (DS tools + Statistics tools)
├── core/ (4 modules)         State management, data I/O, validation
├── stats/ (13 modules)       Statistical computation engine
├── charts/ (7 modules)       Plotly visualization generation
├── components/ (4 modules)   Reusable UI widgets
└── utils/ (2 modules)        Dual-theme styling system
```

### Data Flow

```
User uploads CSV/Excel
  → core/data_manager.py loads & auto-detects variable types
    → st.session_state stores DataFrame + types
      → pages/ render UI, call stats/ for computation
        → charts/ generate Plotly figures
          → components/ format and display results
```

### Key Design Principles

1. **Privacy by architecture**: Zero external network dependencies (ADR-002)
2. **Computation/presentation split**: Stats modules are pure functions, pages handle UI (ADR-011)
3. **Assumption-first testing**: Every statistical test includes automatic assumption checks (ADR-009)
4. **Effect sizes always**: Modern statistical practice with mandatory effect size reporting (ADR-012)
5. **One page, one purpose**: Each statistical test has its own dedicated page (ADR-014)

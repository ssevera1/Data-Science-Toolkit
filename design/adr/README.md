# Architecture Decision Records (ADRs)

This directory contains the Architecture Decision Records for DS Power Tools.
Each ADR documents a significant design decision, the alternatives considered,
the rationale for the choice made, and the trade-offs accepted.

## ADR Index

| ADR | Title | Category | Status |
|---|---|---|---|
| [ADR-001](ADR-001-streamlit-as-framework.md) | Chosen Streamlit as the Application Framework | Architecture / Framework | Accepted |
| [ADR-002](ADR-002-local-only-privacy-architecture.md) | Local-Only Privacy-First Architecture | Architecture / Security | Accepted |
| [ADR-003](ADR-003-session-state-as-data-bus.md) | Chosen Streamlit Session State as the Shared Data Bus | Architecture / State | Accepted |
| [ADR-004](ADR-004-scipy-statsmodels-over-custom-implementations.md) | Chosen SciPy + statsmodels Over Custom Implementations | Technology / Computation | Accepted |
| [ADR-005](ADR-005-plotly-over-matplotlib-for-charts.md) | Chosen Plotly Over Matplotlib for Interactive Charts | Technology / Visualization | Accepted |
| [ADR-006](ADR-006-dual-toolset-shared-data-architecture.md) | Dual-Toolset Architecture with Shared Data Layer | Architecture / Product | Accepted |
| [ADR-007](ADR-007-automatic-variable-type-detection.md) | Automatic Variable Type Detection with Manual Override | UX / Data Handling | Accepted |
| [ADR-008](ADR-008-dual-theme-system.md) | Custom Dual-Theme System via CSS Injection | UX / Styling | Accepted |
| [ADR-009](ADR-009-assumption-first-test-design.md) | Assumption-First Statistical Test Design | Product / Methodology | Accepted |
| [ADR-010](ADR-010-optuna-for-hyperparameter-tuning.md) | Chosen Optuna Over Grid/Random Search for Hyperparameter Tuning | Technology / ML Pipeline | Accepted |
| [ADR-011](ADR-011-separation-of-computation-and-presentation.md) | Strict Separation of Computation and Presentation | Architecture / Code Org | Accepted |
| [ADR-012](ADR-012-effect-sizes-and-modern-statistical-practice.md) | Mandatory Effect Sizes Alongside p-Values | Product / Methodology | Accepted |
| [ADR-013](ADR-013-scikit-learn-as-ml-backbone.md) | Chosen scikit-learn as the ML Backbone | Technology / ML Pipeline | Accepted |
| [ADR-014](ADR-014-page-per-test-navigation.md) | One Page Per Statistical Test Navigation Model | UX / Navigation | Accepted |

## ADR Template

When adding new ADRs, use the following structure:

```markdown
# ADR-NNN: Title

| Field | Value |
|---|---|
| **Status** | Proposed / Accepted / Deprecated / Superseded |
| **Date** | YYYY-MM |
| **Decision Makers** | Names |
| **Category** | Category |

## Context
What is the issue? What forces are at play?

## Options Considered
Table of alternatives with pros and cons.

## Decision
What was chosen?

## Rationale
Why was this option selected? (Numbered reasons)

## Trade-offs Accepted
What downsides were knowingly accepted?

## Consequences
What are the resulting impacts on the project?
```

## Reading Guide

- **Start here**: ADR-001 (framework choice) and ADR-006 (dual-toolset architecture) explain the foundational decisions
- **For security reviewers**: ADR-002 (privacy architecture) documents the zero-network-dependency design
- **For statisticians**: ADR-004, ADR-009, ADR-012 explain the statistical methodology choices
- **For ML engineers**: ADR-010, ADR-013 explain the ML toolchain decisions
- **For frontend/UX**: ADR-005, ADR-007, ADR-008, ADR-014 explain the presentation layer decisions

# C4 Model - Level 1: System Context Diagram

## Overview

The System Context diagram shows DS Power Tools as a single system and its relationship
to the users and external entities it interacts with. This is the highest level of
abstraction, showing the "big picture."

## Key Observations

- DS Power Tools is a **fully local application** with zero external service dependencies
- The only actor is the end user on a local machine
- Data never leaves the local environment - a deliberate privacy-first architecture
- The system boundary encompasses all computation, storage, and presentation

## Diagram

```mermaid
C4Context
    title System Context Diagram - DS Power Tools

    Person(user, "Data Scientist / Researcher", "Performs statistical analysis and machine learning workflows on local datasets")

    System(dspowertools, "DS Power Tools", "Combined Data Science & Statistics Toolkit. Provides automated ML workflows and 20+ statistical tests with assumption checking, effect sizes, and interactive visualizations. Runs 100% locally.")

    System_Ext(filesystem, "Local File System", "CSV and Excel files containing user datasets (up to 200 MB)")

    System_Ext(browser, "Web Browser", "Renders the Streamlit UI on localhost:8501")

    Rel(user, dspowertools, "Uploads data, configures analyses, interprets results")
    Rel(user, browser, "Interacts via")
    Rel(browser, dspowertools, "HTTP requests to localhost:8501")
    Rel(dspowertools, filesystem, "Reads input files, exports results")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

## Data Flow Summary

```mermaid
flowchart LR
    A[User] -->|Upload CSV/Excel/Paste| B[DS Power Tools]
    B -->|Statistical Results & Charts| A
    B -->|Export CSV/Excel| C[Local File System]
    C -->|Read Dataset| B

    style B fill:#4A90D9,stroke:#2C5F8A,color:#fff
    style A fill:#08427B,stroke:#052E56,color:#fff
    style C fill:#999999,stroke:#666666,color:#fff
```

## Latency Characteristics

| Interaction | Typical Latency | Notes |
|---|---|---|
| File Upload | < 1s for < 50 MB | Pandas read_csv/read_excel |
| Statistical Test | < 500ms | SciPy/statsmodels computations |
| ANOVA with Post-hoc | < 2s | Iterative pairwise comparisons |
| Model Arena (10 models) | 10-60s | K-fold cross-validation across all models |
| Hyperparameter Tuning | 30s-5min | Bayesian optimization (Optuna), user-configurable trials |
| SHAP Explainability | 5-30s | Depends on model complexity and data size |
| Chart Rendering | < 1s | Plotly client-side rendering |

## Privacy Boundary

```mermaid
flowchart TB
    subgraph LOCAL["Local Machine Boundary"]
        direction TB
        subgraph APP["DS Power Tools (localhost:8501)"]
            S[Streamlit Server]
            P[Python Runtime]
        end
        subgraph DATA["User Data"]
            F[CSV/Excel Files]
            M[In-Memory DataFrame]
        end
        B[Web Browser]
    end

    CLOUD["Cloud / Internet"]

    B <-->|localhost only| S
    S <--> P
    P <--> M
    P <-->|Read/Export| F

    LOCAL -.->|No connection| CLOUD

    style CLOUD fill:#ff6b6b,stroke:#cc4444,color:#fff
    style LOCAL fill:#e8f5e9,stroke:#4caf50
    style APP fill:#bbdefb,stroke:#1976d2
    style DATA fill:#fff9c4,stroke:#f9a825
```

> **Design Decision**: The system is architecturally incapable of sending data externally.
> There are no HTTP client libraries imported, no API keys configured, and Streamlit's
> usage telemetry is explicitly disabled in `.streamlit/config.toml`.

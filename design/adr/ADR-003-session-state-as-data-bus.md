# ADR-003: Chosen Streamlit Session State as the Shared Data Bus

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Architecture / State Management |

## Context

The application has two independent toolsets (Data Science and Statistics) that
both operate on the same user dataset. A mechanism was needed to share the
DataFrame and its metadata (variable types, column selections) across 32 pages
without passing data through URL parameters or file I/O.

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **st.session_state** | Built-in, simple API, persists across reruns, zero config | In-memory only, lost on session end, single-user, no persistence |
| **SQLite / DuckDB** | Persistent, queryable, handles large data | Overhead for simple use case, file management, serialization |
| **Redis / Memcached** | Fast, shared across processes | External dependency, network calls, overengineered for local app |
| **Global variables** | Simplest possible | Broken by Streamlit's rerun model, not shared across pages |
| **File-based (pickle/parquet)** | Persistent, large data support | Slow I/O, file management, cleanup needed |

## Decision

**Chosen: `st.session_state`** as the sole shared state mechanism.

## Rationale

1. **Native to the framework**: Session state is Streamlit's built-in persistence
   mechanism. It survives page navigation and widget reruns without any external
   infrastructure.

2. **Simple mental model**: Every page can read `st.session_state["df"]` to get
   the current DataFrame and `st.session_state["stats_var_types"]` to get variable
   type assignments. No subscription, no callbacks, no event system.

3. **Atomic updates**: Setting `st.session_state["df"] = new_df` instantly makes
   the updated DataFrame available to all pages on their next render. There is no
   cache invalidation or synchronization problem.

4. **Sufficient for the use case**: The application is single-user, single-session.
   There is no need for cross-session or cross-user state sharing. The 200 MB file
   size limit means DataFrames fit comfortably in memory.

## Implementation

```python
# core/state.py - centralized access
def init_session_state():
    defaults = {
        "df": None,
        "stats_var_types": {},
        "current_theme": "Dark",
        # ... tool-specific state
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_dataframe():
    return st.session_state.get("df", None)

def set_dataframe(df):
    st.session_state["df"] = df
```

### State Keys

| Key | Type | Purpose |
|---|---|---|
| `df` | `DataFrame \| None` | The shared dataset |
| `stats_var_types` | `Dict[str, str]` | Column name to type mapping (Metric/Nominal/Ordinal) |
| `current_theme` | `str` | Active theme ("Light" or "Dark") |
| Tool-specific keys | `varies` | Per-page configuration state |

## Trade-offs Accepted

- **No persistence**: All state is lost when the browser tab closes or the
  Streamlit server restarts. Users must re-upload data each session. This is
  acceptable because the application is a tool, not a workspace - users typically
  complete an analysis in one session.

- **Memory-bound**: The entire DataFrame lives in RAM. With the 200 MB upload
  limit and typical dataset sizes (< 50 MB), this is not a practical limitation,
  but it would be for truly large-scale data work.

- **No undo/redo**: There is no state history. Destructive operations on the
  DataFrame (cleaning, feature engineering) cannot be reversed except by
  re-uploading the original data.

- **No state serialization**: Analysis sessions cannot be saved and resumed.
  This limits reproducibility compared to notebook-based workflows.

## Consequences

- State management code is minimal (~50 lines in `core/state.py`)
- Pages are loosely coupled - they only share state through well-known keys
- The Data Science pipeline naturally shares transformed DataFrames across steps
- Statistics pages can access the same DataFrame without re-uploading
- Performance is bounded by available RAM, not disk I/O

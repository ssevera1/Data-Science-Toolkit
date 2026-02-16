# ADR-010: Chosen Optuna Over Grid/Random Search for Hyperparameter Tuning

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024 |
| **Decision Makers** | Scott Severance |
| **Category** | Technology / ML Pipeline |

## Context

The Hyperparameter Tuning page needs to search for optimal model hyperparameters.
The search strategy significantly impacts both the quality of results (finding
better hyperparameters) and the user experience (time to complete, progress
visibility).

### Options Considered

| Approach | Pros | Cons |
|---|---|---|
| **Grid Search** (sklearn) | Exhaustive, deterministic, simple | Exponential time growth, wasteful for high-dimensional spaces |
| **Random Search** (sklearn) | Better coverage per trial, simple | No learning from previous trials, no convergence |
| **Bayesian Optimization (Optuna)** | Learns from trials, efficient, pruning, visualization | More complex setup, stochastic, requires trial budget |
| **Hyperopt** | Established, Bayesian | Less intuitive API, fewer visualizations, less active development |
| **Ray Tune** | Distributed, many algorithms | Heavy dependency, overengineered for single-machine local app |

## Decision

**Chosen: Optuna** for Bayesian hyperparameter optimization with Tree-structured
Parzen Estimator (TPE).

## Rationale

1. **Sample efficiency**: Optuna's TPE sampler learns from completed trials to
   focus the search on promising regions of the hyperparameter space. For a
   local-only application where compute is bounded by the user's machine, this
   efficiency is critical - 50 Optuna trials often outperform 500 random trials.

2. **Built-in pruning**: Optuna can prune unpromising trials early (e.g., stop
   training a model after 3 cross-validation folds if it's clearly underperforming).
   This further reduces compute time without sacrificing result quality.

3. **Python-native API**: Optuna uses a Pythonic define-by-run API where the
   search space is defined inside the objective function. This integrates cleanly
   with scikit-learn's estimator pattern.

4. **Visualization**: Optuna provides built-in visualization functions
   (optimization history, parameter importance, parallel coordinates) that can
   be rendered during tuning, giving users live feedback on the optimization
   process.

5. **No external infrastructure**: Unlike Ray Tune or distributed Bayesian
   optimization, Optuna runs entirely in-process with no database, no scheduler,
   and no network calls. This aligns with the local-only architecture (ADR-002).

## Trade-offs Accepted

- **Stochastic results**: Bayesian optimization is inherently stochastic. Running
  the same tuning twice may produce different optimal hyperparameters. This is
  acceptable because the goal is finding "good enough" parameters, not
  proving uniqueness.

- **Complexity vs. Grid Search**: The Optuna integration is more complex than
  sklearn's `GridSearchCV`. The objective function, study creation, and trial
  management add code and concepts that users must understand.

- **Trial budget dependency**: Users must choose a number of trials (budget).
  Too few trials may miss the optimum; too many waste time. The application
  provides sensible defaults and guidance, but the trade-off remains.

- **TPE limitations**: TPE works best with moderate-dimensional spaces (< 20
  hyperparameters). For very high-dimensional spaces, random search may be
  competitive. The application's supported algorithms typically have 3-8
  hyperparameters, well within TPE's effective range.

## Consequences

- Hyperparameter tuning is more effective per compute-minute than grid/random search
- Users see live optimization progress and can stop early if satisfied
- The integration adds Optuna as a dependency (~5 MB) but no external services
- Results include parameter importance analysis, helping users understand which hyperparameters matter most

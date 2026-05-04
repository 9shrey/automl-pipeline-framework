# Search Strategy Notes

## Search Space

The search space is intentionally tabular-first. Candidate pipelines combine:

- imputers: median, most-frequent, constant
- scalers: standard, robust, none
- feature selectors: none, mutual information, RFE when budget allows
- estimators: linear baselines, tree ensembles, optional gradient boosting backends

Every trial records the resolved preprocessing choices and estimator parameters in the run leaderboard so a reviewer can reconstruct why the winning pipeline was selected.

## Pruning

ASHA-style pruning should remove trials that underperform early folds or budget slices. The expected artifact shape is represented in [`results/pruning_trace.json`](../results/pruning_trace.json): each row names the trial, model family, partial score, and continue/prune decision.

## Warm Starts

Warm starts use meta-features such as row count, column count, numeric/categorical ratio, missingness, and task type to retrieve prior strong configurations. They are hints, not hard constraints; the sampler can still explore away from stale historical winners.

## Ensemble Selection

The top-k completed pipelines are considered for voting, blending, or stacking. The framework should only keep an ensemble when validation performance improves after accounting for extra complexity and inference cost.

## Benchmark Coverage

The committed `results/` benchmark is deliberately small and deterministic. Serious benchmarking should add multiple OpenML-style datasets, repeated seeds, wall-time budgets, and hardware metadata before making competitive claims.

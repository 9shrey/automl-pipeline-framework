# AutoML Pipeline Framework

A production-grade, modular **AutoML** library for tabular data. Given `(X, y, task)`, it jointly searches over preprocessing, feature selection, algorithm choice, and hyperparameters using **Bayesian optimization (Optuna / TPE)** with **ASHA pruning**, **warm-starts** from a meta-learning store of historical top configurations, **ensembles** the top-k pipelines via stacking/voting/blending, and **explains** the result with **SHAP** + permutation importance.

> Status: **end-to-end working** — search → ensemble → explain → record → warm-start. 200+ unit tests, sklearn-compatible API, typer CLI. See [`MASTER_PROMPT.md`](MASTER_PROMPT.md) for the full spec.

## 60-second quickstart

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from automl.api import AutoMLClassifier

X_arr, y = make_classification(n_samples=400, n_features=8, random_state=0)
X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(8)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, stratify=y)

auto = AutoMLClassifier(
    target="y",
    time_budget_seconds=60,
    n_trials=10,
    seed=42,
)
auto.fit(X_tr, y_tr)

print(f"best CV {auto.metric_name_}: {auto.best_score_:.4f}")
print(f"test accuracy:        {auto.score(X_te, y_te):.4f}")
print(f"run artifacts:        {auto.run_dir_}")
print(auto.leaderboard_.head())
```

Or use a YAML config:

```python
from automl.api import AutoMLClassifier
auto = AutoMLClassifier(config="configs/fast.yaml").fit(X, y)
```

## CLI

```bash
# Fit + record a run
automl fit --config configs/fast.yaml --data train.csv --target label --out runs/

# Inspect, score, predict from a saved run
automl runs        --out-dir runs/
automl leaderboard --run runs/<RUN_ID> --top 10
automl score       --run runs/<RUN_ID> --data test.csv  --target label
automl predict     --run runs/<RUN_ID> --data new.csv   --out preds.csv
```

## Run-directory layout

Every `fit` creates a UTC-timestamped directory with:

| file | contents |
|---|---|
| `config.yaml`        | resolved `AutoMLConfig` snapshot |
| `dataset.json`       | shape + BLAKE2b hash + schema summary |
| `leaderboard.csv`    | every completed trial, best first |
| `best_pipeline.pkl`  | the deployable sklearn pipeline (or ensemble) |
| `best_trial.json`    | best trial's params + score + user attrs |
| `run_card.md`        | human summary (git SHA, top-10 leaderboard, wall time) |
| `explain.md` + CSVs  | feature importance (when `explain.enabled`) |

## Architecture

See [`docs/architecture.md`](docs/architecture.md). Core flow:

```
User → Orchestrator → (Meta-store warm-start) → Optuna TPE + ASHA
     → Pipeline Factory (preproc + feat-sel + estimator)
     → CV Evaluator → Ensembler (stacking) → SHAP/Permutation Explain
     → Run Recorder (config + data hash + git SHA + run_card.md)
```

## Install

```bash
make bootstrap          # uv venv + editable install + dev deps
make test               # unit tests + coverage
```

## Keywords

AutoML · Hyperparameter Optimization · Bayesian Optimization · Optuna · TPE · Hyperband · ASHA · scikit-learn · XGBoost · LightGBM · CatBoost · Stacking · Voting · Blending · Meta-Learning · Warm Starting · Feature Selection · RFE · SHAP · Permutation Importance · Pipeline Design · Cross-Validation · Reproducibility · Experiment Tracking

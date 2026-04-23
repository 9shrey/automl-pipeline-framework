"""Optuna Study orchestration: build study, honor budget, return leaderboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from automl.config.schema import AutoMLConfig
from automl.data.schema_infer import ColumnSchema, infer_schema
from automl.evaluation.metrics import DEFAULT_METRIC, get_metric
from automl.search.objective import Objective
from automl.search.pruners import build_pruner
from automl.search.samplers import build_sampler

__all__ = ["SearchResult", "run_search"]


@dataclass
class SearchResult:
    study: optuna.Study
    leaderboard: pd.DataFrame
    best_trial: optuna.trial.FrozenTrial
    metric_name: str


def _build_leaderboard(study: optuna.Study, metric_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric = get_metric(metric_name)
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        # Undo sign-flip so the leaderboard is in original metric units.
        score = metric.signed * float(t.value if t.value is not None else np.nan)
        rows.append(
            {
                "trial_number": t.number,
                "score": score,
                "model": t.user_attrs.get("model_name"),
                "imputation": t.user_attrs.get("imputation"),
                "encoding": t.user_attrs.get("encoding"),
                "scaling": t.user_attrs.get("scaling"),
                "feature_selection": t.user_attrs.get("feature_selection"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ascending = metric.direction == "min"
    return df.sort_values("score", ascending=ascending).reset_index(drop=True)


def run_search(
    config: AutoMLConfig,
    X: pd.DataFrame,
    y: Any,
    *,
    schema: ColumnSchema | None = None,
    metric: str | None = None,
    show_progress_bar: bool = False,
    warm_start_params: list[dict[str, Any]] | None = None,
) -> SearchResult:
    """Run an Optuna study under ``config``'s budgets and return a ``SearchResult``."""
    if schema is None:
        schema = infer_schema(X)
    metric_name = metric or DEFAULT_METRIC[config.task]

    sampler = build_sampler(config.search.sampler, seed=config.seed)
    pruner = build_pruner(config.search.pruner, n_splits=config.cv.n_splits)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    if warm_start_params:
        for params in warm_start_params:
            try:
                study.enqueue_trial(params, skip_if_exists=True)
            except (ValueError, RuntimeError):
                continue

    objective = Objective(config, X, np.asarray(y), schema=schema, metric=metric_name)
    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.time_budget_seconds,
        show_progress_bar=show_progress_bar,
        catch=(Exception,),  # failure isolation per Working Agreement
    )

    leaderboard = _build_leaderboard(study, metric_name)
    if not study.best_trials:
        raise RuntimeError("Search produced no completed trials.")
    return SearchResult(
        study=study,
        leaderboard=leaderboard,
        best_trial=study.best_trial,
        metric_name=metric_name,
    )

"""Ensembling subpackage: stacking / voting + top-k selector + factory.

The high-level ``build_ensemble`` function takes the leaderboard + study
produced by ``run_search`` plus the ``EnsemblingConfig`` and returns a fitted
estimator, or ``None`` if ensembling is disabled / cannot be assembled (e.g.
fewer than 2 base pipelines available).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd

from automl.config.schema import EnsemblingConfig, Task
from automl.data.schema_infer import ColumnSchema
from automl.ensembling.selector import SelectedPipeline, select_top_k
from automl.ensembling.stacking import build_stacking_ensemble
from automl.ensembling.voting import build_voting_ensemble
from automl.search.objective import SampledPipelineSpec, build_pipeline_from_params

__all__ = [
    "SelectedPipeline",
    "build_ensemble",
    "build_stacking_ensemble",
    "build_voting_ensemble",
    "select_top_k",
]


def _spec_from_selected(
    sel: SelectedPipeline,
    *,
    task: str,
) -> SampledPipelineSpec:
    from automl.api import _extract_model_params  # local import to avoid cycle

    model_params = _extract_model_params(sel.params, sel.model_name, task)
    return SampledPipelineSpec(
        imputation=sel.imputation,
        encoding=sel.encoding,
        scaling=sel.scaling,
        feature_selection=sel.feature_selection,
        model_name=sel.model_name,
        model_params=model_params,
    )


def build_ensemble(
    *,
    config: EnsemblingConfig,
    task: Task,
    leaderboard: pd.DataFrame,
    study: optuna.Study,
    schema: ColumnSchema,
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 0,
) -> Any | None:
    """Build + fit an ensemble of the top-k pipelines, or return ``None``."""
    if not config.enabled or config.strategy == "none":
        return None
    selected = select_top_k(
        leaderboard,
        study,
        k=config.top_k,
        strategy=config.diversity,
        seed=seed,
    )
    if len(selected) < 2:
        # Nothing to ensemble — caller falls back to the best single pipeline.
        return None

    estimators: list[tuple[str, Any]] = []
    for sel in selected:
        spec = _spec_from_selected(sel, task=task)
        pipe = build_pipeline_from_params(spec, schema=schema, task=task)
        estimators.append((f"t{sel.trial_number}_{sel.model_name}", pipe))

    if config.strategy == "voting":
        ensemble = build_voting_ensemble(estimators, task=task, voting="soft")
    elif config.strategy == "stacking":
        ensemble = build_stacking_ensemble(
            estimators, task=task, meta_learner=config.meta_learner, cv=3
        )
    elif config.strategy == "blending":
        # Blending = stacking with a held-out fold; we don't have a true holdout
        # plumbed here, so fall back to stacking with cv=2 for now.
        ensemble = build_stacking_ensemble(
            estimators, task=task, meta_learner=config.meta_learner, cv=2
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown ensembling strategy: {config.strategy!r}")

    ensemble.fit(X, y)
    return ensemble

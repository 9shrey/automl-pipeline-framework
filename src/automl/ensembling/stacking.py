"""Stacking ensembles with a configurable meta-learner."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from automl.config.schema import MetaLearner, Task

__all__ = ["build_stacking_ensemble"]


def _build_meta_learner(meta: MetaLearner, task: Task) -> Any:
    if meta == "logreg":
        if task == "classification":
            return LogisticRegression(max_iter=1000)
        return Ridge()
    if meta == "ridge":
        if task == "regression":
            return Ridge()
        return LogisticRegression(max_iter=1000)
    if meta == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore[import-not-found]
        except ImportError:
            return LogisticRegression(max_iter=1000) if task == "classification" else Ridge()
        return LGBMClassifier() if task == "classification" else LGBMRegressor()
    raise ValueError(f"Unknown meta_learner: {meta!r}")


def build_stacking_ensemble(
    estimators: list[tuple[str, Pipeline]],
    *,
    task: Task,
    meta_learner: MetaLearner = "logreg",
    cv: int = 3,
) -> Any:
    if not estimators:
        raise ValueError("stacking ensemble needs >=1 estimator.")
    final = _build_meta_learner(meta_learner, task)
    if task == "classification":
        return StackingClassifier(
            estimators=list(estimators),
            final_estimator=final,
            cv=cv,
            n_jobs=1,
            passthrough=False,
        )
    return StackingRegressor(
        estimators=list(estimators),
        final_estimator=final,
        cv=cv,
        n_jobs=1,
        passthrough=False,
    )

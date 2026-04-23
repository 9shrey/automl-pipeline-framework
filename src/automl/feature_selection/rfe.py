"""Recursive feature elimination wrapper (task-aware, tree-based)."""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.feature_selection import RFE

from automl.config.schema import Task

__all__ = ["build_rfe_selector"]


def build_rfe_selector(task: Task, n_features_to_select: int | float | None = None) -> RFE:
    if task == "classification":
        base = GradientBoostingClassifier(n_estimators=50, random_state=0)
    elif task == "regression":
        base = GradientBoostingRegressor(n_estimators=50, random_state=0)
    else:
        raise ValueError(f"Unknown task '{task}'.")
    return RFE(estimator=base, n_features_to_select=n_features_to_select, step=0.1)

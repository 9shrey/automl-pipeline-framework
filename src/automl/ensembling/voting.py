"""Soft / hard voting ensembles built from selected sklearn pipelines."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.pipeline import Pipeline

from automl.config.schema import Task

__all__ = ["build_voting_ensemble"]


def build_voting_ensemble(
    estimators: list[tuple[str, Pipeline]],
    *,
    task: Task,
    voting: str = "soft",
) -> Any:
    if not estimators:
        raise ValueError("voting ensemble needs >=1 estimator.")
    if task == "classification":
        # Fall back to hard voting if any base lacks predict_proba.
        eligible = voting == "soft" and all(
            hasattr(p, "predict_proba") for _, p in estimators
        )
        return VotingClassifier(
            estimators=list(estimators),
            voting="soft" if eligible else "hard",
            n_jobs=1,
        )
    return VotingRegressor(estimators=list(estimators), n_jobs=1)

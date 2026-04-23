"""Mutual-information selector wrapper (task-aware)."""

from __future__ import annotations

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)

from automl.config.schema import Task

__all__ = ["build_mutual_info_selector"]


def build_mutual_info_selector(task: Task, k: int | str = "all"):
    score_func = mutual_info_classif if task == "classification" else mutual_info_regression
    return SelectKBest(score_func=score_func, k=k)

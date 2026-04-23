"""Metric registry: name → callable, with task + direction metadata.

Direction ``"max"`` means "higher is better"; the search engine flips the sign
for minimization metrics so Optuna can always maximize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from automl.config.schema import Task

__all__ = [
    "DEFAULT_METRIC",
    "METRICS",
    "Metric",
    "available_metrics",
    "get_metric",
]

Direction = Literal["max", "min"]


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[..., float]
    task: Task
    direction: Direction
    needs_proba: bool = False

    def score(self, y_true: Any, y_pred: Any, **kwargs: Any) -> float:
        return float(self.fn(y_true, y_pred, **kwargs))

    @property
    def signed(self) -> int:
        """+1 if higher-is-better, -1 if lower-is-better."""
        return 1 if self.direction == "max" else -1


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


METRICS: dict[str, Metric] = {
    # Classification
    "accuracy":  Metric("accuracy",  accuracy_score,    "classification", "max"),
    "f1":        Metric("f1",        lambda y, p: f1_score(y, p, average="binary", zero_division=0), "classification", "max"),
    "f1_macro":  Metric("f1_macro",  lambda y, p: f1_score(y, p, average="macro",  zero_division=0), "classification", "max"),
    "roc_auc":   Metric("roc_auc",   roc_auc_score,     "classification", "max", needs_proba=True),
    "logloss":   Metric("logloss",   log_loss,          "classification", "min", needs_proba=True),
    # Regression
    "rmse":      Metric("rmse",      _rmse,             "regression", "min"),
    "mae":       Metric("mae",       mean_absolute_error, "regression", "min"),
    "r2":        Metric("r2",        r2_score,            "regression", "max"),
}

DEFAULT_METRIC: dict[Task, str] = {
    "classification": "roc_auc",
    "regression": "rmse",
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {sorted(METRICS)}"
        ) from exc


def available_metrics(task: Task | None = None) -> list[str]:
    if task is None:
        return sorted(METRICS)
    return sorted(n for n, m in METRICS.items() if m.task == task)

"""Score a fitted estimator on a held-out fold using a ``Metric``.

The scorer hides ``predict`` vs ``predict_proba`` selection (some metrics need
class probabilities, others need predicted labels) so the search objective can
stay metric-agnostic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from automl.evaluation.metrics import Metric, get_metric

__all__ = ["score_estimator", "score_predictions"]


def _predict_for_metric(estimator: BaseEstimator, X: Any, metric: Metric) -> np.ndarray:
    if metric.needs_proba:
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(
                f"Metric '{metric.name}' requires predict_proba but estimator "
                f"{type(estimator).__name__} does not provide it."
            )
        proba = estimator.predict_proba(X)
        # Binary case: many metrics expect a 1-D probability of the positive class.
        if metric.name == "roc_auc" and proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
    return estimator.predict(X)


def score_estimator(
    estimator: BaseEstimator,
    X: Any,
    y_true: Any,
    metric: str | Metric,
) -> float:
    """Compute ``metric`` for a fitted estimator on ``(X, y_true)``."""
    m = metric if isinstance(metric, Metric) else get_metric(metric)
    y_pred = _predict_for_metric(estimator, X, m)
    return m.score(y_true, y_pred)


def score_predictions(y_true: Any, y_pred: Any, metric: str | Metric) -> float:
    """Compute ``metric`` from already-computed predictions."""
    m = metric if isinstance(metric, Metric) else get_metric(metric)
    return m.score(y_true, y_pred)

"""Evaluation subpackage: CV splitters, metrics, multi-metric scorer."""

from automl.evaluation.cv import build_splitter, split_indices
from automl.evaluation.metrics import DEFAULT_METRIC, METRICS, Metric, available_metrics, get_metric
from automl.evaluation.scoring import score_estimator, score_predictions

__all__ = [
    "DEFAULT_METRIC",
    "METRICS",
    "Metric",
    "available_metrics",
    "build_splitter",
    "get_metric",
    "score_estimator",
    "score_predictions",
    "split_indices",
]

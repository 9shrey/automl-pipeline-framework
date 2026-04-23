"""Feature-selection factory: maps a name → unfitted sklearn-compatible step."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import FunctionTransformer

from automl.config.schema import Task
from automl.feature_selection.mutual_info import build_mutual_info_selector
from automl.feature_selection.rfe import build_rfe_selector
from automl.feature_selection.shap_select import build_shap_selector

__all__ = ["FEATURE_SELECTORS", "build_feature_selector"]

FEATURE_SELECTORS = ("none", "variance", "mutual_info", "rfe", "shap")


def build_feature_selector(
    name: str,
    *,
    task: Task,
    k: int | float | str = "all",
    variance_threshold: float = 0.0,
    rfe_n_features: int | float | None = None,
) -> BaseEstimator:
    """Return an unfitted feature selector."""
    if name == "none":
        return FunctionTransformer(validate=False)
    if name == "variance":
        return VarianceThreshold(threshold=variance_threshold)
    if name == "mutual_info":
        return build_mutual_info_selector(task=task, k=k)
    if name == "rfe":
        return build_rfe_selector(task=task, n_features_to_select=rfe_n_features)
    if name == "shap":
        return build_shap_selector(task=task, k=k)
    raise ValueError(f"Unknown feature selector '{name}'. Available: {FEATURE_SELECTORS}")

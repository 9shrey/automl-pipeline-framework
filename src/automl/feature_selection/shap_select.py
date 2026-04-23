"""SHAP-based feature importance selector.

Gated by the optional ``shap`` dependency. The selector fits a fast tree model,
computes mean ``|SHAP|`` per feature, and keeps the top ``k``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from automl.config.schema import Task

__all__ = ["ShapSelector", "build_shap_selector"]


class ShapSelector(BaseEstimator, TransformerMixin):
    """Keep top-``k`` features by mean(|SHAP value|) of a quick tree model."""

    def __init__(self, task: Task = "classification", k: int | float | str = "all") -> None:
        self.task = task
        self.k = k

    def fit(self, X, y=None):  # type: ignore[no-untyped-def]
        try:
            import shap
        except ImportError as exc:  # pragma: no cover - env-specific
            raise ImportError(
                "ShapSelector requires the optional `shap` package. "
                "Install with `pip install shap`."
            ) from exc
        if self.task == "classification":
            from sklearn.ensemble import GradientBoostingClassifier as _Tree
        else:
            from sklearn.ensemble import GradientBoostingRegressor as _Tree
        X_arr = np.asarray(X)
        model = _Tree(n_estimators=50, random_state=0).fit(X_arr, y)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_arr)
        if isinstance(values, list):  # multi-class case → average across classes
            values = np.mean([np.abs(v) for v in values], axis=0)
        importances = np.abs(values).mean(axis=0)
        n_features = X_arr.shape[1]
        if self.k == "all":
            keep = n_features
        elif isinstance(self.k, float):
            keep = max(1, int(round(self.k * n_features)))
        else:
            keep = max(1, min(int(self.k), n_features))
        order = np.argsort(importances)[::-1]
        self.selected_indices_ = np.sort(order[:keep])
        self.n_features_in_ = n_features
        return self

    def transform(self, X):  # type: ignore[no-untyped-def]
        X_arr = np.asarray(X)
        return X_arr[:, self.selected_indices_]


def build_shap_selector(task: Task, k: int | float | str = "all") -> ShapSelector:
    return ShapSelector(task=task, k=k)

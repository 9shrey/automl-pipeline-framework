"""Imputer factory: ``mean``, ``median``, ``most_frequent``, ``iterative``, ``knn``."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

__all__ = ["IMPUTERS", "build_imputer"]

IMPUTERS = ("mean", "median", "most_frequent", "iterative", "knn")


def build_imputer(name: str, *, kind: str = "numeric") -> BaseEstimator:
    """Return an unfitted imputer.

    ``kind`` is ``"numeric"`` or ``"categorical"``. For categorical columns we
    only allow ``most_frequent`` (the others are numeric-only).
    """
    if kind == "categorical":
        if name not in {"most_frequent"}:
            return SimpleImputer(strategy="most_frequent")
        return SimpleImputer(strategy="most_frequent")
    if name == "mean":
        return SimpleImputer(strategy="mean")
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "most_frequent":
        return SimpleImputer(strategy="most_frequent")
    if name == "iterative":
        return IterativeImputer(max_iter=10, random_state=0)
    if name == "knn":
        return KNNImputer(n_neighbors=5)
    raise ValueError(f"Unknown imputer '{name}'. Available: {IMPUTERS}")

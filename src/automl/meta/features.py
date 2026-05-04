"""Cheap dataset meta-features used to key the warm-start store.

We stay in the simple-statistics regime: number of rows / cols, ratio of
numeric vs categorical columns, missing-value rate, target balance / variance,
and class count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from automl.config.schema import Task
from automl.data.schema_infer import ColumnSchema, infer_schema

__all__ = ["MetaFeatures", "compute_meta_features", "meta_feature_distance"]


@dataclass(frozen=True)
class MetaFeatures:
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    n_boolean: int
    pct_missing: float
    n_classes: int
    class_balance: float
    target_std: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_vector(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=float)


def compute_meta_features(
    X: pd.DataFrame,
    y: Any,
    *,
    task: Task,
    schema: ColumnSchema | None = None,
) -> MetaFeatures:
    if schema is None:
        schema = infer_schema(X)
    n_rows, n_cols = int(X.shape[0]), int(X.shape[1])
    pct_missing = float(X.isna().to_numpy().mean()) if n_rows and n_cols else 0.0
    y_arr = np.asarray(y)
    if task == "classification":
        classes, counts = np.unique(y_arr, return_counts=True)
        n_classes = len(classes)
        if n_classes > 1:
            p = counts / counts.sum()
            entropy = float(-np.sum(p * np.log(p)))
            class_balance = float(entropy / np.log(n_classes))
        else:
            class_balance = 0.0
        target_std = 0.0
    else:
        n_classes = 0
        class_balance = 0.0
        target_std = float(np.std(y_arr.astype(float))) if y_arr.size else 0.0
    return MetaFeatures(
        n_rows=n_rows,
        n_cols=n_cols,
        n_numeric=len(schema.numeric),
        n_categorical=len(schema.categorical),
        n_boolean=len(schema.boolean),
        pct_missing=pct_missing,
        n_classes=n_classes,
        class_balance=class_balance,
        target_std=target_std,
    )


def meta_feature_distance(a: MetaFeatures, b: MetaFeatures) -> float:
    """Scale-invariant L1 distance: each feature normalized by max(|a|, |b|, 1)."""
    va, vb = a.as_vector(), b.as_vector()
    denom = np.maximum(np.maximum(np.abs(va), np.abs(vb)), 1.0)
    return float(np.sum(np.abs(va - vb) / denom))


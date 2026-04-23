"""Sklearn permutation-importance wrapper that returns a tidy DataFrame."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

__all__ = ["permutation_feature_importance"]


def permutation_feature_importance(
    estimator: Any,
    X: pd.DataFrame,
    y: Any,
    *,
    n_repeats: int = 10,
    seed: int = 0,
    scoring: str | None = None,
) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    result = permutation_importance(
        estimator,
        X,
        np.asarray(y),
        n_repeats=n_repeats,
        random_state=seed,
        scoring=scoring,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


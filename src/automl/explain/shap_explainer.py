"""SHAP explainer wrapper. Optional dep — gracefully degrades."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["shap_feature_importance", "shap_available"]


def shap_available() -> bool:
    try:
        import shap  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def shap_feature_importance(
    estimator: Any,
    X: pd.DataFrame,
    *,
    max_samples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Return mean(|SHAP|) per *input* feature, ranked desc.

    For sklearn pipelines we attribute on the raw input columns and use the
    permutation-style ``shap.Explainer(model.predict, ...)`` fallback so we do
    not need to introspect the preprocessor. ``TreeExplainer`` is faster but
    requires a tree model and pre-encoded inputs.
    """
    if not shap_available():
        raise ImportError("shap is not installed; install `shap` to enable this report.")
    import shap  # type: ignore[import-not-found]

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X

    predict_fn = getattr(estimator, "predict_proba", None) or estimator.predict
    explainer = shap.Explainer(predict_fn, X_sample, seed=seed)
    sv = explainer(X_sample)
    values = sv.values
    if values.ndim == 3:  # (n, n_features, n_classes)
        values = np.abs(values).mean(axis=2)
    importance = np.abs(values).mean(axis=0)
    df = pd.DataFrame({"feature": list(X.columns), "shap_importance": importance})
    return df.sort_values("shap_importance", ascending=False).reset_index(drop=True)


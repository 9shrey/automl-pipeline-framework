"""Encoder factory: ``onehot``, ``ordinal``, ``frequency``.

``target`` encoding is intentionally not implemented in this batch — it must
fit inside CV folds to avoid leakage, which the search engine handles by
placing the encoder *inside* the per-fold pipeline. A real ``TargetEncoder``
adapter lands with the search engine.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

__all__ = ["ENCODERS", "FrequencyEncoder", "build_encoder"]

ENCODERS = ("onehot", "ordinal", "frequency", "target")


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its training-set frequency.

    Unknown categories at transform time are encoded as 0.0.
    """

    def fit(self, X, y=None):  # type: ignore[no-untyped-def]
        X_arr = np.asarray(X, dtype=object)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.n_features_in_ = X_arr.shape[1]
        self.frequencies_: list[dict] = []
        for j in range(self.n_features_in_):
            col = X_arr[:, j]
            unique, counts = np.unique(col.astype(str), return_counts=True)
            total = float(counts.sum()) or 1.0
            self.frequencies_.append(
                {str(u): float(c) / total for u, c in zip(unique, counts)}
            )
        return self

    def transform(self, X):  # type: ignore[no-untyped-def]
        X_arr = np.asarray(X, dtype=object)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"FrequencyEncoder fit on {self.n_features_in_} cols, got {X_arr.shape[1]}."
            )
        out = np.zeros(X_arr.shape, dtype=float)
        for j, lookup in enumerate(self.frequencies_):
            col = X_arr[:, j].astype(str)
            out[:, j] = np.fromiter((lookup.get(v, 0.0) for v in col), dtype=float, count=col.size)
        return out


def build_encoder(name: str) -> BaseEstimator:
    """Return an unfitted encoder for categorical columns."""
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if name == "frequency":
        return FrequencyEncoder()
    if name == "target":
        raise NotImplementedError(
            "TargetEncoder is gated by the search engine to keep encoding inside CV folds."
        )
    raise ValueError(f"Unknown encoder '{name}'. Available: {ENCODERS}")

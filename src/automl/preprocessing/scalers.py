"""Scaler factory: ``standard``, ``robust``, ``minmax``, ``quantile``, ``none``."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

__all__ = ["SCALERS", "build_scaler"]

SCALERS = ("standard", "robust", "minmax", "quantile", "none")


def build_scaler(name: str) -> BaseEstimator:
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=0)
    if name == "none":
        return FunctionTransformer(validate=False)
    raise ValueError(f"Unknown scaler '{name}'. Available: {SCALERS}")

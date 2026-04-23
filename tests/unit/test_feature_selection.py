"""Unit tests for ``automl.feature_selection``."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from automl.feature_selection import FEATURE_SELECTORS, build_feature_selector


@pytest.mark.parametrize("name", ["none", "variance", "mutual_info", "rfe"])
def test_classification_selectors_fit_and_reduce(name: str) -> None:
    X, y = make_classification(
        n_samples=120, n_features=10, n_informative=5, random_state=0
    )
    kw = {"task": "classification"}
    if name == "mutual_info":
        kw["k"] = 5
    elif name == "rfe":
        kw["rfe_n_features"] = 5
    sel = build_feature_selector(name, **kw)
    out = sel.fit_transform(X, y)
    assert out.shape[0] == X.shape[0]
    assert 1 <= out.shape[1] <= X.shape[1]


@pytest.mark.parametrize("name", ["none", "variance", "mutual_info", "rfe"])
def test_regression_selectors_fit_and_reduce(name: str) -> None:
    X, y = make_regression(
        n_samples=120, n_features=10, n_informative=5, noise=0.1, random_state=0
    )
    kw = {"task": "regression"}
    if name == "mutual_info":
        kw["k"] = 5
    elif name == "rfe":
        kw["rfe_n_features"] = 5
    sel = build_feature_selector(name, **kw)
    out = sel.fit_transform(X, y)
    assert out.shape[0] == X.shape[0]


def test_variance_drops_constant_columns() -> None:
    X = np.column_stack([np.random.RandomState(0).randn(50, 3), np.zeros(50)])
    sel = build_feature_selector("variance", task="classification")
    y = np.random.RandomState(0).randint(0, 2, 50)
    out = sel.fit_transform(X, y)
    assert out.shape[1] == 3


def test_unknown_selector_raises() -> None:
    with pytest.raises(ValueError, match="Unknown feature selector"):
        build_feature_selector("nonsense", task="classification")


def test_feature_selectors_constant() -> None:
    assert "shap" in FEATURE_SELECTORS
    assert "rfe" in FEATURE_SELECTORS

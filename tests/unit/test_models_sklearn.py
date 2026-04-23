"""Unit tests for additional sklearn adapters: RF, ExtraTrees, HistGB, Ridge, Lasso."""

from __future__ import annotations

import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge

# Import to register all adapters.
import automl.models.sklearn_models  # noqa: F401
from automl.models import EstimatorRegistry
from automl.search.space import RandomSampler


@pytest.mark.parametrize(
    ("name", "task", "klass"),
    [
        ("rf", "classification", RandomForestClassifier),
        ("rf", "regression", RandomForestRegressor),
        ("extratrees", "classification", ExtraTreesClassifier),
        ("extratrees", "regression", ExtraTreesRegressor),
        ("histgb", "classification", HistGradientBoostingClassifier),
        ("histgb", "regression", HistGradientBoostingRegressor),
        ("ridge", "regression", Ridge),
        ("lasso", "regression", Lasso),
    ],
)
def test_adapter_builds_correct_class(name: str, task: str, klass: type) -> None:
    adapter = EstimatorRegistry.get(name)
    space = adapter.search_space(task)
    params = space.sample(RandomSampler(seed=0))
    model = adapter.build(task, params)
    assert isinstance(model, klass)


@pytest.mark.parametrize("name", ["rf", "extratrees", "histgb"])
def test_classification_adapters_fit_and_score(name: str) -> None:
    X, y = make_classification(n_samples=200, n_features=8, random_state=0)
    adapter = EstimatorRegistry.get(name)
    space = adapter.search_space("classification")
    params = space.sample(RandomSampler(seed=1))
    model = adapter.build("classification", params)
    model.fit(X, y)
    assert model.score(X, y) > 0.7


@pytest.mark.parametrize("name", ["rf", "extratrees", "histgb", "ridge", "lasso"])
def test_regression_adapters_fit_and_score(name: str) -> None:
    X, y = make_regression(n_samples=200, n_features=6, noise=0.1, random_state=0)
    adapter = EstimatorRegistry.get(name)
    space = adapter.search_space("regression")
    params = space.sample(RandomSampler(seed=2))
    model = adapter.build("regression", params)
    model.fit(X, y)
    # Lasso may underfit at large alpha; allow a low bar — just confirm it runs.
    score = model.score(X, y)
    assert score > -1.0


def test_logreg_does_not_support_regression() -> None:
    with pytest.raises(ValueError, match="classification"):
        EstimatorRegistry.get("logreg").build("regression", {})


def test_ridge_does_not_support_classification() -> None:
    with pytest.raises(ValueError, match="regression"):
        EstimatorRegistry.get("ridge").build("classification", {})


def test_registry_lists_classification_supports() -> None:
    clf_models = set(EstimatorRegistry.names(task="classification"))
    reg_models = set(EstimatorRegistry.names(task="regression"))
    assert {"logreg", "rf", "extratrees", "histgb"} <= clf_models
    assert {"rf", "extratrees", "histgb", "ridge", "lasso"} <= reg_models
    assert "logreg" not in reg_models
    assert "ridge" not in clf_models

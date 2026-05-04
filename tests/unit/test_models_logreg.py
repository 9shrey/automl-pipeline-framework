"""Unit tests for ``automl.models.base`` + registry + LogReg adapter."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from automl.models import EstimatorAdapter, EstimatorRegistry, register_estimator
from automl.models.sklearn_models import LogReg  # noqa: F401  (registers "logreg")
from automl.search.space import RandomSampler, SearchSpace

# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


class TestEstimatorAdapterContract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            EstimatorAdapter()  # type: ignore[abstract]

    def test_supports_dispatch(self) -> None:
        logreg = EstimatorRegistry.get("logreg")
        assert logreg.supports("classification") is True
        assert logreg.supports("regression") is False

    def test_supports_unknown_task_raises(self) -> None:
        logreg = EstimatorRegistry.get("logreg")
        with pytest.raises(ValueError, match="Unknown task"):
            logreg.supports("clustering")  # type: ignore[arg-type]

    def test_repr_contains_name(self) -> None:
        assert "logreg" in repr(EstimatorRegistry.get("logreg"))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_logreg_is_registered(self) -> None:
        assert "logreg" in EstimatorRegistry.names()
        assert "logreg" in EstimatorRegistry.names(task="classification")
        assert "logreg" not in EstimatorRegistry.names(task="regression")

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not registered"):
            EstimatorRegistry.get("does_not_exist")

    def test_decorator_registers_and_unregister_cleans_up(self) -> None:
        @register_estimator("toy_for_test")
        class Toy(EstimatorAdapter):
            supports_classification = True

            def search_space(self, task):
                return SearchSpace([])

            def build(self, task, params):
                return LogisticRegression()

        try:
            assert "toy_for_test" in EstimatorRegistry.names()
            inst = EstimatorRegistry.get("toy_for_test")
            assert inst.name == "toy_for_test"
        finally:
            EstimatorRegistry.unregister("toy_for_test")
        assert "toy_for_test" not in EstimatorRegistry.names()

    def test_duplicate_registration_rejected(self) -> None:
        with pytest.raises(ValueError, match="already registered"):

            @register_estimator("logreg")
            class Dupe(EstimatorAdapter):  # pragma: no cover - registration fails
                supports_classification = True

                def search_space(self, task):
                    return SearchSpace([])

                def build(self, task, params):
                    return LogisticRegression()

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EstimatorRegistry.register("", EstimatorRegistry.get("logreg"))


# ---------------------------------------------------------------------------
# LogReg adapter
# ---------------------------------------------------------------------------


class TestLogReg:
    def setup_method(self) -> None:
        self.adapter = EstimatorRegistry.get("logreg")

    def test_search_space_only_for_classification(self) -> None:
        with pytest.raises(ValueError, match="classification"):
            self.adapter.search_space("regression")

    def test_search_space_samples_yield_compatible_solver(self) -> None:
        space = self.adapter.search_space("classification")
        sampler = RandomSampler(seed=0)
        valid_pairs = {
            ("l2", "lbfgs"), ("l2", "liblinear"), ("l2", "saga"),
            ("l1", "liblinear"), ("l1", "saga"),
            ("elasticnet", "saga"),
            ("none", "lbfgs"), ("none", "saga"),
        }
        for _ in range(100):
            params = space.sample(sampler)
            pair = params["logreg_penalty_solver"]
            assert tuple(pair) in valid_pairs
            penalty, _solver = pair
            if penalty == "elasticnet":
                assert "logreg_l1_ratio" in params
                assert 0.0 <= params["logreg_l1_ratio"] <= 1.0
            else:
                assert "logreg_l1_ratio" not in params
            assert 1e-4 <= params["logreg_C"] <= 1e4
            assert 100 <= params["logreg_max_iter"] <= 2000

    def test_build_returns_logistic_regression(self) -> None:
        space = self.adapter.search_space("classification")
        params = space.sample(RandomSampler(seed=42))
        model = self.adapter.build("classification", params)
        assert isinstance(model, LogisticRegression)

    def test_build_rejects_regression(self) -> None:
        with pytest.raises(ValueError, match="classification"):
            self.adapter.build("regression", {})

    def test_built_model_fits_and_predicts(self) -> None:
        # Use a deterministic, easy-to-fit param combination.
        params = {
            "logreg_C": 1.0,
            "logreg_penalty_solver": ("l2", "lbfgs"),
            "logreg_max_iter": 500,
        }
        model = self.adapter.build("classification", params)
        X, y = make_classification(n_samples=200, n_features=10, random_state=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert set(np.unique(preds)) <= {0, 1}
        assert model.score(X, y) > 0.7

    def test_none_penalty_passes_through_as_none(self) -> None:
        params = {
            "logreg_C": 1.0,
            "logreg_penalty_solver": ("none", "lbfgs"),
            "logreg_max_iter": 200,
        }
        model = self.adapter.build("classification", params)
        assert model.penalty is None

    def test_elasticnet_passes_l1_ratio(self) -> None:
        params = {
            "logreg_C": 1.0,
            "logreg_penalty_solver": ("elasticnet", "saga"),
            "logreg_l1_ratio": 0.3,
            "logreg_max_iter": 200,
        }
        model = self.adapter.build("classification", params)
        assert model.l1_ratio == 0.3
        assert model.penalty == "elasticnet"

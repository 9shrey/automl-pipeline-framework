"""sklearn estimator adapters. First concrete adapter: ``LogReg``.

More adapters (RandomForest, ExtraTrees, HistGradientBoosting, Ridge, ...) land
in subsequent commits per the Working Agreement.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

from automl.models.base import EstimatorAdapter, Task
from automl.models.registry import register_estimator
from automl.search.space import Categorical, Conditional, Float, Int, SearchSpace

__all__ = ["ExtraTrees", "HistGB", "LassoAdapter", "LogReg", "RandomForest", "RidgeAdapter"]


# Valid (penalty, solver) pairs per sklearn's compatibility matrix:
#   l2         → lbfgs / liblinear / saga
#   l1         → liblinear / saga
#   elasticnet → saga only (l1_ratio required)
#   none       → lbfgs / saga
_LOGREG_PENALTY_SOLVER: tuple[tuple[str, str], ...] = (
    ("l2", "lbfgs"),
    ("l2", "liblinear"),
    ("l2", "saga"),
    ("l1", "liblinear"),
    ("l1", "saga"),
    ("elasticnet", "saga"),
    ("none", "lbfgs"),
    ("none", "saga"),
)
_LOGREG_ELASTICNET_PAIRS: tuple[tuple[str, str], ...] = tuple(
    ps for ps in _LOGREG_PENALTY_SOLVER if ps[0] == "elasticnet"
)


@register_estimator("logreg")
class LogReg(EstimatorAdapter):
    """Logistic regression (sklearn) adapter.

    Encodes sklearn's penalty/solver compatibility matrix as a single
    ``Categorical`` over valid ``(penalty, solver)`` pairs, so the optimizer
    never proposes invalid combinations. ``l1_ratio`` is gated to elasticnet.
    """

    supports_classification = True
    supports_regression = False

    def search_space(self, task: Task) -> SearchSpace:
        if task != "classification":
            raise ValueError(f"LogReg only supports classification, got '{task}'.")
        return SearchSpace(
            [
                Float("logreg_C", 1e-4, 1e4, log=True),
                Categorical("logreg_penalty_solver", _LOGREG_PENALTY_SOLVER),
                Conditional(
                    {"logreg_penalty_solver": _LOGREG_ELASTICNET_PAIRS},
                    Float("logreg_l1_ratio", 0.0, 1.0),
                ),
                Int("logreg_max_iter", 100, 2000, log=True),
            ]
        )

    def build(self, task: Task, params: dict[str, Any]) -> BaseEstimator:
        if task != "classification":
            raise ValueError(f"LogReg only supports classification, got '{task}'.")

        penalty, solver = params.get("logreg_penalty_solver", ("l2", "lbfgs"))
        kwargs: dict[str, Any] = {
            "C": params.get("logreg_C", 1.0),
            "penalty": None if penalty == "none" else penalty,
            "solver": solver,
            "max_iter": params.get("logreg_max_iter", 1000),
            "n_jobs": None if solver == "liblinear" else -1,
        }
        if penalty == "elasticnet":
            kwargs["l1_ratio"] = params.get("logreg_l1_ratio", 0.5)
        return LogisticRegression(**kwargs)


# ---------------------------------------------------------------------------
# Tree ensembles (RandomForest / ExtraTrees) — shared search space
# ---------------------------------------------------------------------------


def _forest_space(prefix: str) -> SearchSpace:
    return SearchSpace(
        [
            Int(f"{prefix}_n_estimators", 50, 500, log=True),
            Categorical(f"{prefix}_max_depth", [None, 4, 8, 16, 32]),
            Int(f"{prefix}_min_samples_split", 2, 20),
            Int(f"{prefix}_min_samples_leaf", 1, 20),
            Categorical(f"{prefix}_max_features", ["sqrt", "log2", 0.5, 1.0]),
            Categorical(f"{prefix}_bootstrap", [True, False]),
        ]
    )


def _forest_kwargs(prefix: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": params.get(f"{prefix}_n_estimators", 100),
        "max_depth": params.get(f"{prefix}_max_depth"),
        "min_samples_split": params.get(f"{prefix}_min_samples_split", 2),
        "min_samples_leaf": params.get(f"{prefix}_min_samples_leaf", 1),
        "max_features": params.get(f"{prefix}_max_features", "sqrt"),
        "bootstrap": params.get(f"{prefix}_bootstrap", True),
        "n_jobs": -1,
    }


@register_estimator("rf")
class RandomForest(EstimatorAdapter):
    """RandomForest adapter (classification + regression)."""

    supports_classification = True
    supports_regression = True

    def search_space(self, task: Task) -> SearchSpace:
        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task '{task}'.")
        return _forest_space("rf")

    def build(self, task: Task, params: dict[str, Any]) -> BaseEstimator:
        kwargs = _forest_kwargs("rf", params)
        if task == "classification":
            return RandomForestClassifier(**kwargs)
        if task == "regression":
            return RandomForestRegressor(**kwargs)
        raise ValueError(f"Unknown task '{task}'.")


@register_estimator("extratrees")
class ExtraTrees(EstimatorAdapter):
    """ExtraTrees adapter (classification + regression)."""

    supports_classification = True
    supports_regression = True

    def search_space(self, task: Task) -> SearchSpace:
        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task '{task}'.")
        return _forest_space("extratrees")

    def build(self, task: Task, params: dict[str, Any]) -> BaseEstimator:
        kwargs = _forest_kwargs("extratrees", params)
        if task == "classification":
            return ExtraTreesClassifier(**kwargs)
        if task == "regression":
            return ExtraTreesRegressor(**kwargs)
        raise ValueError(f"Unknown task '{task}'.")


# ---------------------------------------------------------------------------
# Histogram Gradient Boosting
# ---------------------------------------------------------------------------


@register_estimator("histgb")
class HistGB(EstimatorAdapter):
    """HistGradientBoosting adapter (classification + regression)."""

    supports_classification = True
    supports_regression = True

    def search_space(self, task: Task) -> SearchSpace:
        if task not in ("classification", "regression"):
            raise ValueError(f"Unknown task '{task}'.")
        return SearchSpace(
            [
                Float("histgb_learning_rate", 1e-3, 1.0, log=True),
                Int("histgb_max_iter", 50, 1000, log=True),
                Int("histgb_max_leaf_nodes", 15, 255, log=True),
                Categorical("histgb_max_depth", [None, 4, 8, 16, 32]),
                Int("histgb_min_samples_leaf", 5, 200, log=True),
                Float("histgb_l2_regularization", 1e-8, 1.0, log=True),
            ]
        )

    def build(self, task: Task, params: dict[str, Any]) -> BaseEstimator:
        kwargs: dict[str, Any] = {
            "learning_rate": params.get("histgb_learning_rate", 0.1),
            "max_iter": params.get("histgb_max_iter", 100),
            "max_leaf_nodes": params.get("histgb_max_leaf_nodes", 31),
            "max_depth": params.get("histgb_max_depth"),
            "min_samples_leaf": params.get("histgb_min_samples_leaf", 20),
            "l2_regularization": params.get("histgb_l2_regularization", 0.0),
        }
        if task == "classification":
            return HistGradientBoostingClassifier(**kwargs)
        if task == "regression":
            return HistGradientBoostingRegressor(**kwargs)
        raise ValueError(f"Unknown task '{task}'.")


# ---------------------------------------------------------------------------
# Linear regressors
# ---------------------------------------------------------------------------


@register_estimator("ridge")
class RidgeAdapter(EstimatorAdapter):
    """Ridge regression adapter."""

    supports_classification = False
    supports_regression = True

    def search_space(self, task: Task) -> SearchSpace:
        if task != "regression":
            raise ValueError(f"Ridge only supports regression, got '{task}'.")
        return SearchSpace(
            [
                Float("ridge_alpha", 1e-4, 1e4, log=True),
                Categorical(
                    "ridge_solver",
                    ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                ),
            ]
        )

    def build(self, task: Task, params: dict[str, Any]) -> BaseEstimator:
        if task != "regression":
            raise ValueError(f"Ridge only supports regression, got '{task}'.")
        return Ridge(
            alpha=params.get("ridge_alpha", 1.0),
            solver=params.get("ridge_solver", "auto"),
        )


@register_estimator("lasso")
class LassoAdapter(EstimatorAdapter):
    """Lasso regression adapter."""

    supports_classification = False
    supports_regression = True

    def search_space(self, task: Task) -> SearchSpace:
        if task != "regression":
            raise ValueError(f"Lasso only supports regression, got '{task}'.")
        return SearchSpace(
            [
                Float("lasso_alpha", 1e-4, 1e2, log=True),
                Int("lasso_max_iter", 500, 5000, log=True),
                Categorical("lasso_selection", ["cyclic", "random"]),
            ]
        )

    def build(self, task: Task, params: dict[str, Any]) -> BaseEstimator:
        if task != "regression":
            raise ValueError(f"Lasso only supports regression, got '{task}'.")
        return Lasso(
            alpha=params.get("lasso_alpha", 1.0),
            max_iter=params.get("lasso_max_iter", 1000),
            selection=params.get("lasso_selection", "cyclic"),
        )

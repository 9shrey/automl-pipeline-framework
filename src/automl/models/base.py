"""``EstimatorAdapter`` ABC.

Every model in the algorithm zoo (xgboost, lightgbm, sklearn, ...) implements
this single interface. The orchestrator never imports a concrete model — it
fetches an adapter from the registry by name and treats it uniformly.

Adding a new model is therefore a single new file:

    @register_estimator("my_model")
    class MyModel(EstimatorAdapter):
        supports_classification = True
        supports_regression = False

        def search_space(self, task): ...
        def build(self, task, params): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from sklearn.base import BaseEstimator

from automl.search.space import SearchSpace

Task = Literal["classification", "regression"]

__all__ = ["EstimatorAdapter", "Task"]


class EstimatorAdapter(ABC):
    """Adapt a third-party estimator to a uniform AutoML interface.

    Class attributes
    ----------------
    name :
        Registry key. Subclasses must override.
    supports_classification, supports_regression :
        Task capabilities. Subclasses must override appropriately.
    """

    name: str = ""
    supports_classification: bool = False
    supports_regression: bool = False

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abstractmethod
    def search_space(self, task: Task) -> SearchSpace:
        """Return the per-model hyperparameter search space for ``task``."""

    @abstractmethod
    def build(self, task: Task, params: dict[str, Any]) -> BaseEstimator:
        """Instantiate the underlying sklearn-compatible estimator."""

    # ------------------------------------------------------------------
    # Conveniences
    # ------------------------------------------------------------------

    def supports(self, task: Task) -> bool:
        if task == "classification":
            return self.supports_classification
        if task == "regression":
            return self.supports_regression
        raise ValueError(f"Unknown task '{task}'.")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"clf={self.supports_classification}, reg={self.supports_regression})"
        )

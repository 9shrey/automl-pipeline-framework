"""Algorithm zoo: EstimatorAdapter ABC, registry, concrete adapters."""

from automl.models.base import EstimatorAdapter
from automl.models.registry import EstimatorRegistry, register_estimator

__all__ = ["EstimatorAdapter", "EstimatorRegistry", "register_estimator"]

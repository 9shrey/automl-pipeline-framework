"""AutoML Pipeline Framework — public API surface.

Exports:
    AutoMLClassifier, AutoMLRegressor — sklearn-compatible meta-estimators.

See ``MASTER_PROMPT.md`` for the full spec.
"""

from __future__ import annotations

__version__ = "0.1.0"

# NOTE: api module is a stub at scaffold stage; concrete implementations land
# in subsequent commits per the Working Agreement.
try:
    from automl.api import AutoMLClassifier, AutoMLRegressor  # noqa: F401
except Exception:  # pragma: no cover - scaffold tolerance
    AutoMLClassifier = None  # type: ignore[assignment]
    AutoMLRegressor = None  # type: ignore[assignment]

__all__ = ["AutoMLClassifier", "AutoMLRegressor", "__version__"]

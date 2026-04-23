"""Optuna pruner factory: ASHA / Hyperband / Median / NopPruner."""

from __future__ import annotations

from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    SuccessiveHalvingPruner,
)

from automl.config.schema import PrunerName

__all__ = ["build_pruner"]


def build_pruner(name: PrunerName, *, n_splits: int) -> BasePruner:
    """Build a pruner. ``n_splits`` is the CV fold count used as the rung budget.

    ``min_resource=1`` means a trial can be pruned after the first fold; the
    full budget is ``n_splits``. Reduction factor of 3 follows Optuna's default.
    """
    if name == "asha":
        return SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
        )
    if name == "hyperband":
        return HyperbandPruner(
            min_resource=1,
            max_resource=max(1, n_splits),
            reduction_factor=3,
        )
    if name == "median":
        return MedianPruner(n_warmup_steps=1)
    if name == "none":
        return NopPruner()
    raise ValueError(f"Unknown pruner '{name}'.")

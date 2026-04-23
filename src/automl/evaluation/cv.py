"""Task-aware CV splitter factory.

Maps a ``CVConfig`` to a concrete sklearn splitter and validates that the
requested scheme matches the task and grouping inputs.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    StratifiedKFold,
)

from automl.config.schema import CVConfig, Task

__all__ = ["build_splitter", "split_indices"]


def build_splitter(config: CVConfig, task: Task, *, seed: int) -> BaseCrossValidator:
    """Return an sklearn splitter for the given config + task."""
    if config.scheme == "stratified_kfold":
        if task != "classification":
            raise ValueError(
                f"stratified_kfold requires task='classification', got '{task}'."
            )
        return StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=seed if config.shuffle else None,
        )
    if config.scheme == "kfold":
        return KFold(
            n_splits=config.n_splits,
            shuffle=config.shuffle,
            random_state=seed if config.shuffle else None,
        )
    if config.scheme == "group_kfold":
        return GroupKFold(n_splits=config.n_splits)
    raise ValueError(f"Unknown CV scheme: {config.scheme!r}.")


def split_indices(
    splitter: BaseCrossValidator,
    X: Any,
    y: Any,
    *,
    groups: Any | None = None,
) -> list[tuple[Any, Any]]:
    """Materialize ``(train_idx, val_idx)`` pairs as a list (deterministic order)."""
    return list(splitter.split(X, y, groups=groups))

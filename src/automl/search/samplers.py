"""Optuna sampler factory + ``OptunaSampler`` Trial adapter.

The ``Sampler`` protocol from ``automl.search.space`` matches Optuna's Trial
signature except for one thing: Optuna's ``suggest_categorical`` only accepts
primitive choice types (``None | bool | int | float | str``), while the DSL
allows arbitrary objects (e.g. ``LogReg`` uses ``(penalty, solver)`` tuples).
``OptunaSampler`` transparently encodes non-primitive choices as integer
indices and decodes them back, so the DSL works against Optuna unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import optuna
from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    RandomSampler as OptunaRandomSampler,
    TPESampler,
)

from automl.config.schema import SamplerName

__all__ = ["OptunaSampler", "build_sampler"]

_PRIMITIVE = (type(None), bool, int, float, str)


def _is_primitive(value: Any) -> bool:
    return isinstance(value, _PRIMITIVE)


class OptunaSampler:
    """Wrap an ``optuna.Trial`` so it satisfies ``automl.search.space.Sampler``.

    Non-primitive categorical choices are encoded as ``suggest_int(... low=0,
    high=len-1)`` and decoded back to the original object. This is invisible
    to the DSL.
    """

    def __init__(self, trial: optuna.Trial) -> None:
        self._trial = trial

    @property
    def trial(self) -> optuna.Trial:
        return self._trial

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        choices_list = list(choices)
        if not choices_list:
            raise ValueError(f"Categorical '{name}' has no choices.")
        if all(_is_primitive(c) for c in choices_list):
            return self._trial.suggest_categorical(name, choices_list)
        idx = self._trial.suggest_int(f"{name}__idx", 0, len(choices_list) - 1)
        return choices_list[idx]

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        return self._trial.suggest_float(name, low, high, log=log, step=step)

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int:
        return self._trial.suggest_int(name, low, high, log=log, step=step)


def build_sampler(name: SamplerName, *, seed: int) -> BaseSampler:
    if name == "tpe":
        return TPESampler(seed=seed)
    if name == "cmaes":
        return CmaEsSampler(seed=seed)
    if name == "random":
        return OptunaRandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler '{name}'.")

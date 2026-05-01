"""Joint search-space DSL for AutoML.

The DSL is intentionally tiny: a small set of leaf primitives (``Float``,
``Int``, ``Categorical``) plus a single composition primitive (``Conditional``)
gated by previously-sampled values. A ``SearchSpace`` is just an ordered list
of nodes; sampling walks them in order so conditionals can read parent values.

The DSL is decoupled from Optuna via the ``Sampler`` protocol — any object that
exposes ``suggest_categorical`` / ``suggest_float`` / ``suggest_int`` works,
which makes it trivial to unit-test against a deterministic ``RandomSampler``
without spinning up an Optuna study.

Example
-------
>>> from automl.search.space import (
...     Categorical, Float, Int, Conditional, SearchSpace, RandomSampler,
... )
>>> space = SearchSpace([
...     Categorical("model", ["logreg", "rf"]),
...     Conditional({"model": "logreg"}, Float("logreg_C", 1e-3, 1e3, log=True)),
...     Conditional({"model": "rf"}, Int("rf_n_estimators", 50, 1000, log=True)),
... ])
>>> sample = space.sample(RandomSampler(seed=0))
>>> set(sample) <= {"model", "logreg_C", "rf_n_estimators"}
True
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Categorical",
    "Conditional",
    "Float",
    "Int",
    "RandomSampler",
    "Sampler",
    "SearchSpace",
    "SpaceNode",
]


# ---------------------------------------------------------------------------
# Sampler protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Sampler(Protocol):
    """Minimal subset of the Optuna Trial API used by the DSL."""

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any: ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float: ...

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int: ...


class RandomSampler:
    """Deterministic ``Sampler`` backed by ``random.Random``.

    Used in unit tests and for cold-start sanity checks without Optuna.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        if not choices:
            raise ValueError(f"Categorical '{name}' has no choices.")
        return self._rng.choice(list(choices))

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        step: float | None = None,
    ) -> float:
        if low > high:
            raise ValueError(f"Float '{name}': low ({low}) > high ({high}).")
        if log:
            if low <= 0:
                raise ValueError(f"Float '{name}': log scale requires low > 0 (got {low}).")
            return math.exp(self._rng.uniform(math.log(low), math.log(high)))
        if step is not None:
            if step <= 0:
                raise ValueError(f"Float '{name}': step must be > 0 (got {step}).")
            n_steps = int((high - low) / step)
            return low + self._rng.randint(0, n_steps) * step
        return self._rng.uniform(low, high)

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
        step: int = 1,
    ) -> int:
        if low > high:
            raise ValueError(f"Int '{name}': low ({low}) > high ({high}).")
        if step <= 0:
            raise ValueError(f"Int '{name}': step must be > 0 (got {step}).")
        if log:
            if low <= 0:
                raise ValueError(f"Int '{name}': log scale requires low > 0 (got {low}).")
            value = math.exp(self._rng.uniform(math.log(low), math.log(high)))
            return round(value)
        n_steps = (high - low) // step
        return low + self._rng.randint(0, n_steps) * step


# ---------------------------------------------------------------------------
# Space nodes
# ---------------------------------------------------------------------------


class SpaceNode(ABC):
    """A node in the search space. Either a leaf primitive or a ``Conditional``."""

    name: str

    @abstractmethod
    def sample(self, sampler: Sampler, current: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a dict of ``{name: value}`` contributions to the running sample.

        Returns an empty dict when a conditional gate is not satisfied.
        """

    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """All parameter names this node may emit (including conditional children)."""


@dataclass(frozen=True)
class Float(SpaceNode):
    name: str
    low: float
    high: float
    log: bool = False
    step: float | None = None

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Float '{self.name}': low ({self.low}) > high ({self.high}).")
        if self.log and self.low <= 0:
            raise ValueError(
                f"Float '{self.name}': log scale requires low > 0 (got {self.low})."
            )
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Float '{self.name}': step must be > 0 (got {self.step}).")
        if self.log and self.step is not None:
            raise ValueError(f"Float '{self.name}': log and step are mutually exclusive.")

    def sample(self, sampler: Sampler, current: Mapping[str, Any]) -> Mapping[str, Any]:
        value = sampler.suggest_float(self.name, self.low, self.high, log=self.log, step=self.step)
        return {self.name: value}

    def parameter_names(self) -> tuple[str, ...]:
        return (self.name,)


@dataclass(frozen=True)
class Int(SpaceNode):
    name: str
    low: int
    high: int
    log: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Int '{self.name}': low ({self.low}) > high ({self.high}).")
        if self.step <= 0:
            raise ValueError(f"Int '{self.name}': step must be > 0 (got {self.step}).")
        if self.log and self.low <= 0:
            raise ValueError(f"Int '{self.name}': log scale requires low > 0 (got {self.low}).")
        if self.log and self.step != 1:
            raise ValueError(f"Int '{self.name}': log and step != 1 are mutually exclusive.")

    def sample(self, sampler: Sampler, current: Mapping[str, Any]) -> Mapping[str, Any]:
        value = sampler.suggest_int(self.name, self.low, self.high, log=self.log, step=self.step)
        return {self.name: value}

    def parameter_names(self) -> tuple[str, ...]:
        return (self.name,)


@dataclass(frozen=True)
class Categorical(SpaceNode):
    name: str
    choices: tuple[Any, ...]

    def __init__(self, name: str, choices: Iterable[Any]) -> None:
        choices_t = tuple(choices)
        if not choices_t:
            raise ValueError(f"Categorical '{name}': choices must be non-empty.")
        # Frozen dataclass workaround
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "choices", choices_t)

    def sample(self, sampler: Sampler, current: Mapping[str, Any]) -> Mapping[str, Any]:
        return {self.name: sampler.suggest_categorical(self.name, self.choices)}

    def parameter_names(self) -> tuple[str, ...]:
        return (self.name,)


@dataclass(frozen=True)
class Conditional(SpaceNode):
    """Activate ``node`` only when each parent in ``when`` matches a sampled value.

    ``when`` maps parent parameter name → value or iterable of values.
    """

    when: Mapping[str, Any]
    node: SpaceNode
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.when:
            raise ValueError("Conditional 'when' must be a non-empty mapping.")
        if not isinstance(self.node, SpaceNode):
            raise TypeError("Conditional 'node' must be a SpaceNode.")
        if isinstance(self.node, Conditional):
            raise TypeError("Conditional cannot wrap another Conditional directly; nest via SearchSpace.")
        object.__setattr__(self, "name", self.node.name)

    def _is_active(self, current: Mapping[str, Any]) -> bool:
        for parent, expected in self.when.items():
            if parent not in current:
                return False
            value = current[parent]
            if isinstance(expected, (list, tuple, set, frozenset)):
                if value not in expected:
                    return False
            elif value != expected:
                return False
        return True

    def sample(self, sampler: Sampler, current: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._is_active(current):
            return {}
        return self.node.sample(sampler, current)

    def parameter_names(self) -> tuple[str, ...]:
        return self.node.parameter_names()


# ---------------------------------------------------------------------------
# Search space container
# ---------------------------------------------------------------------------


class SearchSpace:
    """An ordered collection of ``SpaceNode``s.

    Order matters: a ``Conditional`` may reference any parent emitted by an
    earlier node. Construction validates parent references and forbids
    duplicate parameter names.
    """

    def __init__(self, nodes: Iterable[SpaceNode]) -> None:
        nodes_t = tuple(nodes)
        seen_names: set[str] = set()
        emitted: set[str] = set()
        for node in nodes_t:
            if not isinstance(node, SpaceNode):
                raise TypeError(f"Expected SpaceNode, got {type(node).__name__}.")
            for pname in node.parameter_names():
                if pname in seen_names:
                    raise ValueError(f"Duplicate parameter name in search space: '{pname}'.")
                seen_names.add(pname)
            if isinstance(node, Conditional):
                missing = [p for p in node.when if p not in emitted]
                if missing:
                    raise ValueError(
                        f"Conditional on '{node.name}' references undeclared parents: {missing}. "
                        "Place parent nodes before the Conditional in the SearchSpace."
                    )
            emitted.update(node.parameter_names())
        self._nodes: tuple[SpaceNode, ...] = nodes_t

    @property
    def nodes(self) -> tuple[SpaceNode, ...]:
        return self._nodes

    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for node in self._nodes:
            names.extend(node.parameter_names())
        return tuple(names)

    def sample(self, sampler: Sampler) -> dict[str, Any]:
        """Sample one configuration from the space using ``sampler``."""
        out: dict[str, Any] = {}
        for node in self._nodes:
            contribution = node.sample(sampler, out)
            for k, v in contribution.items():
                out[k] = v
        return out

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._nodes)

"""Decorator-based estimator registry.

Adding a new model is one file:

    from automl.models import EstimatorAdapter, register_estimator

    @register_estimator("my_model")
    class MyModel(EstimatorAdapter):
        ...
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import ClassVar, TypeVar

from automl.models.base import EstimatorAdapter, Task

__all__ = ["EstimatorRegistry", "register_estimator"]

A = TypeVar("A", bound=EstimatorAdapter)


class EstimatorRegistry:
    """Singleton-style registry mapping name → adapter instance."""

    _registry: ClassVar[dict[str, EstimatorAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter: EstimatorAdapter) -> None:
        if not name:
            raise ValueError("Estimator name must be non-empty.")
        if name in cls._registry:
            raise ValueError(f"Estimator '{name}' is already registered.")
        adapter.name = name
        cls._registry[name] = adapter

    @classmethod
    def get(cls, name: str) -> EstimatorAdapter:
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise KeyError(
                f"Estimator '{name}' is not registered. "
                f"Available: {sorted(cls._registry)}"
            ) from exc

    @classmethod
    def names(cls, task: Task | None = None) -> list[str]:
        if task is None:
            return sorted(cls._registry)
        return sorted(n for n, a in cls._registry.items() if a.supports(task))

    @classmethod
    def items(cls) -> Iterator[tuple[str, EstimatorAdapter]]:
        return iter(cls._registry.items())

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._registry.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


def register_estimator(name: str) -> Callable[[type[A]], type[A]]:
    """Class decorator that registers an ``EstimatorAdapter`` subclass."""

    def decorator(adapter_cls: type[A]) -> type[A]:
        EstimatorRegistry.register(name, adapter_cls())
        return adapter_cls

    return decorator

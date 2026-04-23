"""Search subpackage: joint search-space DSL, samplers, pruners, Optuna engine.

Heavy submodules (``engine``, ``objective``, ``samplers``, ``pruners``) are not
re-exported here to avoid a circular import via ``automl.models.base`` (which
imports ``SearchSpace`` from this package). Import them explicitly:

    from automl.search.engine import run_search
    from automl.search.objective import Objective
"""

from automl.search.space import (
    Categorical,
    Conditional,
    Float,
    Int,
    RandomSampler,
    Sampler,
    SearchSpace,
    SpaceNode,
)

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

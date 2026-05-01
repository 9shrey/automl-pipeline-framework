"""Unit tests for ``automl.search.samplers`` + ``pruners``."""

from __future__ import annotations

import optuna
import pytest
from optuna.pruners import (
    HyperbandPruner,
    MedianPruner,
    NopPruner,
    SuccessiveHalvingPruner,
)
from optuna.samplers import CmaEsSampler, TPESampler
from optuna.samplers import RandomSampler as OptunaRandomSampler

from automl.search.pruners import build_pruner
from automl.search.samplers import OptunaSampler, build_sampler
from automl.search.space import Categorical, Float, Int, SearchSpace


def _make_trial() -> optuna.Trial:
    study = optuna.create_study(direction="maximize")
    return study.ask()


# ---------------------------------------------------------------------------
# OptunaSampler adapter
# ---------------------------------------------------------------------------


class TestOptunaSampler:
    def test_primitive_categorical_passthrough(self) -> None:
        s = OptunaSampler(_make_trial())
        v = s.suggest_categorical("model", ["a", "b", "c"])
        assert v in {"a", "b", "c"}

    def test_non_primitive_categorical_via_index(self) -> None:
        s = OptunaSampler(_make_trial())
        v = s.suggest_categorical("pair", [("a", 1), ("b", 2)])
        assert v in {("a", 1), ("b", 2)}

    def test_float_and_int(self) -> None:
        s = OptunaSampler(_make_trial())
        f = s.suggest_float("x", 0.0, 1.0)
        i = s.suggest_int("n", 1, 10, log=True)
        assert 0.0 <= f <= 1.0
        assert 1 <= i <= 10

    def test_dsl_can_sample_through_optuna(self) -> None:
        space = SearchSpace(
            [
                Categorical("pair", [("a", 1), ("b", 2)]),
                Float("x", 1e-3, 1e3, log=True),
                Int("n", 1, 100),
            ]
        )
        sample = space.sample(OptunaSampler(_make_trial()))
        assert sample["pair"] in {("a", 1), ("b", 2)}
        assert 1e-3 <= sample["x"] <= 1e3
        assert 1 <= sample["n"] <= 100

    def test_empty_choices_rejected(self) -> None:
        s = OptunaSampler(_make_trial())
        with pytest.raises(ValueError, match="no choices"):
            s.suggest_categorical("x", [])


# ---------------------------------------------------------------------------
# Sampler / Pruner factories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "klass"),
    [("tpe", TPESampler), ("cmaes", CmaEsSampler), ("random", OptunaRandomSampler)],
)
def test_build_sampler(name: str, klass: type) -> None:
    assert isinstance(build_sampler(name, seed=0), klass)


def test_build_sampler_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown sampler"):
        build_sampler("nonsense", seed=0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "klass"),
    [
        ("asha", SuccessiveHalvingPruner),
        ("hyperband", HyperbandPruner),
        ("median", MedianPruner),
        ("none", NopPruner),
    ],
)
def test_build_pruner(name: str, klass: type) -> None:
    assert isinstance(build_pruner(name, n_splits=3), klass)


def test_build_pruner_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown pruner"):
        build_pruner("nonsense", n_splits=3)  # type: ignore[arg-type]

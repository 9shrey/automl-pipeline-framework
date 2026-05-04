"""Unit tests for ``automl.search.space``."""

from __future__ import annotations

import math

import pytest

from automl.search.space import (
    Categorical,
    Conditional,
    Float,
    Int,
    RandomSampler,
    SearchSpace,
)

# ---------------------------------------------------------------------------
# Leaf primitives
# ---------------------------------------------------------------------------


class TestFloat:
    def test_uniform_range(self) -> None:
        node = Float("x", 0.0, 1.0)
        sampler = RandomSampler(seed=0)
        for _ in range(50):
            value = node.sample(sampler, {})["x"]
            assert 0.0 <= value <= 1.0

    def test_log_scale_positive_only(self) -> None:
        with pytest.raises(ValueError, match="log scale requires low > 0"):
            Float("x", 0.0, 1.0, log=True)

    def test_low_must_not_exceed_high(self) -> None:
        with pytest.raises(ValueError, match=r"low .* > high"):
            Float("x", 1.0, 0.0)

    def test_log_and_step_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="log and step"):
            Float("x", 1.0, 10.0, log=True, step=0.5)

    def test_log_distribution_stays_in_range(self) -> None:
        node = Float("x", 1e-3, 1e3, log=True)
        sampler = RandomSampler(seed=42)
        for _ in range(100):
            value = node.sample(sampler, {})["x"]
            assert 1e-3 <= value <= 1e3

    def test_step_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="step must be > 0"):
            Float("x", 0.0, 1.0, step=-0.1)


class TestInt:
    def test_range(self) -> None:
        node = Int("n", 1, 10)
        sampler = RandomSampler(seed=1)
        for _ in range(100):
            value = node.sample(sampler, {})["n"]
            assert 1 <= value <= 10
            assert isinstance(value, int)

    def test_log(self) -> None:
        node = Int("n", 1, 1000, log=True)
        sampler = RandomSampler(seed=2)
        for _ in range(100):
            value = node.sample(sampler, {})["n"]
            assert 1 <= value <= 1000

    def test_step_validation(self) -> None:
        with pytest.raises(ValueError, match="step must be > 0"):
            Int("n", 1, 10, step=0)

    def test_log_with_nontrivial_step_rejected(self) -> None:
        with pytest.raises(ValueError, match="log and step"):
            Int("n", 1, 10, log=True, step=2)


class TestCategorical:
    def test_chooses_only_from_choices(self) -> None:
        node = Categorical("m", ["a", "b", "c"])
        sampler = RandomSampler(seed=3)
        seen = {node.sample(sampler, {})["m"] for _ in range(100)}
        assert seen <= {"a", "b", "c"}

    def test_empty_choices_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Categorical("m", [])

    def test_accepts_iterable(self) -> None:
        node = Categorical("m", iter(["a", "b"]))
        assert node.choices == ("a", "b")


# ---------------------------------------------------------------------------
# Conditional gating
# ---------------------------------------------------------------------------


class TestConditional:
    def test_active_when_parent_matches(self) -> None:
        cond = Conditional({"model": "rf"}, Int("rf_n", 1, 100))
        sampler = RandomSampler(seed=0)
        out = cond.sample(sampler, {"model": "rf"})
        assert "rf_n" in out

    def test_inactive_when_parent_mismatches(self) -> None:
        cond = Conditional({"model": "rf"}, Int("rf_n", 1, 100))
        sampler = RandomSampler(seed=0)
        out = cond.sample(sampler, {"model": "logreg"})
        assert out == {}

    def test_inactive_when_parent_missing(self) -> None:
        cond = Conditional({"model": "rf"}, Int("rf_n", 1, 100))
        out = cond.sample(RandomSampler(seed=0), {})
        assert out == {}

    def test_set_membership_gate(self) -> None:
        cond = Conditional({"model": ["rf", "extratrees"]}, Int("trees", 10, 100))
        sampler = RandomSampler(seed=0)
        assert "trees" in cond.sample(sampler, {"model": "rf"})
        assert "trees" in cond.sample(sampler, {"model": "extratrees"})
        assert cond.sample(sampler, {"model": "logreg"}) == {}

    def test_multiple_parents_all_must_match(self) -> None:
        cond = Conditional({"a": 1, "b": "x"}, Float("v", 0.0, 1.0))
        sampler = RandomSampler(seed=0)
        assert cond.sample(sampler, {"a": 1, "b": "x"})
        assert cond.sample(sampler, {"a": 1, "b": "y"}) == {}
        assert cond.sample(sampler, {"a": 2, "b": "x"}) == {}

    def test_empty_when_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Conditional({}, Float("x", 0.0, 1.0))

    def test_nested_conditional_rejected(self) -> None:
        inner = Conditional({"a": 1}, Float("x", 0.0, 1.0))
        with pytest.raises(TypeError, match="another Conditional"):
            Conditional({"b": 2}, inner)


# ---------------------------------------------------------------------------
# SearchSpace composition
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_sampling_respects_conditionals(self) -> None:
        space = SearchSpace(
            [
                Categorical("model", ["logreg", "rf"]),
                Conditional({"model": "logreg"}, Float("logreg_C", 1e-3, 1e3, log=True)),
                Conditional({"model": "rf"}, Int("rf_n", 50, 500, log=True)),
            ]
        )
        sampler = RandomSampler(seed=0)
        for _ in range(50):
            sample = space.sample(sampler)
            assert sample["model"] in {"logreg", "rf"}
            if sample["model"] == "logreg":
                assert "logreg_C" in sample
                assert "rf_n" not in sample
            else:
                assert "rf_n" in sample
                assert "logreg_C" not in sample

    def test_determinism_with_fixed_seed(self) -> None:
        space = SearchSpace(
            [
                Categorical("model", ["a", "b"]),
                Float("x", 0.0, 1.0),
            ]
        )
        s1 = space.sample(RandomSampler(seed=123))
        s2 = space.sample(RandomSampler(seed=123))
        assert s1 == s2

    def test_different_seeds_produce_different_samples(self) -> None:
        space = SearchSpace([Float("x", 0.0, 1.0)])
        s1 = space.sample(RandomSampler(seed=1))
        s2 = space.sample(RandomSampler(seed=2))
        assert not math.isclose(s1["x"], s2["x"])

    def test_duplicate_parameter_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="Duplicate parameter name"):
            SearchSpace([Float("x", 0.0, 1.0), Int("x", 0, 1)])

    def test_conditional_referencing_undeclared_parent_rejected(self) -> None:
        with pytest.raises(ValueError, match="undeclared parents"):
            SearchSpace(
                [
                    Conditional({"model": "rf"}, Int("rf_n", 1, 10)),
                    Categorical("model", ["rf"]),
                ]
            )

    def test_non_spacenode_rejected(self) -> None:
        with pytest.raises(TypeError, match="Expected SpaceNode"):
            SearchSpace(["not a node"])  # type: ignore[list-item]

    def test_parameter_names_lists_all(self) -> None:
        space = SearchSpace(
            [
                Categorical("model", ["a", "b"]),
                Conditional({"model": "a"}, Float("x", 0.0, 1.0)),
            ]
        )
        assert space.parameter_names() == ("model", "x")

    def test_len_and_iter(self) -> None:
        nodes = [Float("x", 0.0, 1.0), Int("n", 0, 10)]
        space = SearchSpace(nodes)
        assert len(space) == 2
        assert list(space) == nodes

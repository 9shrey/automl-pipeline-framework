"""Property tests for the search-space DSL.

These verify invariants that should hold for any valid space:

* every parameter name in the result is declared by the space,
* sampling is deterministic for a fixed seed,
* sampled values lie inside their declared bounds,
* duplicate-name guard fires reliably.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from automl.search.space import (
    Categorical,
    Conditional,
    Float,
    Int,
    RandomSampler,
    SearchSpace,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

names = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1, max_size=8,
)


@st.composite
def float_node(draw, name):  # type: ignore[no-untyped-def]
    low = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    high = low + draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    log = draw(st.booleans()) and low > 0
    return Float(name=name, low=low, high=high, log=log)


@st.composite
def int_node(draw, name):  # type: ignore[no-untyped-def]
    low = draw(st.integers(min_value=1, max_value=50))
    high = low + draw(st.integers(min_value=0, max_value=50))
    return Int(name=name, low=low, high=high)


@st.composite
def cat_node(draw, name):  # type: ignore[no-untyped-def]
    choices = draw(st.lists(st.sampled_from(["a", "b", "c", "d", "e"]), min_size=1, max_size=5, unique=True))
    return Categorical(name=name, choices=choices)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(unique_names=st.lists(names, min_size=1, max_size=5, unique=True), seed=st.integers(0, 2**31 - 1))
def test_sampling_is_deterministic_for_fixed_seed(unique_names, seed):
    nodes = [Categorical(n, ["x", "y", "z"]) for n in unique_names]
    space = SearchSpace(nodes)
    a = space.sample(RandomSampler(seed))
    b = space.sample(RandomSampler(seed))
    assert a == b


@settings(max_examples=50, deadline=None)
@given(
    name=names,
    low=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    seed=st.integers(0, 2**31 - 1),
)
def test_float_sample_within_bounds(name, low, width, seed):
    high = low + width
    space = SearchSpace([Float(name, low, high)])
    out = space.sample(RandomSampler(seed))
    assert low <= out[name] <= high


@settings(max_examples=50, deadline=None)
@given(
    name=names,
    low=st.integers(min_value=-50, max_value=50),
    width=st.integers(min_value=0, max_value=50),
    seed=st.integers(0, 2**31 - 1),
)
def test_int_sample_within_bounds(name, low, width, seed):
    high = low + width
    space = SearchSpace([Int(name, low, high)])
    out = space.sample(RandomSampler(seed))
    assert low <= out[name] <= high
    assert isinstance(out[name], int)


@settings(max_examples=50, deadline=None)
@given(
    name=names,
    choices=st.lists(st.sampled_from([1, 2, "a", "b", None, True]), min_size=1, max_size=6, unique=True),
    seed=st.integers(0, 2**31 - 1),
)
def test_categorical_sample_in_choices(name, choices, seed):
    space = SearchSpace([Categorical(name, choices)])
    out = space.sample(RandomSampler(seed))
    assert out[name] in choices


@settings(max_examples=30, deadline=None)
@given(name=names)
def test_duplicate_parameter_name_rejected(name):
    with pytest.raises(ValueError, match="Duplicate parameter name"):
        SearchSpace([Categorical(name, ["a", "b"]), Categorical(name, ["c"])])


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_conditional_only_emits_when_active(seed):
    space = SearchSpace([
        Categorical("model", ["logreg", "rf"]),
        Conditional({"model": "logreg"}, Float("logreg_C", 1e-3, 1e3, log=True)),
        Conditional({"model": "rf"}, Int("rf_n_estimators", 50, 500)),
    ])
    out = space.sample(RandomSampler(seed))
    assert "model" in out
    if out["model"] == "logreg":
        assert "logreg_C" in out and "rf_n_estimators" not in out
        assert 1e-3 <= out["logreg_C"] <= 1e3
    else:
        assert "rf_n_estimators" in out and "logreg_C" not in out
        assert 50 <= out["rf_n_estimators"] <= 500


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_log_float_is_positive(seed):
    space = SearchSpace([Float("x", 1e-5, 1e5, log=True)])
    out = space.sample(RandomSampler(seed))
    assert out["x"] > 0
    assert math.isfinite(out["x"])

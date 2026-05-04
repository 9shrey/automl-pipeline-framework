"""Unit tests for ``automl.preprocessing``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from automl.data import infer_schema
from automl.preprocessing import (
    IMPUTERS,
    SCALERS,
    FrequencyEncoder,
    build_encoder,
    build_imputer,
    build_preprocessor,
    build_scaler,
)


@pytest.mark.parametrize("name", IMPUTERS)
def test_build_imputer_returns_fittable(name: str) -> None:
    imp = build_imputer(name, kind="numeric")
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
    out = imp.fit_transform(X)
    assert out.shape == X.shape
    assert not np.isnan(out).any()


def test_unknown_imputer_raises() -> None:
    with pytest.raises(ValueError, match="Unknown imputer"):
        build_imputer("nonsense", kind="numeric")


@pytest.mark.parametrize("name", [n for n in SCALERS if n != "none"])
def test_build_scaler_transforms(name: str) -> None:
    X = np.random.RandomState(0).randn(50, 3)
    sc = build_scaler(name)
    out = sc.fit_transform(X)
    assert out.shape == X.shape


def test_unknown_scaler_raises() -> None:
    with pytest.raises(ValueError, match="Unknown scaler"):
        build_scaler("nonsense")


def test_scaler_none_is_passthrough() -> None:
    X = np.random.RandomState(0).randn(10, 2)
    sc = build_scaler("none")
    np.testing.assert_array_equal(sc.fit_transform(X), X)


@pytest.mark.parametrize("name", ["onehot", "ordinal", "frequency"])
def test_encoders_round_trip(name: str) -> None:
    enc = build_encoder(name)
    X = np.array([["a"], ["b"], ["a"], ["c"]], dtype=object)
    out = enc.fit_transform(X)
    assert out.shape[0] == 4


def test_target_encoder_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="TargetEncoder"):
        build_encoder("target")


def test_unknown_encoder_raises() -> None:
    with pytest.raises(ValueError, match="Unknown encoder"):
        build_encoder("nonsense")


def test_frequency_encoder_handles_unknown_at_transform() -> None:
    enc = FrequencyEncoder().fit(np.array([["a"], ["a"], ["b"]], dtype=object))
    out = enc.transform(np.array([["a"], ["c"]], dtype=object))
    assert out[0, 0] == pytest.approx(2 / 3)
    assert out[1, 0] == 0.0


def test_frequency_encoder_rejects_dim_mismatch() -> None:
    enc = FrequencyEncoder().fit(np.array([["a"], ["b"]], dtype=object))
    with pytest.raises(ValueError, match="cols"):
        enc.transform(np.array([["a", "x"], ["b", "y"]], dtype=object))


# ---------------------------------------------------------------------------
# Preprocessor factory
# ---------------------------------------------------------------------------


def _mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, 40, 50],
            "income": [1.0, 2.0, np.nan, 4.0, 5.0],
            "is_member": [True, False, True, False, True],
            "city": ["NY", "LA", "NY", "SF", "LA"],
        }
    )


class TestBuildPreprocessor:
    def test_returns_column_transformer(self) -> None:
        df = _mixed_df()
        schema = infer_schema(df)
        pre = build_preprocessor(
            schema, imputation="mean", encoding="onehot", scaling="standard"
        )
        assert isinstance(pre, ColumnTransformer)

    def test_fit_transform_handles_missing_and_categorical(self) -> None:
        df = _mixed_df()
        schema = infer_schema(df)
        pre = build_preprocessor(
            schema, imputation="median", encoding="onehot", scaling="robust"
        )
        out = pre.fit_transform(df)
        # Numeric (2) + bool (1) + onehot of {NY,LA,SF} (3) = 6 cols.
        assert out.shape == (5, 6)
        assert not np.isnan(np.asarray(out, dtype=float)).any()

    def test_handles_only_numeric(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        schema = infer_schema(df)
        pre = build_preprocessor(
            schema, imputation="mean", encoding="onehot", scaling="standard"
        )
        out = pre.fit_transform(df)
        assert out.shape == (3, 2)

    def test_handles_only_categorical(self) -> None:
        df = pd.DataFrame({"c": ["a", "b", "a"], "d": ["x", "x", "y"]})
        schema = infer_schema(df)
        pre = build_preprocessor(
            schema, imputation="mean", encoding="ordinal", scaling="none"
        )
        out = pre.fit_transform(df)
        assert out.shape == (3, 2)

    def test_empty_schema_rejected(self) -> None:
        from automl.data.schema_infer import ColumnSchema

        empty = ColumnSchema((), (), (), ())
        with pytest.raises(ValueError, match="empty"):
            build_preprocessor(
                empty, imputation="mean", encoding="onehot", scaling="standard"
            )

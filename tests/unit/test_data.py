"""Unit tests for ``automl.data`` (schema_infer + hashing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.data import hash_array, hash_dataframe, hash_xy, infer_schema

# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------


class TestInferSchema:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": [10, 20, 30, None],
                "income": [1.5, 2.5, 3.5, 4.5],
                "is_member": [True, False, True, False],
                "city": ["NY", "LA", "NY", "SF"],
                "user_id": [f"u{i}" for i in range(4)],  # high cardinality
            }
        )

    def test_basic_split(self) -> None:
        s = infer_schema(self._df(), high_card_threshold=3)
        assert "age" in s.numeric and "income" in s.numeric
        assert "is_member" in s.boolean
        assert "city" in s.low_card_categorical
        assert "user_id" in s.high_card_categorical

    def test_categorical_property(self) -> None:
        s = infer_schema(self._df(), high_card_threshold=3)
        assert set(s.categorical) == {"city", "user_id"}

    def test_all_columns_round_trip(self) -> None:
        df = self._df()
        s = infer_schema(df, high_card_threshold=10)
        assert set(s.all_columns) == set(df.columns)

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            infer_schema(np.array([[1, 2], [3, 4]]))  # type: ignore[arg-type]

    def test_threshold_validation(self) -> None:
        with pytest.raises(ValueError, match="high_card_threshold"):
            infer_schema(self._df(), high_card_threshold=0)

    def test_datetime_treated_as_high_card(self) -> None:
        df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, freq="D")})
        s = infer_schema(df)
        assert "ts" in s.high_card_categorical


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


class TestHashing:
    def test_dataframe_hash_deterministic(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert hash_dataframe(df) == hash_dataframe(df.copy())

    def test_dataframe_hash_changes_with_content(self) -> None:
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 4]})
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_dataframe_hash_changes_with_column_order(self) -> None:
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = df1[["b", "a"]]
        assert hash_dataframe(df1) != hash_dataframe(df2)

    def test_array_hash_deterministic(self) -> None:
        arr = np.arange(10).reshape(5, 2).astype(float)
        assert hash_array(arr) == hash_array(arr.copy())

    def test_array_hash_dtype_matters(self) -> None:
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1, 2, 3], dtype=np.int64)
        assert hash_array(a) != hash_array(b)

    def test_xy_hash_combines(self) -> None:
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        h = hash_xy(X, y)
        # Different y → different hash.
        h2 = hash_xy(X, pd.Series([1, 1, 0]))
        assert h != h2
        assert isinstance(h, str) and len(h) == 32

    def test_dataframe_hash_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            hash_dataframe([1, 2, 3])  # type: ignore[arg-type]

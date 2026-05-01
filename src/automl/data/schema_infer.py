"""Infer column roles (numeric / low-cardinality categorical / high-cardinality
categorical) from a ``pandas.DataFrame`` without mutating it.

The split drives the preprocessing factory: numeric columns route to imputers
+ scalers; categorical columns route to encoders. Cardinality threshold lets
the encoder choice (one-hot vs ordinal/target/frequency) react sensibly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt
from pandas.api.types import CategoricalDtype

__all__ = ["ColumnSchema", "infer_schema"]


@dataclass(frozen=True)
class ColumnSchema:
    numeric: tuple[str, ...]
    low_card_categorical: tuple[str, ...]
    high_card_categorical: tuple[str, ...]
    boolean: tuple[str, ...]

    @property
    def categorical(self) -> tuple[str, ...]:
        return self.low_card_categorical + self.high_card_categorical

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.numeric + self.boolean + self.categorical


def infer_schema(
    X: pd.DataFrame,
    *,
    high_card_threshold: int = 32,
) -> ColumnSchema:
    """Categorize each column of ``X`` by dtype + cardinality.

    Numeric columns are anything ``np.number``; booleans are split out so they
    can bypass scaling. Object / string / pandas ``category`` dtypes are
    classified by ``nunique`` against ``high_card_threshold``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"infer_schema expects a pandas DataFrame, got {type(X).__name__}.")
    if high_card_threshold < 1:
        raise ValueError("high_card_threshold must be >= 1.")

    numeric: list[str] = []
    boolean: list[str] = []
    low_card: list[str] = []
    high_card: list[str] = []

    for col in X.columns:
        s = X[col]
        if pdt.is_bool_dtype(s):
            boolean.append(col)
        elif pdt.is_numeric_dtype(s):
            numeric.append(col)
        elif isinstance(s.dtype, CategoricalDtype) or pdt.is_object_dtype(s) or pdt.is_string_dtype(s):
            n_unique = int(s.nunique(dropna=True))
            if n_unique <= high_card_threshold:
                low_card.append(col)
            else:
                high_card.append(col)
        elif pdt.is_datetime64_any_dtype(s):
            # Treat datetimes as high-cardinality categorical for now; a
            # dedicated datetime expander lands in a future commit.
            high_card.append(col)
        else:
            # Fall back: treat as high-cardinality categorical so it doesn't
            # silently get scaled as numeric.
            high_card.append(col)

    return ColumnSchema(
        numeric=tuple(numeric),
        low_card_categorical=tuple(low_card),
        high_card_categorical=tuple(high_card),
        boolean=tuple(boolean),
    )

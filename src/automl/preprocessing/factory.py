"""Compose preprocessing choices into a sklearn ``ColumnTransformer``.

Given a ``ColumnSchema`` and a sampled ``{imputation, encoding, scaling}``
config, returns a ColumnTransformer that:
  - imputes + scales numeric columns,
  - imputes (most_frequent) + encodes categorical columns,
  - passes booleans through.

The ColumnTransformer is *unfitted* and ready to plug into a Pipeline. Fitting
inside CV folds is enforced by the search engine, not by this factory.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from automl.data.schema_infer import ColumnSchema
from automl.preprocessing.encoders import build_encoder
from automl.preprocessing.imputers import build_imputer
from automl.preprocessing.scalers import build_scaler

__all__ = ["build_preprocessor"]


def build_preprocessor(
    schema: ColumnSchema,
    *,
    imputation: str,
    encoding: str,
    scaling: str,
) -> BaseEstimator:
    """Build a ``ColumnTransformer`` from a sampled preprocessing config."""
    transformers: list[tuple[str, BaseEstimator, list[str]]] = []

    if schema.numeric:
        num_pipe = Pipeline(
            steps=[
                ("impute", build_imputer(imputation, kind="numeric")),
                ("scale", build_scaler(scaling)),
            ]
        )
        transformers.append(("num", num_pipe, list(schema.numeric)))

    if schema.categorical:
        cat_pipe = Pipeline(
            steps=[
                ("impute", build_imputer("most_frequent", kind="categorical")),
                ("encode", build_encoder(encoding)),
            ]
        )
        transformers.append(("cat", cat_pipe, list(schema.categorical)))

    if schema.boolean:
        transformers.append(("bool", "passthrough", list(schema.boolean)))

    if not transformers:
        raise ValueError("Schema has no usable columns; refusing to build empty preprocessor.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

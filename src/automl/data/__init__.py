"""Data subpackage: loaders, schema inference, deterministic hashing."""

from automl.data.hashing import hash_array, hash_dataframe, hash_xy
from automl.data.schema_infer import ColumnSchema, infer_schema

__all__ = [
    "ColumnSchema",
    "hash_array",
    "hash_dataframe",
    "hash_xy",
    "infer_schema",
]

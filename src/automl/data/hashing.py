"""Deterministic dataset hashing for reproducibility.

Used by the run recorder to stamp every run with the exact dataset fingerprint.
Uses ``pandas.util.hash_pandas_object`` (consistent across pandas versions for
a fixed schema) plus a final BLAKE2b digest.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["hash_array", "hash_dataframe", "hash_xy"]

_DIGEST_SIZE = 16  # bytes → 32-char hex


def hash_dataframe(df: pd.DataFrame) -> str:
    """Return a stable hex digest of a DataFrame's contents and column order."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"hash_dataframe expects DataFrame, got {type(df).__name__}.")
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    # Column order + dtypes participate in the hash.
    for col in df.columns:
        h.update(str(col).encode("utf-8"))
        h.update(str(df[col].dtype).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=False).values
    h.update(np.ascontiguousarray(row_hashes).tobytes())
    return h.hexdigest()


def hash_array(arr: Any) -> str:
    """Return a stable hex digest of a numpy array (or array-like)."""
    a = np.asarray(arr)
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def hash_xy(X: Any, y: Any) -> str:
    """Combined hash of features + target."""
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    if isinstance(X, pd.DataFrame):
        h.update(hash_dataframe(X).encode("utf-8"))
    else:
        h.update(hash_array(X).encode("utf-8"))
    if isinstance(y, pd.Series):
        h.update(hash_array(y.values).encode("utf-8"))
    else:
        h.update(hash_array(y).encode("utf-8"))
    return h.hexdigest()

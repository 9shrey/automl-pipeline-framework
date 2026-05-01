"""Preprocessing zoo: imputers, encoders, scalers + factory."""

from automl.preprocessing.encoders import ENCODERS, FrequencyEncoder, build_encoder
from automl.preprocessing.factory import build_preprocessor
from automl.preprocessing.imputers import IMPUTERS, build_imputer
from automl.preprocessing.scalers import SCALERS, build_scaler

__all__ = [
    "ENCODERS",
    "IMPUTERS",
    "SCALERS",
    "FrequencyEncoder",
    "build_encoder",
    "build_imputer",
    "build_preprocessor",
    "build_scaler",
]

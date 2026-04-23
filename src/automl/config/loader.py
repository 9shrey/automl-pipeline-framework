"""Load + validate ``AutoMLConfig`` from a YAML file or a dict.

Validation runs at load time. Optional-dependency checks (e.g. CatBoost in the
search space without the package installed) live in ``check_runtime_deps`` so
callers can fail loudly *before* starting a search.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml

from automl.config.schema import AutoMLConfig

__all__ = ["check_runtime_deps", "load_config", "load_config_from_path"]


# Maps a model name in the search space to the import name that must exist.
_MODEL_IMPORT_NAMES: dict[str, str] = {
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}

# Selectors / preprocessors with optional deps.
_OPTIONAL_FEATURE_DEPS: dict[str, str] = {
    "shap": "shap",
}


def load_config(data: dict[str, Any]) -> AutoMLConfig:
    """Validate a dict into an ``AutoMLConfig`` (no I/O, no dep check)."""
    return AutoMLConfig.model_validate(data)


def load_config_from_path(path: str | Path) -> AutoMLConfig:
    """Load YAML at ``path`` and validate as ``AutoMLConfig``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw).__name__}.")
    return load_config(raw)


def check_runtime_deps(config: AutoMLConfig) -> None:
    """Raise ``ImportError`` if any item in the search space requires a
    package that is not importable. Run this *before* the search starts.
    """
    missing: list[tuple[str, str]] = []
    for model_name in config.search.space.models:
        import_name = _MODEL_IMPORT_NAMES.get(model_name)
        if import_name and importlib.util.find_spec(import_name) is None:
            missing.append((model_name, import_name))
    for fs_name in config.search.space.feature_selection:
        import_name = _OPTIONAL_FEATURE_DEPS.get(fs_name)
        if import_name and importlib.util.find_spec(import_name) is None:
            missing.append((fs_name, import_name))
    if missing:
        details = ", ".join(f"{name} (needs `{pkg}`)" for name, pkg in missing)
        raise ImportError(
            f"Search space requires packages that are not installed: {details}. "
            "Either install them or remove the entries from the config."
        )

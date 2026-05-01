"""Unit tests for ``automl.config`` (schema + loader)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from automl.config import (
    AutoMLConfig,
    check_runtime_deps,
    load_config,
    load_config_from_path,
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestAutoMLConfig:
    def test_minimal_classification(self) -> None:
        cfg = load_config({"task": "classification", "target": "y"})
        assert cfg.task == "classification"
        assert cfg.target == "y"
        # Defaults populated
        assert cfg.cv.scheme == "stratified_kfold"
        assert cfg.search.sampler == "tpe"
        assert cfg.ensembling.strategy == "stacking"

    def test_minimal_regression_uses_kfold(self) -> None:
        cfg = load_config(
            {"task": "regression", "target": "y", "cv": {"scheme": "kfold"}}
        )
        assert cfg.task == "regression"
        assert cfg.cv.scheme == "kfold"

    def test_regression_rejects_stratified_kfold(self) -> None:
        with pytest.raises(ValidationError, match="stratified_kfold"):
            load_config({"task": "regression", "target": "y"})

    def test_unknown_sampler_rejected(self) -> None:
        with pytest.raises(ValidationError):
            load_config(
                {
                    "task": "classification",
                    "target": "y",
                    "search": {"sampler": "unknown"},
                }
            )

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            load_config({"task": "classification", "target": "y", "bogus": 1})

    def test_n_jobs_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="n_jobs"):
            load_config({"task": "classification", "target": "y", "n_jobs": 0})

    def test_n_jobs_negative_below_minus_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match="n_jobs"):
            load_config({"task": "classification", "target": "y", "n_jobs": -2})

    def test_empty_search_space_list_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            load_config(
                {
                    "task": "classification",
                    "target": "y",
                    "search": {"space": {"models": []}},
                }
            )

    def test_n_splits_bounds(self) -> None:
        with pytest.raises(ValidationError):
            load_config(
                {"task": "classification", "target": "y", "cv": {"n_splits": 1}}
            )
        with pytest.raises(ValidationError):
            load_config(
                {"task": "classification", "target": "y", "cv": {"n_splits": 999}}
            )

    def test_round_trip_dump_then_load(self) -> None:
        cfg = load_config({"task": "classification", "target": "y"})
        cfg2 = AutoMLConfig.model_validate(cfg.model_dump())
        assert cfg == cfg2


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


class TestYamlLoader:
    def test_loads_default_yaml(self) -> None:
        cfg = load_config_from_path(Path("configs/default.yaml"))
        assert cfg.task == "classification"
        assert cfg.cv.n_splits == 5
        assert cfg.search.sampler == "tpe"

    def test_loads_fast_yaml(self) -> None:
        cfg = load_config_from_path(Path("configs/fast.yaml"))
        assert cfg.time_budget_seconds == 60
        assert cfg.meta_store.enabled is False

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config_from_path(tmp_path / "does_not_exist.yaml")

    def test_non_mapping_root_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(TypeError, match="mapping"):
            load_config_from_path(p)

    def test_yaml_validation_errors_propagate(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            dedent(
                """
                task: classification
                target: y
                search:
                  sampler: not_a_real_sampler
                """
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_config_from_path(p)


# ---------------------------------------------------------------------------
# Runtime dep check
# ---------------------------------------------------------------------------


class TestRuntimeDeps:
    def test_no_missing_when_only_sklearn_models(self) -> None:
        cfg = load_config(
            {
                "task": "classification",
                "target": "y",
                "search": {
                    "space": {
                        "models": ["logreg", "rf", "histgb"],
                        "feature_selection": ["none", "variance"],
                    }
                },
            }
        )
        # Should not raise.
        check_runtime_deps(cfg)

    def test_missing_xgboost_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.util as iu

        original = iu.find_spec

        def fake_find_spec(name: str, *a, **kw):  # type: ignore[no-untyped-def]
            if name == "xgboost":
                return None
            return original(name, *a, **kw)

        monkeypatch.setattr(iu, "find_spec", fake_find_spec)

        cfg = load_config(
            {
                "task": "classification",
                "target": "y",
                "search": {
                    "space": {
                        "models": ["xgboost", "logreg"],
                        "feature_selection": ["none"],
                    }
                },
            }
        )
        with pytest.raises(ImportError, match="xgboost"):
            check_runtime_deps(cfg)

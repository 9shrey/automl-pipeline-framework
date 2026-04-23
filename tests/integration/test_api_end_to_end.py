"""End-to-end integration test for ``AutoMLClassifier`` / ``AutoMLRegressor``.

These exercise the full orchestrator path: config → search → refit → recorder.
Kept tiny (small dataset, n_trials=4) so they run in a few seconds in CI.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from automl.api import AutoMLClassifier, AutoMLRegressor, load_pipeline


def _classification_frame(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    X_arr, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=seed,
    )
    cols = [f"f{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=cols), y


def _regression_frame(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    X_arr, y = make_regression(
        n_samples=120,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=seed,
    )
    cols = [f"f{i}" for i in range(X_arr.shape[1])]
    return pd.DataFrame(X_arr, columns=cols), y


@pytest.fixture
def cls_config(tmp_path: Path) -> dict:
    return {
        "task": "classification",
        "target": "y",
        "time_budget_seconds": 60,
        "n_trials": 4,
        "seed": 7,
        "n_jobs": 1,
        "cv": {"scheme": "stratified_kfold", "n_splits": 3, "shuffle": True},
        "search": {
            "sampler": "random",
            "pruner": "none",
            "warm_start": {"enabled": False},
            "space": {
                "imputation": ["mean"],
                "encoding": ["onehot"],
                "scaling": ["standard", "none"],
                "feature_selection": ["none"],
                "models": ["logreg", "rf"],
            },
        },
        "ensembling": {"enabled": False, "strategy": "none"},
        "explain": {"enabled": False},
        "run": {"out_dir": str(tmp_path / "runs")},
        "meta_store": {"enabled": False, "path": str(tmp_path / "meta.sqlite")},
    }


@pytest.fixture
def reg_config(tmp_path: Path) -> dict:
    return {
        "task": "regression",
        "target": "y",
        "time_budget_seconds": 60,
        "n_trials": 4,
        "seed": 7,
        "n_jobs": 1,
        "cv": {"scheme": "kfold", "n_splits": 3, "shuffle": True},
        "search": {
            "sampler": "random",
            "pruner": "none",
            "warm_start": {"enabled": False},
            "space": {
                "imputation": ["mean"],
                "encoding": ["onehot"],
                "scaling": ["standard"],
                "feature_selection": ["none"],
                "models": ["ridge", "rf"],
            },
        },
        "ensembling": {"enabled": False, "strategy": "none"},
        "explain": {"enabled": False},
        "run": {"out_dir": str(tmp_path / "runs")},
        "meta_store": {"enabled": False, "path": str(tmp_path / "meta.sqlite")},
    }


class TestAutoMLClassifier:
    def test_fit_predict_end_to_end(self, cls_config):
        X, y = _classification_frame()
        auto = AutoMLClassifier(config=cls_config)
        auto.fit(X, y)

        # Sklearn-API surface.
        preds = auto.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(np.unique(preds)) <= set(np.unique(y))

        proba = auto.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

        score = auto.score(X, y)
        assert 0.0 <= score <= 1.0

        # Fitted attributes.
        assert auto.metric_name_ == "roc_auc"
        assert 0.0 <= auto.best_score_ <= 1.0
        assert auto.n_features_in_ == X.shape[1]
        np.testing.assert_array_equal(auto.classes_, np.array([0, 1]))
        assert isinstance(auto.leaderboard_, pd.DataFrame)
        assert not auto.leaderboard_.empty
        assert "score" in auto.leaderboard_.columns
        # Sorted best-first → desc for roc_auc (max direction).
        scores = auto.leaderboard_["score"].to_numpy()
        assert np.all(np.diff(scores) <= 1e-12)

    def test_run_artifacts_written(self, cls_config):
        X, y = _classification_frame()
        auto = AutoMLClassifier(config=cls_config).fit(X, y)
        run_dir = auto.run_dir_
        assert run_dir.is_dir()
        for fname in (
            "config.yaml",
            "dataset.json",
            "leaderboard.csv",
            "best_pipeline.pkl",
            "best_trial.json",
            "run_card.md",
        ):
            assert (run_dir / fname).is_file(), f"missing artifact: {fname}"

        ds = json.loads((run_dir / "dataset.json").read_text())
        assert ds["n_rows"] == X.shape[0]
        assert ds["n_cols"] == X.shape[1]
        assert ds["target"] == "y"
        assert isinstance(ds["hash"], str) and len(ds["hash"]) == 32

        bt = json.loads((run_dir / "best_trial.json").read_text())
        assert bt["metric"] == "roc_auc"
        assert "model_name" in bt["user_attrs"]

    def test_save_and_load_pipeline(self, cls_config, tmp_path):
        X, y = _classification_frame()
        auto = AutoMLClassifier(config=cls_config).fit(X, y)
        out = tmp_path / "model.pkl"
        auto.save(out)
        loaded = load_pipeline(out)
        np.testing.assert_array_equal(loaded.predict(X), auto.predict(X))

    def test_recorded_pipeline_matches_in_memory(self, cls_config):
        X, y = _classification_frame()
        auto = AutoMLClassifier(config=cls_config).fit(X, y)
        with (auto.run_dir_ / "best_pipeline.pkl").open("rb") as f:
            disk_pipeline = pickle.load(f)
        np.testing.assert_array_equal(disk_pipeline.predict(X), auto.predict(X))

    def test_rejects_mismatched_task_in_config(self, reg_config):
        with pytest.raises(ValueError, match="task='regression'"):
            AutoMLClassifier(config=reg_config).fit(*_classification_frame())

    def test_accepts_numpy_arrays(self, cls_config):
        X_df, y = _classification_frame()
        X_arr = X_df.to_numpy()
        auto = AutoMLClassifier(config=cls_config).fit(X_arr, y)
        preds = auto.predict(X_arr)
        assert preds.shape == (X_arr.shape[0],)


class TestAutoMLRegressor:
    def test_fit_predict_end_to_end(self, reg_config):
        X, y = _regression_frame()
        auto = AutoMLRegressor(config=reg_config).fit(X, y)
        preds = auto.predict(X)
        assert preds.shape == (X.shape[0],)
        # Sklearn's RegressorMixin.score returns R²; can be negative on hard data
        # but should be > 0 for these well-behaved synthetic problems.
        assert auto.score(X, y) > 0.0
        assert auto.metric_name_ == "rmse"
        # rmse → min direction; leaderboard sorted ascending.
        scores = auto.leaderboard_["score"].to_numpy()
        assert np.all(np.diff(scores) >= -1e-12)

    def test_artifacts_round_trip(self, reg_config):
        X, y = _regression_frame()
        auto = AutoMLRegressor(config=reg_config).fit(X, y)
        loaded = load_pipeline(auto.run_dir_ / "best_pipeline.pkl")
        np.testing.assert_allclose(loaded.predict(X), auto.predict(X))

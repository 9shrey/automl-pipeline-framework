"""Tests for the runs store: list/load past runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from automl.api import AutoMLClassifier
from automl.runs.store import list_runs, load_pipeline_from_run, load_run


def _quick_config(tmp_path) -> dict:
    return {
        "task": "classification",
        "target": "y",
        "time_budget_seconds": 60,
        "n_trials": 3,
        "seed": 0,
        "n_jobs": 1,
        "cv": {"scheme": "stratified_kfold", "n_splits": 3, "shuffle": True},
        "search": {
            "sampler": "random", "pruner": "none", "warm_start": {"enabled": False},
            "space": {
                "imputation": ["mean"], "encoding": ["onehot"], "scaling": ["none"],
                "feature_selection": ["none"], "models": ["logreg"],
            },
        },
        "ensembling": {"enabled": False, "strategy": "none"},
        "explain": {"enabled": False},
        "run": {"out_dir": str(tmp_path / "runs")},
        "meta_store": {"enabled": False, "path": str(tmp_path / "meta.sqlite")},
    }


@pytest.fixture
def trained(tmp_path):
    X_arr, y = make_classification(n_samples=80, n_features=5, random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])
    auto = AutoMLClassifier(config=_quick_config(tmp_path)).fit(X, y)
    return auto, X, y, tmp_path


def test_list_runs_returns_handle(trained):
    auto, _, _, tmp_path = trained
    out_dir = tmp_path / "runs"
    handles = list_runs(out_dir)
    assert len(handles) == 1
    h = handles[0]
    assert h.run_dir == auto.run_dir_
    assert h.metric == "roc_auc"
    assert h.task == "classification"
    assert h.target == "y"
    assert h.n_rows == 80
    assert h.n_cols == 5


def test_list_runs_missing_dir_returns_empty(tmp_path):
    assert list_runs(tmp_path / "does-not-exist") == []


def test_load_run_returns_full_artifacts(trained):
    auto, X, y, _ = trained
    info = load_run(auto.run_dir_)
    assert set(info) == {"run_dir", "config", "dataset", "best_trial", "leaderboard", "pipeline"}
    assert isinstance(info["leaderboard"], pd.DataFrame)
    np.testing.assert_array_equal(info["pipeline"].predict(X), auto.predict(X))


def test_load_run_rejects_invalid(tmp_path):
    (tmp_path / "junk").mkdir()
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "junk")


def test_load_pipeline_from_run(trained):
    auto, X, _, _ = trained
    pipe = load_pipeline_from_run(auto.run_dir_)
    np.testing.assert_array_equal(pipe.predict(X), auto.predict(X))

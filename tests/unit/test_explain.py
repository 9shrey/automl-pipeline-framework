"""Unit tests for explain modules and api integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from automl.api import AutoMLClassifier
from automl.explain.permutation import permutation_feature_importance
from automl.explain.reports import write_importance_report


def test_permutation_importance_returns_sorted_dataframe():
    X_arr, y = make_classification(n_samples=80, n_features=4, random_state=0)
    X = pd.DataFrame(X_arr, columns=["a", "b", "c", "d"])
    model = LogisticRegression(max_iter=1000).fit(X, y)
    df = permutation_feature_importance(model, X, y, n_repeats=3, seed=0)
    assert list(df.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(df["feature"]) == {"a", "b", "c", "d"}
    means = df["importance_mean"].to_numpy()
    assert np.all(np.diff(means) <= 1e-12)


def test_write_importance_report(tmp_path):
    perm = pd.DataFrame({"feature": ["a", "b"], "importance_mean": [0.5, 0.1], "importance_std": [0.0, 0.0]})
    paths = write_importance_report(out_dir=tmp_path, permutation=perm, shap=None, top_n=5)
    assert "permutation" in paths and paths["permutation"].is_file()
    assert "explain" in paths
    text = paths["explain"].read_text(encoding="utf-8")
    assert "Permutation importance" in text


def test_write_importance_report_empty_inputs(tmp_path):
    paths = write_importance_report(out_dir=tmp_path)
    assert paths["explain"].is_file()
    assert "no importance signals" in paths["explain"].read_text(encoding="utf-8")


def _quick_cfg(tmp_path, *, explain: bool, meta: bool) -> dict:
    return {
        "task": "classification",
        "target": "y",
        "time_budget_seconds": 60,
        "n_trials": 3,
        "seed": 0,
        "n_jobs": 1,
        "cv": {"scheme": "stratified_kfold", "n_splits": 3, "shuffle": True},
        "search": {
            "sampler": "random", "pruner": "none",
            "warm_start": {"enabled": meta, "top_k_neighbors": 3, "min_overlap_meta_features": 0.5},
            "space": {
                "imputation": ["mean"], "encoding": ["onehot"], "scaling": ["none"],
                "feature_selection": ["none"], "models": ["logreg"],
            },
        },
        "ensembling": {"enabled": False, "strategy": "none"},
        "explain": {"enabled": explain, "shap_max_samples": 100, "permutation_repeats": 2},
        "run": {"out_dir": str(tmp_path / "runs")},
        "meta_store": {"enabled": meta, "path": str(tmp_path / "meta.sqlite")},
    }


def test_api_emits_explain_artifacts(tmp_path):
    X_arr, y = make_classification(n_samples=80, n_features=4, random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(4)])
    auto = AutoMLClassifier(config=_quick_cfg(tmp_path, explain=True, meta=False)).fit(X, y)
    assert "explain" in auto.explain_paths_
    explain_md = auto.run_dir_ / "explain.md"
    assert explain_md.is_file()
    perm_csv = auto.run_dir_ / "permutation_importance.csv"
    assert perm_csv.is_file()
    df = pd.read_csv(perm_csv)
    assert set(df.columns) >= {"feature", "importance_mean"}


def test_api_skips_explain_when_disabled(tmp_path):
    X_arr, y = make_classification(n_samples=80, n_features=4, random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(4)])
    auto = AutoMLClassifier(config=_quick_cfg(tmp_path, explain=False, meta=False)).fit(X, y)
    assert auto.explain_paths_ == {}
    assert not (auto.run_dir_ / "explain.md").exists()

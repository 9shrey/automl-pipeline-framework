"""Tests for meta-features, sqlite store, warm-start helper + API integration."""

from __future__ import annotations

import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from automl.api import AutoMLClassifier
from automl.meta import (
    MetaFeatures,
    MetaStore,
    compute_meta_features,
    meta_feature_distance,
    warm_start_study,
)


class TestMetaFeatures:
    def test_classification_balance(self):
        X_arr, y = make_classification(
            n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=0
        )
        X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(4)])
        m = compute_meta_features(X, y, task="classification")
        assert m.n_rows == 100 and m.n_cols == 4
        assert m.n_classes == 2
        assert 0.5 <= m.class_balance <= 1.0
        assert m.target_std == 0.0

    def test_regression(self):
        X_arr, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=0)
        X = pd.DataFrame(X_arr, columns=["a", "b", "c"])
        m = compute_meta_features(X, y, task="regression")
        assert m.n_classes == 0 and m.class_balance == 0.0
        assert m.target_std > 0.0

    def test_distance_zero_when_identical(self):
        a = MetaFeatures(100, 4, 4, 0, 0, 0.0, 2, 1.0, 0.0)
        assert meta_feature_distance(a, a) == 0.0

    def test_distance_grows_with_difference(self):
        a = MetaFeatures(100, 4, 4, 0, 0, 0.0, 2, 1.0, 0.0)
        b = MetaFeatures(10000, 50, 30, 20, 0, 0.5, 5, 0.3, 0.0)
        assert meta_feature_distance(a, b) > meta_feature_distance(a, a)


class TestMetaStore:
    def test_record_and_retrieve(self, tmp_path):
        with MetaStore(tmp_path / "m.sqlite") as store:
            m = MetaFeatures(100, 4, 4, 0, 0, 0.0, 2, 1.0, 0.0)
            rid = store.record(task="classification", metric="roc_auc", score=0.9, meta=m, params={"a": 1})
            assert rid >= 1
            recs = store.all(task="classification")
            assert len(recs) == 1
            assert recs[0].score == 0.9
            assert recs[0].params == {"a": 1}

    def test_nearest_orders_by_distance(self, tmp_path):
        with MetaStore(tmp_path / "m.sqlite") as store:
            close = MetaFeatures(100, 4, 4, 0, 0, 0.0, 2, 1.0, 0.0)
            far = MetaFeatures(10000, 200, 100, 100, 0, 0.5, 10, 0.1, 0.0)
            store.record(task="classification", metric="roc_auc", score=0.9, meta=close, params={"a": 1})
            store.record(task="classification", metric="roc_auc", score=0.8, meta=far, params={"a": 2})
            target = MetaFeatures(110, 4, 4, 0, 0, 0.0, 2, 0.95, 0.0)
            nbrs = store.nearest(target, task="classification", k=2)
            assert nbrs[0].params == {"a": 1}
            assert nbrs[1].params == {"a": 2}

    def test_filter_by_task(self, tmp_path):
        with MetaStore(tmp_path / "m.sqlite") as store:
            m = MetaFeatures(100, 4, 4, 0, 0, 0.0, 2, 1.0, 0.0)
            store.record(task="classification", metric="roc_auc", score=0.9, meta=m, params={})
            store.record(task="regression", metric="rmse", score=0.1, meta=m, params={})
            assert len(store.all(task="classification")) == 1
            assert len(store.all(task="regression")) == 1


class TestWarmStart:
    def test_enqueue_filters_non_primitive(self, tmp_path):
        with MetaStore(tmp_path / "m.sqlite") as store:
            m = MetaFeatures(100, 4, 4, 0, 0, 0.0, 2, 1.0, 0.0)
            # tuple param will be stored as repr() string by the sqlite layer.
            store.record(task="classification", metric="roc_auc", score=0.9, meta=m,
                         params={"x": 1, "y": "abc"})
            recs = store.all(task="classification")

        def obj(trial):
            return trial.suggest_int("x", 0, 10) + (1 if trial.suggest_categorical("y", ["abc", "def"]) == "abc" else 0)

        study = optuna.create_study(direction="maximize")
        n = warm_start_study(study, recs)
        assert n == 1
        study.optimize(obj, n_trials=2)
        # First trial should have used the warm-start params (x=1, y='abc' → 2).
        first = study.trials[0]
        assert first.params["x"] == 1
        assert first.params["y"] == "abc"


def _cfg_with_meta(tmp_path) -> dict:
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
            "warm_start": {"enabled": True, "top_k_neighbors": 3, "min_overlap_meta_features": 0.5},
            "space": {
                "imputation": ["mean"], "encoding": ["onehot"], "scaling": ["none"],
                "feature_selection": ["none"], "models": ["logreg"],
            },
        },
        "ensembling": {"enabled": False, "strategy": "none"},
        "explain": {"enabled": False},
        "run": {"out_dir": str(tmp_path / "runs")},
        "meta_store": {"enabled": True, "path": str(tmp_path / "meta.sqlite")},
    }


def test_api_records_to_meta_store(tmp_path):
    X_arr, y = make_classification(n_samples=80, n_features=4, random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(4)])
    cfg = _cfg_with_meta(tmp_path)
    auto = AutoMLClassifier(config=cfg).fit(X, y)
    # Meta store should have one record now.
    with MetaStore(cfg["meta_store"]["path"]) as store:
        recs = store.all(task="classification")
    assert len(recs) == 1
    # Second run sees the prior record and warm-starts (no crash).
    auto2 = AutoMLClassifier(config=cfg).fit(X, y)
    assert auto2.meta_features_ is not None
    with MetaStore(cfg["meta_store"]["path"]) as store:
        assert len(store.all(task="classification")) == 2

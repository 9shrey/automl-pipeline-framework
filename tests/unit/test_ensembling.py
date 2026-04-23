"""Tests for ensembling: selector + voting/stacking + build_ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from automl.api import AutoMLClassifier, AutoMLRegressor


@pytest.fixture
def cls_xy():
    X, y = make_classification(
        n_samples=140, n_features=6, n_informative=4, n_redundant=0,
        n_classes=2, random_state=0,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y


@pytest.fixture
def reg_xy():
    X, y = make_regression(n_samples=140, n_features=6, n_informative=4, noise=0.1, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), y


def _cls_config_with_ensembling(strategy: str, tmp_path) -> dict:
    return {
        "task": "classification",
        "target": "y",
        "time_budget_seconds": 60,
        "n_trials": 6,
        "seed": 11,
        "n_jobs": 1,
        "cv": {"scheme": "stratified_kfold", "n_splits": 3, "shuffle": True},
        "search": {
            "sampler": "random",
            "pruner": "none",
            "warm_start": {"enabled": False},
            "space": {
                "imputation": ["mean"], "encoding": ["onehot"],
                "scaling": ["standard"], "feature_selection": ["none"],
                "models": ["logreg", "rf"],
            },
        },
        "ensembling": {"enabled": True, "strategy": strategy, "top_k": 3,
                       "diversity": "metric_pareto", "meta_learner": "logreg"},
        "explain": {"enabled": False},
        "run": {"out_dir": str(tmp_path / "runs")},
        "meta_store": {"enabled": False, "path": str(tmp_path / "meta.sqlite")},
    }


class TestSelector:
    def test_select_top_k_basic(self):
        from unittest.mock import MagicMock

        from automl.ensembling.selector import select_top_k

        # Fake leaderboard sorted desc.
        lb = pd.DataFrame([
            {"trial_number": 0, "score": 0.9, "model": "rf",
             "imputation": "mean", "encoding": "onehot", "scaling": "standard", "feature_selection": "none"},
            {"trial_number": 1, "score": 0.8, "model": "rf",
             "imputation": "mean", "encoding": "onehot", "scaling": "standard", "feature_selection": "none"},
            {"trial_number": 2, "score": 0.7, "model": "logreg",
             "imputation": "mean", "encoding": "onehot", "scaling": "standard", "feature_selection": "none"},
        ])
        study = MagicMock()
        t0, t1, t2 = MagicMock(), MagicMock(), MagicMock()
        t0.number, t0.params = 0, {"model": "rf", "rf_n_estimators": 100}
        t1.number, t1.params = 1, {"model": "rf", "rf_n_estimators": 200}
        t2.number, t2.params = 2, {"model": "logreg"}
        study.trials = [t0, t1, t2]

        # Top-k=2 strict.
        out = select_top_k(lb, study, k=2, strategy="none")
        assert [s.trial_number for s in out] == [0, 1]

        # Diversity → one rf, one logreg.
        out = select_top_k(lb, study, k=2, strategy="metric_pareto")
        assert [s.model_name for s in out] == ["rf", "logreg"]

    def test_empty_leaderboard(self):
        from automl.ensembling.selector import select_top_k

        out = select_top_k(pd.DataFrame(), study=None, k=3)
        assert out == []


class TestEnsembleEndToEnd:
    @pytest.mark.parametrize("strategy", ["voting", "stacking"])
    def test_classification_ensemble_fits_and_predicts(self, cls_xy, tmp_path, strategy):
        X, y = cls_xy
        cfg = _cls_config_with_ensembling(strategy, tmp_path)
        auto = AutoMLClassifier(config=cfg).fit(X, y)
        # Ensemble was actually used (not None).
        assert auto.ensemble_ is not None
        assert auto.pipeline_ is auto.ensemble_
        # Predictions work and probabilities sum to 1.
        proba = auto.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        # Recorded artifact equals the fitted ensemble.
        from automl.runs.store import load_pipeline_from_run

        loaded = load_pipeline_from_run(auto.run_dir_)
        np.testing.assert_array_equal(loaded.predict(X), auto.predict(X))

    def test_regression_voting(self, reg_xy, tmp_path):
        X, y = reg_xy
        cfg = _cls_config_with_ensembling("voting", tmp_path)
        cfg.update({
            "task": "regression",
            "cv": {"scheme": "kfold", "n_splits": 3, "shuffle": True},
        })
        cfg["search"]["space"]["models"] = ["ridge", "rf"]
        auto = AutoMLRegressor(config=cfg).fit(X, y)
        assert auto.ensemble_ is not None
        preds = auto.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_disabled_falls_back_to_single_pipeline(self, cls_xy, tmp_path):
        X, y = cls_xy
        cfg = _cls_config_with_ensembling("none", tmp_path)
        cfg["ensembling"]["enabled"] = False
        cfg["ensembling"]["strategy"] = "none"
        auto = AutoMLClassifier(config=cfg).fit(X, y)
        assert auto.ensemble_ is None
        assert auto.pipeline_ is auto.best_pipeline_

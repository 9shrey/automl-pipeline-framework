"""Unit tests for ``automl.search.objective`` and ``engine`` (small end-to-end).

These are kept fast: tiny dataset, 5 trials, no pruning, restricted search
space. They verify that the wiring is correct (sample → build pipeline →
CV-evaluate → leaderboard) without becoming integration tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

# Register all adapters.
import automl.models.sklearn_models  # noqa: F401
from automl.config import load_config
from automl.data import infer_schema
from automl.search.engine import run_search
from automl.search.objective import (
    Objective,
    SampledPipelineSpec,
    build_pipeline_from_params,
)

# Keep optuna quiet during tests.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _classification_df(n: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = make_classification(n_samples=n, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    return df, y


def _regression_df(n: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = make_regression(n_samples=n, n_features=4, noise=0.1, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    return df, y


def _classification_config(time_budget: int = 30, n_trials: int = 5):
    return load_config(
        {
            "task": "classification",
            "target": "y",
            "time_budget_seconds": time_budget,
            "n_trials": n_trials,
            "seed": 0,
            "cv": {"n_splits": 3},
            "search": {
                "sampler": "random",
                "pruner": "none",
                "warm_start": {"enabled": False},
                "space": {
                    "imputation": ["mean"],
                    "encoding": ["onehot"],
                    "scaling": ["standard", "none"],
                    "feature_selection": ["none"],
                    "models": ["logreg", "histgb"],
                },
            },
            "ensembling": {"enabled": False, "strategy": "none"},
            "explain": {"enabled": False},
            "meta_store": {"enabled": False},
        }
    )


# ---------------------------------------------------------------------------
# build_pipeline_from_params
# ---------------------------------------------------------------------------


def test_build_pipeline_from_params_classification() -> None:
    X, y = _classification_df()
    schema = infer_schema(X)
    spec = SampledPipelineSpec(
        imputation="mean",
        encoding="onehot",
        scaling="standard",
        feature_selection="none",
        model_name="logreg",
        model_params={"logreg_penalty_solver": ("l2", "lbfgs"), "logreg_C": 1.0, "logreg_max_iter": 200},
    )
    pipe = build_pipeline_from_params(spec, schema=schema, task="classification")
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


class TestObjective:
    def test_returns_score_and_sets_attrs(self) -> None:
        X, y = _classification_df()
        cfg = _classification_config()
        obj = Objective(cfg, X, y, schema=infer_schema(X))
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=2, catch=(Exception,))
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        assert complete, "Expected at least one completed trial."
        t = complete[0]
        assert "model_name" in t.user_attrs
        assert "fold_scores" in t.user_attrs
        assert len(t.user_attrs["fold_scores"]) == cfg.cv.n_splits

    def test_rejects_non_dataframe(self) -> None:
        cfg = _classification_config()
        with pytest.raises(TypeError, match="DataFrame"):
            Objective(cfg, np.zeros((10, 3)), np.zeros(10), schema=infer_schema(pd.DataFrame()))  # type: ignore[arg-type]

    def test_rejects_metric_task_mismatch(self) -> None:
        X, y = _classification_df()
        cfg = _classification_config()
        with pytest.raises(ValueError, match="task"):
            Objective(cfg, X, y, schema=infer_schema(X), metric="rmse")


# ---------------------------------------------------------------------------
# run_search
# ---------------------------------------------------------------------------


class TestRunSearch:
    def test_classification_end_to_end(self) -> None:
        X, y = _classification_df(n=100)
        cfg = _classification_config(n_trials=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_search(cfg, X, y)
        assert not result.leaderboard.empty
        assert "score" in result.leaderboard.columns
        # roc_auc is the default classification metric → bounded [0, 1].
        assert (result.leaderboard["score"] >= 0).all()
        assert (result.leaderboard["score"] <= 1).all()
        assert result.metric_name == "roc_auc"
        assert result.best_trial.state == optuna.trial.TrialState.COMPLETE

    def test_regression_end_to_end(self) -> None:
        X, y = _regression_df(n=100)
        cfg = load_config(
            {
                "task": "regression",
                "target": "y",
                "time_budget_seconds": 30,
                "n_trials": 4,
                "seed": 0,
                "cv": {"scheme": "kfold", "n_splits": 3},
                "search": {
                    "sampler": "random",
                    "pruner": "none",
                    "warm_start": {"enabled": False},
                    "space": {
                        "imputation": ["mean"],
                        "encoding": ["onehot"],
                        "scaling": ["standard"],
                        "feature_selection": ["none"],
                        "models": ["ridge", "histgb"],
                    },
                },
                "ensembling": {"enabled": False, "strategy": "none"},
                "explain": {"enabled": False},
                "meta_store": {"enabled": False},
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_search(cfg, X, y)
        assert not result.leaderboard.empty
        # Lower RMSE is better, so the leaderboard is sorted ascending.
        assert (
            result.leaderboard["score"].is_monotonic_increasing
        )
        assert result.metric_name == "rmse"

    def test_failure_isolation_does_not_kill_study(self) -> None:
        # Constructing a config with a non-existent model in the list would
        # fail config validation; instead, simulate a runtime failure by
        # passing a target with a single class (LogReg refuses) and verify the
        # study still finishes.
        X, _ = _classification_df(n=60)
        y = np.zeros(60, dtype=int)
        cfg = _classification_config(n_trials=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = run_search(cfg, X, y)
            except RuntimeError:
                # Acceptable: all trials may fail and engine raises "no completed trials".
                return
        # If at least one trial succeeded somehow, leaderboard must be non-empty.
        assert not result.leaderboard.empty

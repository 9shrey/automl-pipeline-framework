"""End-to-end performance smoke test using the fast-CI config.

Asserts that an AutoML run on a small synthetic dataset completes within a
generous time budget and produces a non-empty leaderboard. Intentionally
loose: this is a smoke gate, not a benchmark.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from automl.api import AutoMLClassifier
from automl.config.loader import load_config_from_path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAST_CFG_PATH = _REPO_ROOT / "configs" / "fast.yaml"


@pytest.mark.slow
def test_fast_config_runs_under_budget(tmp_path):
    cfg = load_config_from_path(_FAST_CFG_PATH)
    # Keep models to ones that don't need optional deps in this venv.
    cfg = cfg.model_copy(update={
        "n_trials": 6,
        "time_budget_seconds": 60,
    })
    cfg.search.space.models[:] = [m for m in cfg.search.space.models if m in {"logreg", "histgb"}]
    cfg.search.space.feature_selection[:] = ["none"]
    cfg.run.out_dir = str(tmp_path / "runs")

    X_arr, y = make_classification(n_samples=200, n_features=8, n_informative=5, random_state=0)
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])

    auto = AutoMLClassifier(config=cfg)
    t0 = time.perf_counter()
    auto.fit(X, y)
    wall = time.perf_counter() - t0

    # Allow 3x overhead vs the configured budget for variance on shared CI.
    assert wall < cfg.time_budget_seconds * 3, f"fit took {wall:.1f}s"
    assert not auto.leaderboard_.empty
    assert auto.run_dir_.is_dir()
    assert (auto.run_dir_ / "best_pipeline.pkl").is_file()

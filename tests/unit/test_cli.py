"""Tests for the typer-based ``automl`` CLI."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification
from typer.testing import CliRunner

from automl.cli import app

runner = CliRunner()


@pytest.fixture
def tiny_dataset_path(tmp_path):
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["label"] = y
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def quick_config_path(tmp_path):
    cfg = {
        "task": "classification",
        "target": "label",
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
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_fit_command_writes_run(tiny_dataset_path, quick_config_path, tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        ["fit", "--config", str(quick_config_path), "--data", str(tiny_dataset_path),
         "--target", "label", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "best roc_auc" in result.output
    runs_in_out = list(out.iterdir())
    assert len(runs_in_out) == 1


def test_full_pipeline_score_predict_leaderboard_runs(tiny_dataset_path, quick_config_path, tmp_path):
    out = tmp_path / "out"
    fit_res = runner.invoke(
        app,
        ["fit", "--config", str(quick_config_path), "--data", str(tiny_dataset_path),
         "--target", "label", "--out", str(out)],
    )
    assert fit_res.exit_code == 0
    run_dir = next(out.iterdir())

    score_res = runner.invoke(
        app,
        ["score", "--run", str(run_dir), "--data", str(tiny_dataset_path), "--target", "label"],
    )
    assert score_res.exit_code == 0, score_res.output
    assert "score:" in score_res.output

    pred_path = tmp_path / "preds.csv"
    pred_res = runner.invoke(
        app,
        ["predict", "--run", str(run_dir), "--data", str(tiny_dataset_path),
         "--target", "label", "--out", str(pred_path)],
    )
    assert pred_res.exit_code == 0, pred_res.output
    assert pred_path.is_file()
    preds = pd.read_csv(pred_path)
    assert "prediction" in preds.columns and len(preds) == 80

    lb_res = runner.invoke(app, ["leaderboard", "--run", str(run_dir), "--top", "3"])
    assert lb_res.exit_code == 0
    assert "score" in lb_res.output

    runs_res = runner.invoke(app, ["runs", "--out-dir", str(out)])
    assert runs_res.exit_code == 0
    assert run_dir.name in runs_res.output


def test_runs_empty_dir(tmp_path):
    res = runner.invoke(app, ["runs", "--out-dir", str(tmp_path / "empty")])
    assert res.exit_code == 0
    assert "no runs found" in res.output

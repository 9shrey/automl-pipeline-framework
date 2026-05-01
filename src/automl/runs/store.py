"""Load and list persisted AutoML runs."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

__all__ = ["RunHandle", "list_runs", "load_pipeline_from_run", "load_run"]


@dataclass(frozen=True)
class RunHandle:
    run_dir: Path
    run_id: str
    task: str
    target: str
    metric: str
    score: float
    n_rows: int
    n_cols: int


_REQUIRED = {
    "config.yaml",
    "dataset.json",
    "leaderboard.csv",
    "best_pipeline.pkl",
    "best_trial.json",
}


def _require_run_dir(run_dir: Path) -> None:
    missing = [name for name in sorted(_REQUIRED) if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is not a valid AutoML run; missing {missing}")


def load_run(run: str | Path) -> dict[str, Any]:
    run_dir = Path(run)
    _require_run_dir(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    dataset = json.loads((run_dir / "dataset.json").read_text(encoding="utf-8"))
    best_trial = json.loads((run_dir / "best_trial.json").read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(run_dir / "leaderboard.csv")
    pipeline = load_pipeline_from_run(run_dir)
    return {
        "run_dir": run_dir,
        "config": config,
        "dataset": dataset,
        "best_trial": best_trial,
        "leaderboard": leaderboard,
        "pipeline": pipeline,
    }


def load_pipeline_from_run(run: str | Path) -> Any:
    run_dir = Path(run)
    _require_run_dir(run_dir)
    with (run_dir / "best_pipeline.pkl").open("rb") as fh:
        return pickle.load(fh)


def list_runs(out_dir: str | Path) -> list[RunHandle]:
    root = Path(out_dir)
    if not root.exists():
        return []
    handles: list[RunHandle] = []
    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        try:
            info = load_run(run_dir)
        except FileNotFoundError:
            continue
        dataset = info["dataset"]
        best_trial = info["best_trial"]
        config = info["config"]
        handles.append(
            RunHandle(
                run_dir=run_dir,
                run_id=run_dir.name,
                task=str(config["task"]),
                target=str(dataset["target"]),
                metric=str(best_trial["metric"]),
                score=float(best_trial["score"]),
                n_rows=int(dataset["n_rows"]),
                n_cols=int(dataset["n_cols"]),
            )
        )
    return handles

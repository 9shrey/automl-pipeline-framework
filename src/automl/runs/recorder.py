"""Persist fitted AutoML run artifacts."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from automl.config.schema import AutoMLConfig
from automl.data.hashing import hash_xy
from automl.data.schema_infer import ColumnSchema

__all__ = ["RunArtifacts", "RunRecorder"]


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    config_path: Path
    dataset_path: Path
    leaderboard_path: Path
    best_pipeline_path: Path
    best_trial_path: Path
    run_card_path: Path


class RunRecorder:
    """Write a complete, reloadable run directory."""

    def __init__(self, config: AutoMLConfig, *, root: Path | None = None) -> None:
        self.config = config
        self.root = root or Path(config.run.out_dir)

    def finalize(
        self,
        *,
        X: pd.DataFrame,
        y,
        schema: ColumnSchema,
        leaderboard: pd.DataFrame,
        best_pipeline,
        best_trial,
        metric_name: str,
        best_score: float,
        n_trials_completed: int,
        wall_time_seconds: float,
    ) -> RunArtifacts:
        run_dir = self._new_run_dir()
        run_dir.mkdir(parents=True, exist_ok=False)

        config_path = run_dir / "config.yaml"
        dataset_path = run_dir / "dataset.json"
        leaderboard_path = run_dir / "leaderboard.csv"
        best_pipeline_path = run_dir / "best_pipeline.pkl"
        best_trial_path = run_dir / "best_trial.json"
        run_card_path = run_dir / "run_card.md"

        config_path.write_text(
            yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=True),
            encoding="utf-8",
        )
        dataset = {
            "target": self.config.target,
            "task": self.config.task,
            "n_rows": int(X.shape[0]),
            "n_cols": int(X.shape[1]),
            "hash": hash_xy(X, y),
            "schema": {
                "numeric": list(schema.numeric),
                "boolean": list(schema.boolean),
                "low_card_categorical": list(schema.low_card_categorical),
                "high_card_categorical": list(schema.high_card_categorical),
            },
        }
        dataset_path.write_text(json.dumps(dataset, indent=2, sort_keys=True), encoding="utf-8")
        leaderboard.to_csv(leaderboard_path, index=False)
        with best_pipeline_path.open("wb") as fh:
            pickle.dump(best_pipeline, fh)

        best_trial_doc = {
            "number": best_trial.number,
            "metric": metric_name,
            "score": float(best_score),
            "value": None if best_trial.value is None else float(best_trial.value),
            "params": dict(best_trial.params),
            "user_attrs": dict(best_trial.user_attrs),
        }
        best_trial_path.write_text(
            json.dumps(best_trial_doc, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        run_card_path.write_text(
            self._run_card(
                dataset=dataset,
                metric_name=metric_name,
                best_score=best_score,
                n_trials_completed=n_trials_completed,
                wall_time_seconds=wall_time_seconds,
                leaderboard=leaderboard,
            ),
            encoding="utf-8",
        )

        return RunArtifacts(
            run_dir=run_dir,
            config_path=config_path,
            dataset_path=dataset_path,
            leaderboard_path=leaderboard_path,
            best_pipeline_path=best_pipeline_path,
            best_trial_path=best_trial_path,
            run_card_path=run_card_path,
        )

    def _new_run_dir(self) -> Path:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return self.root / f"{ts}-{uuid4().hex[:8]}"

    def _run_card(
        self,
        *,
        dataset: dict,
        metric_name: str,
        best_score: float,
        n_trials_completed: int,
        wall_time_seconds: float,
        leaderboard: pd.DataFrame,
    ) -> str:
        git_sha = os.environ.get("GITHUB_SHA") or os.environ.get("AUTOML_GIT_SHA") or "unknown"
        top = _markdown_table(leaderboard.head(10)) if not leaderboard.empty else "_empty_"
        return (
            "# AutoML Run Card\n\n"
            f"- Task: {self.config.task}\n"
            f"- Target: {self.config.target}\n"
            f"- Git SHA: {git_sha}\n"
            f"- Dataset hash: {dataset['hash']}\n"
            f"- Rows: {dataset['n_rows']}\n"
            f"- Columns: {dataset['n_cols']}\n"
            f"- Metric: {metric_name}\n"
            f"- Best score: {best_score:.6f}\n"
            f"- Completed trials: {n_trials_completed}\n"
            f"- Wall time seconds: {wall_time_seconds:.3f}\n\n"
            "## Top Trials\n\n"
            f"{top}\n"
        )

    def asdict(self) -> dict:
        return asdict(self)


def _markdown_table(df: pd.DataFrame) -> str:
    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)

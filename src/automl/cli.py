"""``automl`` CLI entry point.

Commands::

    automl fit         --config CFG --data CSV [--target T] [--out OUT]
    automl score       --run RUN_DIR --data CSV [--target T]
    automl predict     --run RUN_DIR --data CSV --out PRED.csv
    automl leaderboard --run RUN_DIR [--top N]
    automl runs        [--out-dir OUT]

All commands are thin wrappers around :mod:`automl.api` and :mod:`automl.runs.store`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from automl.api import AutoMLClassifier, AutoMLRegressor
from automl.config.loader import load_config_from_path
from automl.runs.store import list_runs, load_pipeline_from_run, load_run

app = typer.Typer(add_completion=False, help="AutoML Pipeline Framework CLI")


def _read_data(path: Path, target: str | None) -> tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(path)
    if target is None:
        return df, None
    if target not in df.columns:
        raise typer.BadParameter(f"target column '{target}' not in {list(df.columns)}")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


@app.command()
def fit(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="Override target column name."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Override out_dir."),
) -> None:
    """Fit AutoML on ``data`` using ``config`` and write a run directory."""
    cfg = load_config_from_path(config)
    target_col = target or cfg.target
    X, y = _read_data(data, target_col)
    if y is None:  # pragma: no cover - defensive
        raise typer.BadParameter("target is required for fit")

    cls = AutoMLClassifier if cfg.task == "classification" else AutoMLRegressor
    auto = cls(config=cfg, target=target_col, out_dir=out)
    auto.fit(X, y)
    typer.echo(f"best {auto.metric_name_}: {auto.best_score_:.6f}")
    typer.echo(f"run_dir: {auto.run_dir_}")


@app.command()
def score(
    run: Path = typer.Option(..., "--run", "-r", exists=True, file_okay=False),
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True),
    target: str = typer.Option(..., "--target", "-t"),
) -> None:
    """Score a saved run against a labeled dataset."""
    pipeline = load_pipeline_from_run(run)
    X, y = _read_data(data, target)
    s = pipeline.score(X, y)
    typer.echo(f"score: {s:.6f}")


@app.command()
def predict(
    run: Path = typer.Option(..., "--run", "-r", exists=True, file_okay=False),
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True),
    out: Path = typer.Option(..., "--out", "-o"),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Drop this column from data before predicting."
    ),
) -> None:
    """Predict with a saved run and write a CSV with a 'prediction' column."""
    pipeline = load_pipeline_from_run(run)
    X, _ = _read_data(data, target)
    preds = pipeline.predict(X)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": preds}).to_csv(out, index=False)
    typer.echo(f"wrote {len(preds)} predictions \u2192 {out}")


@app.command()
def leaderboard(
    run: Path = typer.Option(..., "--run", "-r", exists=True, file_okay=False),
    top: int = typer.Option(10, "--top", "-n", min=1),
) -> None:
    """Print the top-N rows of a run's leaderboard."""
    info = load_run(run)
    lb: pd.DataFrame = info["leaderboard"]
    typer.echo(lb.head(top).to_string(index=False))


@app.command()
def runs(
    out_dir: Path = typer.Option(Path("runs"), "--out-dir", "-o"),
) -> None:
    """List all runs under ``out_dir`` (newest first)."""
    handles = list_runs(out_dir)
    if not handles:
        typer.echo(f"(no runs found under {out_dir})")
        return
    rows = [
        {
            "run_id": h.run_id,
            "task": h.task,
            "metric": h.metric,
            "score": f"{h.score:.6f}",
            "n_rows": h.n_rows,
            "n_cols": h.n_cols,
        }
        for h in handles
    ]
    typer.echo(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    app()

"""Pick top-k pipelines from a study leaderboard, optionally with diversity.

The selector consumes the leaderboard produced by ``run_search`` and the
matching ``optuna.Study`` (so we can recover each trial's full set of sampled
parameters and rebuild its pipeline). Returns a list of ``SelectedPipeline``
specs that the ensembler turns into fitted base estimators.

Diversity strategies (``config.ensembling.diversity``):

* ``"none"``           — strict top-k by score.
* ``"random"``         — top-k by score after shuffling within score bins.
* ``"metric_pareto"``  — keep only one pipeline per (model_name) so the
  ensemble is at least somewhat heterogeneous, falling back to the next-best
  trial of an unseen model when ties occur.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from automl.config.schema import DiversityStrategy

__all__ = ["SelectedPipeline", "select_top_k"]


@dataclass(frozen=True)
class SelectedPipeline:
    trial_number: int
    score: float
    model_name: str
    imputation: str
    encoding: str
    scaling: str
    feature_selection: str
    params: dict[str, Any]


def _params_for_trial(study: optuna.Study, trial_number: int) -> dict[str, Any]:
    for t in study.trials:
        if t.number == trial_number:
            return dict(t.params)
    raise KeyError(f"Trial {trial_number} not found in study.")


def _row_to_spec(row: pd.Series, study: optuna.Study) -> SelectedPipeline:
    return SelectedPipeline(
        trial_number=int(row["trial_number"]),
        score=float(row["score"]),
        model_name=str(row["model"]),
        imputation=str(row["imputation"]),
        encoding=str(row["encoding"]),
        scaling=str(row["scaling"]),
        feature_selection=str(row["feature_selection"]),
        params=_params_for_trial(study, int(row["trial_number"])),
    )


def select_top_k(
    leaderboard: pd.DataFrame,
    study: optuna.Study,
    *,
    k: int,
    strategy: DiversityStrategy = "none",
    seed: int = 0,
) -> list[SelectedPipeline]:
    if k < 1:
        raise ValueError(f"k must be >= 1 (got {k}).")
    if leaderboard.empty:
        return []

    df = leaderboard.copy()

    if strategy == "metric_pareto":
        # One pipeline per model_name; leaderboard is already best-first.
        df = df.drop_duplicates(subset=["model"], keep="first")
    elif strategy == "random":
        rng = random.Random(seed)
        # Shuffle within rows that share the same score, then take top-k by score.
        groups = []
        for _, grp in df.groupby("score", sort=False):
            shuffled = grp.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1))
            groups.append(shuffled)
        df = pd.concat(groups, ignore_index=True) if groups else df
    elif strategy == "none":
        pass
    else:  # pragma: no cover - schema validation prevents this
        raise ValueError(f"Unknown diversity strategy: {strategy!r}")

    selected = df.head(k)
    return [_row_to_spec(row, study) for _, row in selected.iterrows()]

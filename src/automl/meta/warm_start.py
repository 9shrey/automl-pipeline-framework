"""Warm-start helper: turn nearest-neighbor records into Optuna trials."""

from __future__ import annotations

from typing import Any

import optuna

from automl.meta.store import MetaRecord

__all__ = ["warm_start_study"]

_PRIMITIVE = (type(None), bool, int, float, str)


def _filter_primitives(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if isinstance(v, _PRIMITIVE)}


def warm_start_study(
    study: optuna.Study,
    records: list[MetaRecord],
    *,
    max_trials: int | None = None,
) -> int:
    """Enqueue ``records`` as fixed warm-start trials. Returns count enqueued."""
    n = 0
    for rec in records[: (max_trials or len(records))]:
        params = _filter_primitives(rec.params)
        if not params:
            continue
        try:
            study.enqueue_trial(params, skip_if_exists=True)
            n += 1
        except (ValueError, RuntimeError):
            continue
    return n


"""SQLite-backed store of past runs keyed by dataset meta-features."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from automl.meta.features import MetaFeatures, meta_feature_distance

__all__ = ["MetaRecord", "MetaStore"]


@dataclass(frozen=True)
class MetaRecord:
    id: int
    task: str
    metric: str
    score: float
    meta: MetaFeatures
    params: dict[str, Any]


class MetaStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                task TEXT NOT NULL,
                metric TEXT NOT NULL,
                score REAL NOT NULL,
                meta_features_json TEXT NOT NULL,
                params_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MetaStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def record(
        self,
        *,
        task: str,
        metric: str,
        score: float,
        meta: MetaFeatures,
        params: dict[str, Any],
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO runs (created_at, task, metric, score, meta_features_json, params_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                task,
                metric,
                float(score),
                json.dumps(meta.as_dict()),
                json.dumps(_jsonable_params(params)),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def all(self, *, task: str | None = None) -> list[MetaRecord]:
        sql = "SELECT id, task, metric, score, meta_features_json, params_json FROM runs"
        args: tuple[Any, ...] = ()
        if task is not None:
            sql += " WHERE task = ?"
            args = (task,)
        rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_record(r) for r in rows]

    def nearest(
        self,
        meta: MetaFeatures,
        *,
        task: str,
        k: int = 5,
    ) -> list[MetaRecord]:
        records = self.all(task=task)
        records.sort(key=lambda r: meta_feature_distance(meta, r.meta))
        return records[:k]


def _row_to_record(row: tuple[Any, ...]) -> MetaRecord:
    rid, task, metric, score, mf_json, params_json = row
    meta = MetaFeatures(**json.loads(mf_json))
    return MetaRecord(
        id=int(rid),
        task=str(task),
        metric=str(metric),
        score=float(score),
        meta=meta,
        params=json.loads(params_json),
    )


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)
    return out


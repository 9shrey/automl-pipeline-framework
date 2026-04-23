"""Public sklearn-compatible API: ``AutoMLClassifier`` / ``AutoMLRegressor``.

Wires together: config → schema inference → search engine → refit best
pipeline on the full training set → run recorder. Ensembling, meta warm-start,
and explainers are deferred to later batches; the orchestrator exposes hooks
for them but ships with a clean single-best-pipeline path today.

Usage:

    >>> from automl.api import AutoMLClassifier
    >>> auto = AutoMLClassifier(time_budget_seconds=30, n_trials=10)
    >>> auto.fit(X_train, y_train)
    >>> auto.predict(X_test)
    >>> auto.leaderboard_   # pandas.DataFrame
    >>> auto.run_dir_       # pathlib.Path
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from automl.config.loader import check_runtime_deps, load_config, load_config_from_path
from automl.config.schema import AutoMLConfig
from automl.data.schema_infer import infer_schema
from automl.runs.recorder import RunArtifacts, RunRecorder
from automl.search.engine import run_search

__all__ = ["AutoMLClassifier", "AutoMLRegressor", "load_pipeline"]


def _as_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {arr.shape}.")
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])


def _resolve_config(
    *,
    task: str,
    config: AutoMLConfig | str | Path | dict | None,
    overrides: dict[str, Any],
) -> AutoMLConfig:
    if isinstance(config, AutoMLConfig):
        cfg = config
    elif isinstance(config, (str, Path)):
        cfg = load_config_from_path(config)
    elif isinstance(config, dict):
        cfg = load_config(config)
    elif config is None:
        # Minimum viable defaults; caller MUST pass target.
        target = overrides.pop("target", "y")
        base: dict[str, Any] = {"task": task, "target": target}
        cfg = load_config(base)
    else:
        raise TypeError(f"Unsupported config type: {type(config).__name__}")

    # Apply scalar overrides (time_budget_seconds, n_trials, seed, target …).
    if overrides:
        merged = cfg.model_dump()
        for k, v in overrides.items():
            if k not in merged:
                raise ValueError(f"Unknown override key: {k}")
            merged[k] = v
        cfg = load_config(merged)

    if cfg.task != task:
        raise ValueError(
            f"Config task='{cfg.task}' does not match estimator task='{task}'. "
            "Use the matching AutoMLClassifier / AutoMLRegressor."
        )
    return cfg


class _AutoMLBase(BaseEstimator):
    """Shared fit/predict/save/load logic for classifier + regressor facades."""

    _task: str = ""  # filled by subclasses

    def __init__(
        self,
        config: AutoMLConfig | str | Path | dict | None = None,
        *,
        target: str = "y",
        time_budget_seconds: int | None = None,
        n_trials: int | None = None,
        seed: int | None = None,
        metric: str | None = None,
        out_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.target = target
        self.time_budget_seconds = time_budget_seconds
        self.n_trials = n_trials
        self.seed = seed
        self.metric = metric
        self.out_dir = out_dir

    # ------------------------------------------------------------------
    # Internal: build the resolved config from constructor args.
    # ------------------------------------------------------------------
    def _resolved_config(self) -> AutoMLConfig:
        overrides: dict[str, Any] = {"target": self.target}
        for k in ("time_budget_seconds", "n_trials", "seed"):
            v = getattr(self, k)
            if v is not None:
                overrides[k] = v
        return _resolve_config(task=self._task, config=self.config, overrides=overrides)

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "_AutoMLBase":
        cfg = self._resolved_config()
        check_runtime_deps(cfg)

        X_df = _as_dataframe(X)
        y_arr = np.asarray(y)
        schema = infer_schema(X_df)

        # Optional warm-start from the meta store. We pull nearest neighbors
        # *before* the search; recording back happens after.
        warm_params: list[dict[str, Any]] = []
        meta = None
        if cfg.meta_store.enabled and cfg.search.warm_start.enabled:
            from automl.meta import MetaStore, compute_meta_features

            meta = compute_meta_features(X_df, y_arr, task=cfg.task, schema=schema)
            try:
                with MetaStore(cfg.meta_store.path) as store:
                    nbrs = store.nearest(
                        meta, task=cfg.task, k=cfg.search.warm_start.top_k_neighbors
                    )
                warm_params = [r.params for r in nbrs]
            except Exception:  # noqa: BLE001 - meta store is best-effort
                warm_params = []

        t0 = time.perf_counter()
        result = run_search(
            cfg, X_df, y_arr,
            schema=schema, metric=self.metric,
            warm_start_params=warm_params or None,
        )
        wall = time.perf_counter() - t0

        # Refit best pipeline on full training data.
        from automl.search.objective import (
            SampledPipelineSpec,
            build_pipeline_from_params,
        )

        best_params = dict(result.best_trial.params)
        attrs = result.best_trial.user_attrs
        spec = SampledPipelineSpec(
            imputation=attrs["imputation"],
            encoding=attrs["encoding"],
            scaling=attrs["scaling"],
            feature_selection=attrs["feature_selection"],
            model_name=attrs["model_name"],
            model_params=_extract_model_params(
                best_params, attrs["model_name"], cfg.task
            ),
        )
        pipeline = build_pipeline_from_params(spec, schema=schema, task=cfg.task)
        pipeline.fit(X_df, y_arr)

        # Optional ensembling.
        ensemble_pipeline: Any | None = None
        if cfg.ensembling.enabled and cfg.ensembling.strategy != "none":
            from automl.ensembling import build_ensemble

            ensemble_pipeline = build_ensemble(
                config=cfg.ensembling,
                task=cfg.task,
                leaderboard=result.leaderboard,
                study=result.study,
                schema=schema,
                X=X_df,
                y=y_arr,
                seed=cfg.seed,
            )
        deployable = ensemble_pipeline if ensemble_pipeline is not None else pipeline

        # Recorder.
        recorder_root = Path(self.out_dir) if self.out_dir is not None else None
        recorder = RunRecorder(cfg, root=recorder_root)
        signed = result.study.best_value
        from automl.evaluation.metrics import get_metric

        m = get_metric(result.metric_name)
        best_score = m.signed * float(signed)
        artifacts = recorder.finalize(
            X=X_df,
            y=y_arr,
            schema=schema,
            leaderboard=result.leaderboard,
            best_pipeline=deployable,
            best_trial=result.best_trial,
            metric_name=result.metric_name,
            best_score=best_score,
            n_trials_completed=int(len(result.leaderboard)),
            wall_time_seconds=wall,
        )

        # Optional explainability report (best-effort).
        explain_paths: dict[str, Path] = {}
        if cfg.explain.enabled:
            try:
                from automl.explain import explain_pipeline

                explain_paths = explain_pipeline(
                    estimator=deployable,
                    X=X_df,
                    y=y_arr,
                    config=cfg.explain,
                    out_dir=artifacts.run_dir,
                    seed=cfg.seed,
                )
            except Exception:  # noqa: BLE001
                explain_paths = {}

        # Record back to the meta-store after success.
        if meta is not None:
            try:
                from automl.meta import MetaStore

                with MetaStore(cfg.meta_store.path) as store:
                    store.record(
                        task=cfg.task,
                        metric=result.metric_name,
                        score=best_score,
                        meta=meta,
                        params=best_params,
                    )
            except Exception:  # noqa: BLE001
                pass

        # Public fitted attributes.
        self.config_ = cfg
        self.schema_ = schema
        self.pipeline_ = deployable
        self.best_pipeline_ = pipeline
        self.ensemble_ = ensemble_pipeline
        self.leaderboard_ = result.leaderboard
        self.best_score_ = best_score
        self.metric_name_ = result.metric_name
        self.run_dir_ = artifacts.run_dir
        self.artifacts_ = artifacts
        self.explain_paths_ = explain_paths
        self.meta_features_ = meta
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        if self._task == "classification":
            self.classes_ = np.unique(y_arr)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        X_df = _coerce_for_predict(X, self.feature_names_in_)
        return self.pipeline_.predict(X_df)

    def score(self, X: Any, y: Any) -> float:
        check_is_fitted(self, "pipeline_")
        X_df = _coerce_for_predict(X, self.feature_names_in_)
        return float(self.pipeline_.score(X_df, np.asarray(y)))

    def save(self, path: str | Path) -> Path:
        check_is_fitted(self, "pipeline_")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self.pipeline_, f)
        return p


class AutoMLClassifier(_AutoMLBase, ClassifierMixin):
    """sklearn-compatible AutoML classifier."""

    _task = "classification"

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        if not hasattr(self.pipeline_, "predict_proba"):
            raise AttributeError("Best pipeline does not implement predict_proba.")
        X_df = _coerce_for_predict(X, self.feature_names_in_)
        return self.pipeline_.predict_proba(X_df)


class AutoMLRegressor(_AutoMLBase, RegressorMixin):
    """sklearn-compatible AutoML regressor."""

    _task = "regression"


def load_pipeline(path: str | Path) -> Any:
    """Load a pickled pipeline saved by ``AutoMLClassifier.save`` / recorder."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_for_predict(X: Any, feature_names: np.ndarray) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {arr.shape}.")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"X has {arr.shape[1]} columns; estimator was fit on {len(feature_names)}."
        )
    return pd.DataFrame(arr, columns=list(feature_names))


_TOP_LEVEL_PARAMS = {
    "imputation",
    "encoding",
    "scaling",
    "feature_selection",
    "model",
}


def _extract_model_params(
    params: dict[str, Any], model_name: str, task: str
) -> dict[str, Any]:
    """Reverse of how ``Objective`` samples model params.

    The model adapter declares its own search space (parameter names), so we
    forward everything that is *not* one of the top-level orchestration knobs.
    The OptunaSampler may also have stored ``"<name>__idx"`` proxies for
    non-primitive categorical choices; we re-resolve those via the model's
    declared search space.
    """
    from automl.models.registry import EstimatorRegistry

    adapter = EstimatorRegistry.get(model_name)
    space = adapter.search_space(task)
    declared = set(space.parameter_names())
    # Build name → node lookup for non-primitive categorical resolution.
    nodes_by_name: dict[str, Any] = {}
    for node in space.nodes:
        inner = getattr(node, "node", node)  # unwrap Conditional
        nodes_by_name[inner.name] = inner

    out: dict[str, Any] = {}
    for k, v in params.items():
        if k in _TOP_LEVEL_PARAMS:
            continue
        if k.endswith("__idx"):
            base = k[: -len("__idx")]
            node = nodes_by_name.get(base)
            choices = getattr(node, "choices", None)
            if choices is not None:
                try:
                    out[base] = choices[int(v)]
                    continue
                except (IndexError, ValueError, TypeError):
                    pass
        if k in declared:
            out[k] = v
    return out

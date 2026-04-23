"""Optuna ``Objective`` callable: sample → build pipeline → CV evaluate → prune.

For each trial:
  1. Sample preprocessing (imputation/encoding/scaling), feature selection, and
     model name from the configured search space lists.
  2. Look up the model adapter and sample its model-specific space.
  3. Build a sklearn ``Pipeline`` (preprocessor → selector → estimator).
  4. CV-evaluate; after each fold report the running mean to the pruner so ASHA
     / Hyperband can kill hopeless trials early.
  5. Return the metric value, sign-flipped if the metric is minimize-direction
     (so the study is always maximization).

Leakage-safety: the preprocessor + selector are *part of* the per-fold pipeline,
so they fit only on the training rows of each fold. Target encoding is
deliberately not implemented yet (would need a per-fold-aware adapter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from automl.config.schema import AutoMLConfig
from automl.data.schema_infer import ColumnSchema
from automl.evaluation.cv import build_splitter
from automl.evaluation.metrics import DEFAULT_METRIC, Metric, get_metric
from automl.evaluation.scoring import score_estimator
from automl.feature_selection.factory import build_feature_selector
from automl.models.registry import EstimatorRegistry
from automl.preprocessing.factory import build_preprocessor
from automl.search.samplers import OptunaSampler

__all__ = ["Objective", "build_pipeline_from_params"]


def _filter_models_by_task(models: list[str], task: str) -> list[str]:
    keep: list[str] = []
    for name in models:
        try:
            adapter = EstimatorRegistry.get(name)
        except KeyError:
            continue
        if adapter.supports(task):
            keep.append(name)
    if not keep:
        raise ValueError(
            f"No registered models in {models} support task='{task}'."
        )
    return keep


def _filter_encoders_for_search(encoders: list[str]) -> list[str]:
    # Target encoding is not yet implemented; silently drop it from the search
    # space so old configs still work.
    return [e for e in encoders if e != "target"] or ["onehot"]


@dataclass
class SampledPipelineSpec:
    imputation: str
    encoding: str
    scaling: str
    feature_selection: str
    model_name: str
    model_params: dict[str, Any]


def build_pipeline_from_params(
    spec: SampledPipelineSpec,
    *,
    schema: ColumnSchema,
    task: str,
) -> Pipeline:
    pre = build_preprocessor(
        schema,
        imputation=spec.imputation,
        encoding=spec.encoding,
        scaling=spec.scaling,
    )
    sel = build_feature_selector(spec.feature_selection, task=task)
    adapter = EstimatorRegistry.get(spec.model_name)
    estimator: BaseEstimator = adapter.build(task, spec.model_params)
    return Pipeline(
        steps=[("preprocess", pre), ("select", sel), ("estimator", estimator)]
    )


class Objective:
    """Callable Optuna objective."""

    def __init__(
        self,
        config: AutoMLConfig,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        schema: ColumnSchema,
        metric: str | None = None,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Objective expects X as pandas DataFrame.")
        self.config = config
        self.X = X
        self.y = np.asarray(y)
        self.schema = schema
        metric_name = metric or DEFAULT_METRIC[config.task]
        self.metric: Metric = get_metric(metric_name)
        if self.metric.task != config.task:
            raise ValueError(
                f"Metric '{self.metric.name}' is for task '{self.metric.task}', "
                f"but config.task='{config.task}'."
            )
        self._models = _filter_models_by_task(config.search.space.models, config.task)
        self._encoders = _filter_encoders_for_search(config.search.space.encoding)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_spec(self, sampler: OptunaSampler) -> SampledPipelineSpec:
        imputation = sampler.suggest_categorical(
            "imputation", self.config.search.space.imputation
        )
        encoding = sampler.suggest_categorical("encoding", self._encoders)
        scaling = sampler.suggest_categorical(
            "scaling", self.config.search.space.scaling
        )
        feature_selection = sampler.suggest_categorical(
            "feature_selection", self.config.search.space.feature_selection
        )
        model_name = sampler.suggest_categorical("model", self._models)
        adapter = EstimatorRegistry.get(model_name)
        model_space = adapter.search_space(self.config.task)
        model_params = model_space.sample(sampler)
        return SampledPipelineSpec(
            imputation=imputation,
            encoding=encoding,
            scaling=scaling,
            feature_selection=feature_selection,
            model_name=model_name,
            model_params=model_params,
        )

    # ------------------------------------------------------------------
    # Trial entry
    # ------------------------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        sampler = OptunaSampler(trial)
        spec = self._sample_spec(sampler)
        pipeline = build_pipeline_from_params(spec, schema=self.schema, task=self.config.task)

        splitter = build_splitter(self.config.cv, self.config.task, seed=self.config.seed)
        fold_scores: list[float] = []
        try:
            for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(self.X, self.y)):
                X_tr, X_va = self.X.iloc[tr_idx], self.X.iloc[va_idx]
                y_tr, y_va = self.y[tr_idx], self.y[va_idx]
                model = clone(pipeline)
                model.fit(X_tr, y_tr)
                score = score_estimator(model, X_va, y_va, self.metric)
                fold_scores.append(score)
                # Report (signed so we always maximize) for the pruner.
                trial.report(self.metric.signed * float(np.mean(fold_scores)), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            # Failure isolation: log the failure as a trial failure.
            trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
            raise

        # Stash useful attrs for the leaderboard / recorder.
        trial.set_user_attr("model_name", spec.model_name)
        trial.set_user_attr("imputation", spec.imputation)
        trial.set_user_attr("encoding", spec.encoding)
        trial.set_user_attr("scaling", spec.scaling)
        trial.set_user_attr("feature_selection", spec.feature_selection)
        trial.set_user_attr("fold_scores", fold_scores)
        return self.metric.signed * float(np.mean(fold_scores))

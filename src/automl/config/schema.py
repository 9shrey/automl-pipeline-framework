"""Pydantic v2 config models for the AutoML framework.

The ``AutoMLConfig`` model mirrors ``configs/default.yaml`` and is the single
validated representation of run knobs. Validation runs at config-load time so
invalid combinations (unknown sampler, missing optional dependency, etc.) fail
loudly before the search starts.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AutoMLConfig",
    "CVConfig",
    "EnsemblingConfig",
    "ExplainConfig",
    "MetaStoreConfig",
    "MLflowConfig",
    "RunConfig",
    "SearchConfig",
    "SearchSpaceConfig",
    "Task",
    "WarmStartConfig",
]

Task = Literal["classification", "regression"]
SamplerName = Literal["tpe", "cmaes", "random"]
PrunerName = Literal["asha", "hyperband", "median", "none"]
CVScheme = Literal["kfold", "stratified_kfold", "group_kfold"]
EnsembleStrategy = Literal["stacking", "voting", "blending", "none"]
DiversityStrategy = Literal["metric_pareto", "random", "none"]
MetaLearner = Literal["logreg", "ridge", "lightgbm"]


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class CVConfig(_StrictBase):
    scheme: CVScheme = "stratified_kfold"
    n_splits: Annotated[int, Field(ge=2, le=20)] = 5
    shuffle: bool = True


class WarmStartConfig(_StrictBase):
    enabled: bool = True
    top_k_neighbors: Annotated[int, Field(ge=1, le=100)] = 5
    min_overlap_meta_features: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6


class SearchSpaceConfig(_StrictBase):
    imputation: list[str] = Field(
        default_factory=lambda: ["mean", "median", "most_frequent", "iterative", "knn"]
    )
    encoding: list[str] = Field(
        default_factory=lambda: ["onehot", "ordinal", "target", "frequency"]
    )
    scaling: list[str] = Field(
        default_factory=lambda: ["standard", "robust", "minmax", "quantile", "none"]
    )
    feature_selection: list[str] = Field(
        default_factory=lambda: ["none", "variance", "mutual_info", "rfe", "shap"]
    )
    models: list[str] = Field(
        default_factory=lambda: [
            "xgboost", "lightgbm", "catboost", "histgb", "logreg", "rf", "extratrees",
        ]
    )

    @model_validator(mode="after")
    def _all_non_empty(self) -> SearchSpaceConfig:
        empty = [k for k, v in self.model_dump().items() if not v]
        if empty:
            raise ValueError(f"Search-space lists must be non-empty: {empty}")
        return self


class SearchConfig(_StrictBase):
    sampler: SamplerName = "tpe"
    pruner: PrunerName = "asha"
    warm_start: WarmStartConfig = Field(default_factory=WarmStartConfig)
    space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)


class EnsemblingConfig(_StrictBase):
    enabled: bool = True
    strategy: EnsembleStrategy = "stacking"
    top_k: Annotated[int, Field(ge=1, le=50)] = 5
    diversity: DiversityStrategy = "metric_pareto"
    meta_learner: MetaLearner = "logreg"


class ExplainConfig(_StrictBase):
    enabled: bool = True
    shap_max_samples: Annotated[int, Field(ge=10, le=100_000)] = 1000
    permutation_repeats: Annotated[int, Field(ge=1, le=200)] = 10


class MLflowConfig(_StrictBase):
    enabled: bool = False
    tracking_uri: str = "file:./mlruns"


class RunConfig(_StrictBase):
    out_dir: str = "runs"
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


class MetaStoreConfig(_StrictBase):
    path: str = ".automl_meta.sqlite"
    enabled: bool = True


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class AutoMLConfig(_StrictBase):
    """Top-level AutoML configuration."""

    task: Task
    target: str
    time_budget_seconds: Annotated[int, Field(ge=1)] = 1800
    n_trials: Annotated[int, Field(ge=1)] = 200
    seed: Annotated[int, Field(ge=0)] = 42
    n_jobs: int = -1

    cv: CVConfig = Field(default_factory=CVConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    ensembling: EnsemblingConfig = Field(default_factory=EnsemblingConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    meta_store: MetaStoreConfig = Field(default_factory=MetaStoreConfig)

    @model_validator(mode="after")
    def _cv_scheme_matches_task(self) -> AutoMLConfig:
        if self.task == "regression" and self.cv.scheme == "stratified_kfold":
            raise ValueError(
                "cv.scheme='stratified_kfold' is not valid for task='regression'; "
                "use 'kfold' or 'group_kfold'."
            )
        return self

    @model_validator(mode="after")
    def _n_jobs_sane(self) -> AutoMLConfig:
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError(f"n_jobs must be -1 or a positive int (got {self.n_jobs}).")
        return self

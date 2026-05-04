"""Explainability subpackage: SHAP + permutation + report rendering.

The top-level ``explain_pipeline`` function is the orchestrator entry point —
it runs whichever explainers are available + enabled and writes a tidy
``explain.md`` plus CSVs into the run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from automl.config.schema import ExplainConfig
from automl.explain.permutation import permutation_feature_importance
from automl.explain.reports import write_importance_report
from automl.explain.shap_explainer import shap_available, shap_feature_importance

__all__ = [
    "explain_pipeline",
    "permutation_feature_importance",
    "shap_available",
    "shap_feature_importance",
    "write_importance_report",
]


def explain_pipeline(
    *,
    estimator: Any,
    X: pd.DataFrame,
    y: Any,
    config: ExplainConfig,
    out_dir: Path,
    seed: int = 0,
) -> dict[str, Path]:
    """Run enabled explainers and write a report. Returns artifacts written.

    Never raises on individual explainer failure (best-effort, since
    explainability should not break a successful run).
    """
    if not config.enabled:
        return {}

    perm_df: pd.DataFrame | None = None
    shap_df: pd.DataFrame | None = None

    try:
        perm_df = permutation_feature_importance(
            estimator, X, y,
            n_repeats=config.permutation_repeats,
            seed=seed,
        )
    except Exception:
        perm_df = None

    if shap_available():
        try:
            shap_df = shap_feature_importance(
                estimator, X,
                max_samples=config.shap_max_samples,
                seed=seed,
            )
        except Exception:
            shap_df = None

    return write_importance_report(out_dir=out_dir, permutation=perm_df, shap=shap_df)


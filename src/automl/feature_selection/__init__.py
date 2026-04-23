"""Feature selection zoo: rfe / shap / mutual_info / variance / model-based + factory."""

from automl.feature_selection.factory import FEATURE_SELECTORS, build_feature_selector
from automl.feature_selection.shap_select import ShapSelector

__all__ = ["FEATURE_SELECTORS", "ShapSelector", "build_feature_selector"]

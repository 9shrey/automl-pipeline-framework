"""Unit tests for ``automl.evaluation``."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from automl.config.schema import CVConfig
from automl.evaluation import (
    DEFAULT_METRIC,
    Metric,
    available_metrics,
    build_splitter,
    get_metric,
    score_estimator,
    score_predictions,
    split_indices,
)


# ---------------------------------------------------------------------------
# CV splitters
# ---------------------------------------------------------------------------


class TestBuildSplitter:
    def test_stratified_kfold_for_classification(self) -> None:
        sp = build_splitter(CVConfig(n_splits=3), task="classification", seed=0)
        assert isinstance(sp, StratifiedKFold)
        assert sp.n_splits == 3

    def test_kfold_for_regression(self) -> None:
        sp = build_splitter(
            CVConfig(scheme="kfold", n_splits=4), task="regression", seed=0
        )
        assert isinstance(sp, KFold)
        assert sp.n_splits == 4

    def test_group_kfold(self) -> None:
        sp = build_splitter(
            CVConfig(scheme="group_kfold", n_splits=3), task="classification", seed=0
        )
        assert isinstance(sp, GroupKFold)

    def test_stratified_kfold_rejects_regression(self) -> None:
        with pytest.raises(ValueError, match="stratified_kfold"):
            build_splitter(CVConfig(), task="regression", seed=0)

    def test_split_indices_deterministic_with_seed(self) -> None:
        X, y = make_classification(n_samples=60, n_features=4, random_state=0)
        sp1 = build_splitter(CVConfig(n_splits=3), task="classification", seed=42)
        sp2 = build_splitter(CVConfig(n_splits=3), task="classification", seed=42)
        s1 = split_indices(sp1, X, y)
        s2 = split_indices(sp2, X, y)
        assert len(s1) == len(s2) == 3
        for (tr1, va1), (tr2, va2) in zip(s1, s2, strict=True):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(va1, va2)

    def test_no_shuffle_means_no_random_state(self) -> None:
        sp = build_splitter(
            CVConfig(n_splits=3, shuffle=False), task="classification", seed=42
        )
        assert sp.random_state is None


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


class TestMetricRegistry:
    def test_default_metrics(self) -> None:
        assert DEFAULT_METRIC["classification"] == "roc_auc"
        assert DEFAULT_METRIC["regression"] == "rmse"

    def test_available_metrics_filter_by_task(self) -> None:
        clf = available_metrics("classification")
        reg = available_metrics("regression")
        assert "roc_auc" in clf and "rmse" not in clf
        assert "rmse" in reg and "roc_auc" not in reg

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown metric"):
            get_metric("not_a_metric")

    def test_signed_direction(self) -> None:
        assert get_metric("roc_auc").signed == 1
        assert get_metric("rmse").signed == -1

    def test_metric_is_callable_via_score(self) -> None:
        m = get_metric("accuracy")
        assert isinstance(m, Metric)
        assert m.score([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_score_classifier_with_proba_metric(self) -> None:
        X, y = make_classification(n_samples=200, n_features=6, random_state=0)
        clf = LogisticRegression(max_iter=500).fit(X, y)
        auc = score_estimator(clf, X, y, "roc_auc")
        assert 0.7 <= auc <= 1.0

    def test_score_classifier_with_label_metric(self) -> None:
        X, y = make_classification(n_samples=200, n_features=6, random_state=0)
        clf = LogisticRegression(max_iter=500).fit(X, y)
        acc = score_estimator(clf, X, y, "accuracy")
        assert acc > 0.7

    def test_score_regressor(self) -> None:
        X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=0)
        reg = LinearRegression().fit(X, y)
        rmse = score_estimator(reg, X, y, "rmse")
        r2 = score_estimator(reg, X, y, "r2")
        assert rmse >= 0.0
        assert r2 > 0.9

    def test_proba_metric_on_estimator_without_proba_raises(self) -> None:
        X, y = make_regression(n_samples=50, n_features=3, random_state=0)
        reg = LinearRegression().fit(X, y)
        # Force a proba metric on a regressor (deliberately wrong) → AttributeError.
        with pytest.raises(AttributeError, match="predict_proba"):
            score_estimator(reg, X, y, "roc_auc")

    def test_score_predictions_passthrough(self) -> None:
        s = score_predictions([0, 1, 1, 0], [0, 1, 1, 1], "accuracy")
        assert s == pytest.approx(0.75)

    def test_score_accepts_metric_object(self) -> None:
        X, y = make_classification(n_samples=100, n_features=5, random_state=0)
        clf = LogisticRegression(max_iter=500).fit(X, y)
        m = get_metric("accuracy")
        assert score_estimator(clf, X, y, m) > 0.5

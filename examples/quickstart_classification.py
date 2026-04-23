"""Quickstart: fit AutoMLClassifier on a tiny synthetic dataset.

Run with:
    PYTHONPATH=src python examples/quickstart_classification.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from automl.api import AutoMLClassifier


def main() -> None:
    X_arr, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        random_state=0,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, stratify=y)

    config = {
        "task": "classification",
        "target": "y",
        "time_budget_seconds": 60,
        "n_trials": 10,
        "seed": 42,
        "n_jobs": 1,
        "cv": {"scheme": "stratified_kfold", "n_splits": 3, "shuffle": True},
        "search": {
            "sampler": "tpe",
            "pruner": "asha",
            "warm_start": {"enabled": False},
            "space": {
                "imputation": ["mean", "median"],
                "encoding": ["onehot"],
                "scaling": ["standard", "none"],
                "feature_selection": ["none", "variance"],
                "models": ["logreg", "rf", "histgb"],
            },
        },
        "ensembling": {"enabled": False, "strategy": "none"},
        "explain": {"enabled": False},
        "run": {"out_dir": "runs"},
        "meta_store": {"enabled": False},
    }

    auto = AutoMLClassifier(config=config)
    auto.fit(X_tr, y_tr)

    print(f"best CV {auto.metric_name_}: {auto.best_score_:.4f}")
    print(f"test accuracy:       {auto.score(X_te, y_te):.4f}")
    print(f"run artifacts:       {auto.run_dir_}")
    print()
    print("leaderboard (top 5):")
    print(auto.leaderboard_.head(5).to_string(index=False))


if __name__ == "__main__":
    main()

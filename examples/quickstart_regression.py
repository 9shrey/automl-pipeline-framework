"""Minimal regression quickstart for the sklearn-compatible AutoML API."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from automl.api import AutoMLRegressor


def main() -> None:
    x_arr, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=6,
        noise=12.0,
        random_state=7,
    )
    x = pd.DataFrame(x_arr, columns=[f"feature_{idx}" for idx in range(x_arr.shape[1])])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7)

    auto = AutoMLRegressor(
        target="target",
        time_budget_seconds=60,
        n_trials=10,
        seed=7,
    )
    auto.fit(x_train, y_train)

    print(f"best CV {auto.metric_name_}: {auto.best_score_:.4f}")
    print(f"test score: {auto.score(x_test, y_test):.4f}")
    print(f"run artifacts: {auto.run_dir_}")
    print(auto.leaderboard_.head())


if __name__ == "__main__":
    main()

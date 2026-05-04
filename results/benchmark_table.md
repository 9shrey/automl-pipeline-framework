# Toy Benchmark Results

## Classification

| rank | pipeline | preprocessing | model | cv_accuracy | test_accuracy | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | AutoML stacked ensemble | median+standard_scale | logreg+random_forest | 0.8939 | 0.8745 | 39.2 |
| 2 | XGBoost default | median | xgboost | 0.8739 | 0.8618 | 18.6 |
| 3 | LightGBM default | median | lightgbm | 0.8613 | 0.8571 | 16.8 |
| 4 | sklearn baseline | median+standard_scale | logistic_regression | 0.8362 | 0.8317 | 4.1 |

## Regression

| rank | pipeline | preprocessing | model | cv_rmse | test_rmse | fit_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | AutoML blended ensemble | median+robust_scale | elasticnet+random_forest | 18.8672 | 19.6591 | 42.7 |
| 2 | LightGBM default | median | lightgbm | 20.129 | 20.2901 | 17.4 |
| 3 | XGBoost default | median | xgboost | 20.2957 | 21.0935 | 20.2 |
| 4 | sklearn baseline | median+standard_scale | ridge | 24.2006 | 25.1914 | 3.9 |

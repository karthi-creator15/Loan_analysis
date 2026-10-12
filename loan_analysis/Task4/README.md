# Week 2 ML Assessment — Task 4

## Run (Windows / VS Code)
```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python task4.py
```

The script creates synthetic loan-customer data, so no external CSV is required.

## Outputs
- `outputs/task4_validation_metrics.csv`
- `outputs/task4_roc_curves.png`
- `outputs/task4_threshold_results.csv`
- `outputs/task4_xgboost_random_search.csv` (if XGBoost is available)
- `outputs/task4_random_search_fallback.csv` (if XGBoost is unavailable)
- `outputs/task4_regression_metrics.csv`
- `outputs/task4_final_test_result.csv`

The assessment requires GridSearchCV on Random Forest (three parameters), RandomizedSearchCV with `n_iter=15` on XGBoost, ROC curves, metrics, a threshold with recall >= 75%, and regression metrics against a mean baseline. XGBoost is included in requirements; if it is not installed, the script safely runs a Random Forest fallback for randomized search.

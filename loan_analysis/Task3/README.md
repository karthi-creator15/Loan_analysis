# Week 2 ML Assessment — Task 3

## Run (Windows / VS Code)
```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python task3.py
```

No external CSV is needed. The project generates a reproducible synthetic loan dataset.
XGBoost is optional to keep the project runnable on environments where its package is unavailable.
To enable it, run `pip install xgboost`.

## Outputs
- `outputs/task3_classifier_results.csv`
- `outputs/task3_random_forest_top5.csv`
- `outputs/task3_xgboost_top5.csv` (if XGBoost is installed and compatible)
- `outputs/task3_regression_coefficients.csv`
- `outputs/task3_kmeans_metrics.csv`
- `outputs/task3_kmeans_elbow.png`
- `outputs/task3_kmeans_silhouette.png`

Note: the dataset is synthetic, so results are educational only.

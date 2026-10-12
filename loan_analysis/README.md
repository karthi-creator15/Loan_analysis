# Week 2 Machine Learning Assessment

Four separate Python projects based on the Week 2 Machine Learning Assessment PDF.

## Folders
- `Task1/` — Splits, overfitting, cross-validation, data leakage, problem types and workflow.
- `Task2/` — Imputation, encoding, feature engineering, pipeline, joblib and predictions.
- `Task3/` — Classification models, OOB score, feature importance, regression and K-Means.
- `Task4/` — Metrics, ROC-AUC, threshold selection, tuning and regression evaluation.

Each task has its own `requirements.txt` and `README.md`. Synthetic datasets are generated automatically; no external CSV is required.

## Run a task (Windows / VS Code)
Open a terminal in the task folder. Example for Task 1:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python task1.py
```
Run the matching script for each task: `python task2.py`, `python task3.py`, or `python task4.py`.
For Task 2, run `python predict_new.py` after `python task2.py`.

## Notes
Datasets are synthetic and intended for learning only. Generated reports, plots and models are stored in each task's `outputs/` directory.

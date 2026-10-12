# Week 2 ML Assessment — Task 1

## Run on Windows / VS Code
```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python task1.py
```

No external CSV is needed. The script uses scikit-learn's built-in Breast Cancer dataset.

## Includes
- Stratified 60/20/20 split (`random_state=42`)
- Decision tree depths 1, 3, 5, 10 and None; training/validation accuracy plot
- Logistic Regression 5-fold cross-validation mean and standard deviation
- Incorrect full-data scaling vs correct train-only scaling demonstration
- Four problem classifications and the seven-step ML workflow

The held-out test set is reserved and not used for model selection. Plot is saved to `outputs/task1_overfitting.png`.

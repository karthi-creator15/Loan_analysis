# Week 2 ML Assessment — Task 2

## Run (Windows / VS Code)
```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python task2.py
python predict_new.py
```

No external CSV is required; a repeatable synthetic loan dataset is generated automatically.

## Files
- `task2.py`: data generation, preprocessing pipeline, CV, model saving and three raw-row predictions.
- `features.py`: reusable feature-engineering transformer needed when loading the saved joblib model.
- `predict_new.py`: loads the saved pipeline and predicts new rows.
- `outputs/`: saved pipeline and predictions CSV.

The pipeline uses median numeric imputation, most-frequent categorical/ordinal imputation, OneHotEncoder, OrdinalEncoder, StandardScaler, and LogisticRegression. A separate KNNImputer numeric-column experiment is also run. Feature engineering is inside the main Pipeline to keep it within cross-validation.

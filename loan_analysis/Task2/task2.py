from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from features import LoanFeatureEngineer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "task2_loan_pipeline.joblib"
RANDOM_STATE = 42

def make_dataset(n=800, random_state=42):
    """Create a repeatable synthetic loan dataset; no external CSV needed."""
    rng = np.random.default_rng(random_state)
    age = rng.integers(21, 70, n).astype(float)
    income = rng.lognormal(mean=10.8, sigma=0.55, size=n)
    loan_amount = rng.uniform(20000, 500000, n)
    credit_score = rng.normal(680, 55, n).clip(400, 850)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n)
    employment_type = rng.choice(["Salaried", "Self-employed", "Contract", "Unemployed"], n,
                                 p=[0.52, 0.24, 0.16, 0.08])
    education_level = rng.choice(["High School", "Bachelor", "Master", "Doctorate"], n,
                                 p=[0.28, 0.43, 0.23, 0.06])
    dates = pd.to_datetime(rng.choice(pd.date_range("2021-01-01", "2025-12-31"), n))
    risk = (0.000004 * loan_amount - 0.0000015 * income
            - 0.006 * (credit_score - 650)
            + 0.45 * (employment_type == "Unemployed")
            + 0.20 * (education_level == "High School")
            + rng.normal(0, 0.65, n))
    probability = 1 / (1 + np.exp(-risk))
    default = rng.binomial(1, probability)
    X = pd.DataFrame({
        "age": age, "income": income, "loan_amount": loan_amount,
        "credit_score": credit_score, "loan_term_months": loan_term_months,
        "employment_type": employment_type, "education_level": education_level,
        "application_date": dates.astype(str)
    })
    y = pd.Series(default, name="default")
    # Add missing values to exercise imputation.
    for column in ["age", "income", "credit_score", "employment_type", "education_level"]:
        mask = rng.random(n) < 0.04
        X.loc[mask, column] = np.nan
    return X, y

def build_pipeline():
    numeric_columns = [
        "age", "income", "loan_amount", "credit_score", "loan_term_months",
        "application_year", "application_month", "income_loan_ratio", "log_income"
    ]
    categorical_columns = ["employment_type"]
    ordinal_columns = ["education_level"]
    education_order = [["High School", "Bachelor", "Master", "Doctorate"]]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=education_order, handle_unknown="use_encoded_value", unknown_value=-1
        ))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
        ("ordinal", ordinal_pipeline, ordinal_columns)
    ], remainder="drop")
    return Pipeline([
        ("feature_engineering", LoanFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

def main():
    print("=" * 68)
    print("sTASK 2: PREPROCESSING, FEATURES AND PIPELINE")
    print("=" * 68)
    X, y = make_dataset()
    print(f"Generated synthetic dataset: {X.shape[0]} rows, {X.shape[1]} raw features.")
    print("No external CSV required.")
    print(f"Missing values before preprocessing: {int(X.isna().sum().sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = build_pipeline()

    # All preprocessing and feature engineering are inside the Pipeline.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print("\n1. PIPELINE CROSS-VALIDATION")
    print("Fold accuracies:", np.round(scores, 4))
    print(f"Pipeline CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    pipeline.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Held-out test accuracy (reported once): {test_accuracy:.4f}")

    # Required alternative: demonstrate KNNImputer on numeric columns.
    # It is a comparison experiment only; the main model uses median imputation.
    print("\n2. KNN IMPUTER NUMERIC-COLUMN COMPARISON")
    engineer = LoanFeatureEngineer()
    engineered_train = engineer.fit_transform(X_train)
    numeric_columns = [
        "age", "income", "loan_amount", "credit_score", "loan_term_months",
        "application_year", "application_month", "income_loan_ratio", "log_income"
    ]
    knn = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])
    numeric_array = knn.fit_transform(engineered_train[numeric_columns])
    print(f"KNNImputer completed on {len(numeric_array)} training rows and "
          f"{numeric_array.shape[1]} numeric features.")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n3. SAVED PIPELINE: {MODEL_PATH}")

    # Required three raw rows, with raw input schema and no manually engineered features.
    new_rows = pd.DataFrame([
        {"age": 29, "income": 72000, "loan_amount": 150000, "credit_score": 710,
         "loan_term_months": 36, "employment_type": "Salaried",
         "education_level": "Bachelor", "application_date": "2026-01-12"},
        {"age": 46, "income": 48000, "loan_amount": 320000, "credit_score": 590,
         "loan_term_months": 60, "employment_type": "Self-employed",
         "education_level": "High School", "application_date": "2026-02-08"},
        {"age": 35, "income": 110000, "loan_amount": 200000, "credit_score": 760,
         "loan_term_months": 24, "employment_type": "Salaried",
         "education_level": "Master", "application_date": "2026-03-18"}
    ])
    loaded_pipeline = joblib.load(MODEL_PATH)
    predictions = loaded_pipeline.predict(new_rows)
    probabilities = loaded_pipeline.predict_proba(new_rows)[:, 1]
    result = new_rows.copy()
    result["predicted_default"] = predictions
    result["default_probability"] = probabilities
    result.to_csv(OUTPUT_DIR / "task2_new_predictions.csv", index=False)
    print("\n4. PREDICTIONS FOR 3 NEW RAW ROWS (after reloading saved pipeline)")
    print(result[["predicted_default", "default_probability"]].to_string(index=False))
    print(f"Saved predictions: {OUTPUT_DIR / 'task2_new_predictions.csv'}")
    print("\nFeature engineering included: application year/month, income-to-loan ratio, log income.")
    print("Task 2 completed successfully.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "task2_loan_pipeline.joblib"
OUTPUT_PATH = BASE_DIR / "outputs" / "task2_new_predictions.csv"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Saved model not found. Run 'python task2.py' first.")
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
    model = joblib.load(MODEL_PATH)
    new_rows["predicted_default"] = model.predict(new_rows)
    new_rows["default_probability"] = model.predict_proba(new_rows)[:, 1]
    new_rows.to_csv(OUTPUT_PATH, index=False)
    print(new_rows[["predicted_default", "default_probability"]].to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

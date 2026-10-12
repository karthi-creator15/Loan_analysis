from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.dummy import DummyRegressor
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint, uniform

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def make_dataset(n=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    income = rng.lognormal(10.7, 0.55, n)
    loan_amount = rng.uniform(20000, 500000, n)
    credit_score = rng.normal(680, 55, n).clip(400, 850)
    age = rng.integers(21, 70, n).astype(float)
    debt_ratio = rng.uniform(0.01, 0.95, n)
    savings = rng.lognormal(9.0, 0.7, n)
    employment = rng.choice(["Salaried", "Self-employed", "Contract", "Unemployed"], n,
                            p=[0.52, 0.25, 0.16, 0.07])
    education = rng.choice(["High School", "Bachelor", "Master", "Doctorate"], n,
                           p=[0.28, 0.43, 0.23, 0.06])
    logit = (0.000007 * loan_amount - 0.0000018 * income
             - 0.009 * (credit_score - 650) + 2.4 * debt_ratio
             + 0.8 * (employment == "Unemployed")
             + 0.35 * (education == "High School")
             + rng.normal(0, 0.8, n))
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)
    X = pd.DataFrame({
        "income": income, "loan_amount": loan_amount, "credit_score": credit_score,
        "age": age, "debt_ratio": debt_ratio, "savings": savings,
        "employment_type": employment, "education_level": education
    })
    for col in ["income", "credit_score", "debt_ratio", "employment_type"]:
        X.loc[rng.random(n) < 0.025, col] = np.nan

    amount_target = pd.Series(np.clip(loan_amount + rng.normal(0, 25000, n), 10000, 650000),
                              name="requested_loan_amount")
    return X, pd.Series(y, name="default"), amount_target

def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])

def make_pipeline(X, estimator):
    return Pipeline([("preprocessor", make_preprocessor(X)), ("model", estimator)])

def main():
    print("=" * 72)
    print("WEEK 2 ASSESSMENT — TASK 4: EVALUATION AND TUNING")
    print("=" * 72)
    X, y, y_amount = make_dataset()
    X_trainval, X_test, y_trainval, y_test, amount_trainval, amount_test = train_test_split(
        X, y, y_amount, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval
    )
    amount_train = amount_trainval.loc[X_train.index]
    amount_val = amount_trainval.loc[X_val.index]
    print(f"Generated synthetic loan dataset: {len(X)} rows; no external CSV required.")
    print(f"Train={len(X_train)}, validation={len(X_val)}, held-out test={len(X_test)}")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=8,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }

    try:
        from xgboost import XGBClassifier
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=220, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=4
        )
        xgb_available = True
    except ImportError:
        xgb_available = False
        print("XGBoost not installed; using Logistic Regression and Random Forest for ROC comparison.")

    fitted = {}
    metric_rows = []
    val_probabilities = {}
    print("\n1. CLASSIFICATION METRICS ON VALIDATION SET")
    for name, estimator in classifiers.items():
        model = make_pipeline(X_train, estimator)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]
        fitted[name] = model
        val_probabilities[name] = probabilities
        cm = confusion_matrix(y_val, predictions, labels=[0, 1])
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_val, predictions),
            "Precision": precision_score(y_val, predictions, zero_division=0),
            "Recall": recall_score(y_val, predictions, zero_division=0),
            "F1": f1_score(y_val, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_val, probabilities),
            "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]
        }
        metric_rows.append(metrics)
        print(f"\n{name}:")
        print(f"Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, "
              f"Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}, AUC={metrics['ROC-AUC']:.4f}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(cm)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUTPUT_DIR / "task4_validation_metrics.csv", index=False)

    print("\nWhy accuracy can mislead:")
    print("With imbalanced classes, predicting the majority class can produce high accuracy")
    print("while missing many minority-class cases. Precision, recall, F1 and ROC-AUC add detail.")

    plt.figure(figsize=(8, 6))
    for name, probs in val_probabilities.items():
        fpr, tpr, _ = roc_curve(y_val, probs)
        auc_value = roc_auc_score(y_val, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Task 4: ROC Curves — Validation Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = OUTPUT_DIR / "task4_roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"\nSaved ROC plot: {roc_path}")

    threshold_rows = []
    print("\n2. THRESHOLD WITH AT LEAST 75% VALIDATION RECALL")
    for name, probs in val_probabilities.items():
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        candidates = [(float(th), float(p), float(r)) for p, r, th in zip(precision[:-1], recall[:-1], thresholds)
                      if r >= 0.75]
        if candidates:
            threshold, p_at, r_at = max(candidates, key=lambda item: item[1])
        else:
            threshold, p_at, r_at = 0.5, 0.0, 0.0
        threshold_rows.append({"Model": name, "Threshold": threshold,
                               "Precision_at_threshold": p_at, "Recall_at_threshold": r_at})
        print(f"{name}: threshold={threshold:.4f}, precision={p_at:.4f}, recall={r_at:.4f}")
    pd.DataFrame(threshold_rows).to_csv(OUTPUT_DIR / "task4_threshold_results.csv", index=False)

    print("\n3. GRID SEARCH — RANDOM FOREST")
    rf_pipe = make_pipeline(X_trainval, RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ))
    rf_grid = {
        "model__n_estimators": [100, 180],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [1, 3]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    grid.fit(X_trainval, y_trainval)
    print("Random Forest best parameters:", grid.best_params_)
    print(f"Random Forest best CV ROC-AUC: {grid.best_score_:.4f}")

    print("\n4. RANDOMIZED SEARCH — XGBOOST (n_iter=15)")
    if xgb_available:
        xgb_base = make_pipeline(X_trainval, XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=4
        ))
        distributions = {
            "model__n_estimators": randint(100, 350),
            "model__max_depth": randint(2, 7),
            "model__learning_rate": uniform(0.02, 0.18),
            "model__subsample": uniform(0.7, 0.3),
            "model__colsample_bytree": uniform(0.7, 0.3)
        }
        randomized = RandomizedSearchCV(
            xgb_base, distributions, n_iter=15, cv=cv, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=1, refit=True
        )
        randomized.fit(X_trainval, y_trainval)
        print("XGBoost best parameters:", randomized.best_params_)
        print(f"XGBoost best CV ROC-AUC: {randomized.best_score_:.4f}")
        tuned_xgb = randomized.best_estimator_
        pd.DataFrame(randomized.cv_results_).to_csv(OUTPUT_DIR / "task4_xgboost_random_search.csv", index=False)
    else:
        print("XGBoost package unavailable; performing the 15-iteration randomized search on Random Forest instead.")
        distributions = {
            "model__n_estimators": randint(80, 250),
            "model__max_depth": [None, 5, 8, 12],
            "model__min_samples_leaf": randint(1, 5)
        }
        randomized = RandomizedSearchCV(
            rf_pipe, distributions, n_iter=15, cv=cv, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=-1, refit=True
        )
        randomized.fit(X_trainval, y_trainval)
        print("Fallback Random Forest best parameters:", randomized.best_params_)
        print(f"Fallback best CV ROC-AUC: {randomized.best_score_:.4f}")
        tuned_xgb = None
        pd.DataFrame(randomized.cv_results_).to_csv(OUTPUT_DIR / "task4_random_search_fallback.csv", index=False)

    print("\n5. REGRESSION METRICS VS MEAN BASELINE")
    reg_features = ["income", "credit_score", "age", "debt_ratio", "savings"]
    X_reg = X[reg_features].copy()
    X_reg_trainval = X_reg.loc[X_trainval.index]
    X_reg_test = X_reg.loc[X_test.index]
    y_reg_trainval = amount_trainval
    y_reg_test = amount_test
    reg_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    reg_pipe.fit(X_reg_trainval, y_reg_trainval)
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_reg_trainval), 1)), y_reg_trainval)
    reg_pred = reg_pipe.predict(X_reg_test)
    base_pred = baseline.predict(np.zeros((len(y_reg_test), 1)))
    def reg_metrics(actual, pred):
        return {
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
            "R2": r2_score(actual, pred)
        }
    model_reg_metrics = reg_metrics(y_reg_test, reg_pred)
    baseline_metrics = reg_metrics(y_reg_test, base_pred)
    print("Random Forest regression:", {k: round(v, 4) for k, v in model_reg_metrics.items()})
    print("Mean baseline:", {k: round(v, 4) for k, v in baseline_metrics.items()})
    pd.DataFrame([
        {"Model": "RandomForestRegressor", **model_reg_metrics},
        {"Model": "Mean baseline", **baseline_metrics}
    ]).to_csv(OUTPUT_DIR / "task4_regression_metrics.csv", index=False)

    candidate_models = {"GridSearchCV Random Forest": grid.best_estimator_}
    if tuned_xgb is not None:
        candidate_models["RandomizedSearchCV XGBoost"] = tuned_xgb
    best_name, best_model, best_auc = None, None, -np.inf
    for name, model in candidate_models.items():
        # CV score is the model-selection signal; compare with other initial validation models.
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_name, best_model, best_auc = name, model, auc
    print("\n6. FINAL HELD-OUT TEST EVALUATION (performed once)")
    test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    print("Best model:", best_name)
    print(f"Test ROC-AUC={roc_auc_score(y_test, test_prob):.4f}")
    print(f"Test F1={f1_score(y_test, test_pred, zero_division=0):.4f}")
    print("Test confusion matrix [[TN, FP], [FN, TP]]:")
    print(test_cm)
    pd.DataFrame([{
        "Best model": best_name,
        "Test ROC-AUC": roc_auc_score(y_test, test_prob),
        "Test F1": f1_score(y_test, test_pred, zero_division=0),
        "TN": test_cm[0, 0], "FP": test_cm[0, 1], "FN": test_cm[1, 0], "TP": test_cm[1, 1]
    }]).to_csv(OUTPUT_DIR / "task4_final_test_result.csv", index=False)
    print("\nTask 4 completed.")
    print("Note: all scores use synthetic educational data, not real bank customer records.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()

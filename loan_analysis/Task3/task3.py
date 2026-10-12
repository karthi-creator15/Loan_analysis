from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def make_loan_dataset(n=900, random_state=42):
    """Create a reproducible synthetic loan dataset, with mixed data types."""
    rng = np.random.default_rng(random_state)
    numeric = rng.normal(size=(n, 6))
    income = np.exp(10.7 + 0.55 * numeric[:, 0])
    loan_amount = np.clip(180000 + 90000 * numeric[:, 1], 20000, 600000)
    credit_score = np.clip(680 + 50 * numeric[:, 2], 400, 850)
    age = np.clip(40 + 10 * numeric[:, 3], 21, 75)
    debt_ratio = np.clip(0.35 + 0.15 * numeric[:, 4], 0.01, 0.95)
    savings = np.clip(np.exp(9.0 + 0.7 * numeric[:, 5]), 1000, 1000000)
    employment = rng.choice(["Salaried", "Self-employed", "Contract"], size=n, p=[0.58, 0.28, 0.14])
    education = rng.choice(["High School", "Bachelor", "Master"], size=n, p=[0.3, 0.5, 0.2])
    logit = (0.000006 * loan_amount - 0.0000018 * income
             - 0.008 * (credit_score - 650) + 2.1 * debt_ratio
             + 0.5 * (employment == "Contract") + rng.normal(0, 0.7, n))
    prob = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob)
    X = pd.DataFrame({
        "income": income, "loan_amount": loan_amount, "credit_score": credit_score,
        "age": age, "debt_ratio": debt_ratio, "savings": savings,
        "employment_type": employment, "education_level": education
    })
    for col in ["income", "credit_score", "debt_ratio", "employment_type"]:
        mask = rng.random(n) < 0.025
        X.loc[mask, col] = np.nan
    amount_target = loan_amount + rng.normal(0, 15000, n)
    return X, pd.Series(default, name="default"), pd.Series(amount_target, name="requested_loan_amount")

def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

def main():
    print("=" * 72)
    print("WEEK 2 ASSESSMENT — TASK 3: SUPERVISED AND UNSUPERVISED MODELS")
    print("=" * 72)
    X, y, y_amount = make_loan_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Generated synthetic dataset: {len(X)} rows, {X.shape[1]} raw features.")
    print("No external CSV required.")

    results = []
    fitted_models = {}
    print("\n1. FOUR CLASSIFIERS — VALIDATION ACCURACY")
    for name, estimator in [
        ("LogisticRegression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
        ("RandomForestClassifier", RandomForestClassifier(
            n_estimators=250, max_features="sqrt", oob_score=True,
            bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ]:
        model = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = accuracy_score(y_val, pred)
        results.append({"Model": name, "Validation score": score})
        fitted_models[name] = model
        print(f"{name}: validation accuracy={score:.4f}")
        if name == "RandomForestClassifier":
            print(f"Random Forest OOB score: {model.named_steps['model'].oob_score_:.4f}")

    try:
        from xgboost import XGBClassifier
        xgb_estimator = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=4
        )
        xgb_model = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", xgb_estimator)])
        xgb_model.fit(X_train, y_train)
        xgb_score = accuracy_score(y_val, xgb_model.predict(X_val))
        results.append({"Model": "XGBClassifier", "Validation score": xgb_score})
        fitted_models["XGBClassifier"] = xgb_model
        print(f"XGBClassifier: validation accuracy={xgb_score:.4f}")
        print("Note: XGBoost fitted with validation set reserved for model comparison; "
              "early stopping API varies by installed XGBoost version.")
    except ImportError:
        print("XGBoost not installed; skipped without stopping the project.")
        print("To enable it, install with: pip install xgboost")
    except Exception as exc:
        print(f"XGBoost was skipped due to an environment/API compatibility issue: {exc}")
        print("The remaining Task 3 outputs have still been generated.")

    results_df = pd.DataFrame(results).sort_values("Validation score", ascending=False)
    results_df.to_csv(OUTPUT_DIR / "task3_classifier_results.csv", index=False)
    print("\nResults table:")
    print(results_df.to_string(index=False))
    print(f"Saved: {OUTPUT_DIR / 'task3_classifier_results.csv'}")

    print("\n2. LOGISTIC REGRESSION: predict_proba VS predict (5 rows)")
    logistic = fitted_models["LogisticRegression"]
    proba = logistic.predict_proba(X_val.iloc[:5])[:, 1]
    pred = logistic.predict(X_val.iloc[:5])
    print(pd.DataFrame({"default_probability": proba, "predicted_class": pred}).to_string(index=False))

    print("\n3. TOP 5 FEATURE IMPORTANCES")
    forest_pipe = fitted_models["RandomForestClassifier"]
    forest_pre = forest_pipe.named_steps["preprocessor"]
    forest = forest_pipe.named_steps["model"]
    feature_names = forest_pre.get_feature_names_out()
    forest_imp = pd.DataFrame({"feature": feature_names, "importance": forest.feature_importances_})
    forest_imp = forest_imp.sort_values("importance", ascending=False).head(5)
    print("Random Forest:")
    print(forest_imp.to_string(index=False))
    forest_imp.to_csv(OUTPUT_DIR / "task3_random_forest_top5.csv", index=False)

    if "XGBClassifier" in fitted_models:
        xgb_pipe = fitted_models["XGBClassifier"]
        xgb_pre = xgb_pipe.named_steps["preprocessor"]
        xgb = xgb_pipe.named_steps["model"]
        xgb_imp = pd.DataFrame({
            "feature": xgb_pre.get_feature_names_out(),
            "importance": xgb.feature_importances_
        }).sort_values("importance", ascending=False).head(5)
        print("XGBoost:")
        print(xgb_imp.to_string(index=False))
        xgb_imp.to_csv(OUTPUT_DIR / "task3_xgboost_top5.csv", index=False)
    else:
        print("XGBoost feature importance unavailable because XGBoost was not run.")

    print("\n4. LINEAR REGRESSION, RIDGE AND LASSO")
    X_reg = X[["income", "credit_score", "age", "debt_ratio", "savings"]].copy()
    X_reg = X_reg.fillna(X_reg.median(numeric_only=True))
    Xr_train, Xr_val, yr_train, yr_val = train_test_split(
        X_reg, y_amount, test_size=0.20, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    Xr_train_s = scaler.fit_transform(Xr_train)
    Xr_val_s = scaler.transform(Xr_val)
    reg_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1000.0, max_iter=10000)
    }
    coef_rows = []
    for name, model in reg_models.items():
        model.fit(Xr_train_s, yr_train)
        pred_amount = model.predict(Xr_val_s)
        rmse = float(np.sqrt(mean_squared_error(yr_val, pred_amount)))
        r2 = r2_score(yr_val, pred_amount)
        print(f"{name}: validation RMSE={rmse:.2f}, R2={r2:.4f}")
        for feature, coef in zip(X_reg.columns, np.ravel(model.coef_)):
            coef_rows.append({"model": name, "feature": feature, "coefficient": float(coef)})
        if name == "Lasso":
            print(f"Lasso zero coefficients: {int(np.sum(np.isclose(model.coef_, 0.0)))}")
    pd.DataFrame(coef_rows).to_csv(OUTPUT_DIR / "task3_regression_coefficients.csv", index=False)

    print("\n5. K-MEANS CLUSTERING (k=2..8)")
    cluster_data = X[["income", "loan_amount", "credit_score"]].copy()
    cluster_data = cluster_data.fillna(cluster_data.median(numeric_only=True))
    cluster_scaled = StandardScaler().fit_transform(cluster_data)
    ks, inertias, silhouettes = [], [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(cluster_scaled)
        sil = silhouette_score(cluster_scaled, labels)
        ks.append(k)
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        print(f"k={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Task 3: K-Means Elbow Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    elbow_path = OUTPUT_DIR / "task3_kmeans_elbow.png"
    plt.savefig(elbow_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ks, silhouettes, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Task 3: K-Means Silhouette Scores")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    silhouette_path = OUTPUT_DIR / "task3_kmeans_silhouette.png"
    plt.savefig(silhouette_path, dpi=150)
    plt.close()
    pd.DataFrame({"k": ks, "inertia": inertias, "silhouette_score": silhouettes}).to_csv(
        OUTPUT_DIR / "task3_kmeans_metrics.csv", index=False
    )
    print(f"Saved elbow plot: {elbow_path}")
    print(f"Saved silhouette plot: {silhouette_path}")
    print("Segments can be profiled by comparing average income, loan amount, and credit score per cluster.")
    print("\nTask 3 completed.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()

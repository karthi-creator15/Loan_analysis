from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 68)
    print("WEEK 2 ASSESSMENT — TASK 1: CONCEPTS, SPLITS AND LEAKAGE")
    print("=" * 68)
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data.copy(), dataset.target.copy()
    print(f"Dataset: Built-in Breast Cancer Wisconsin dataset | rows={len(X)} | features={X.shape[1]}")
    print("No external CSV required.")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE,
        stratify=y_trainval)
    print("\n1. STRATIFIED 60/20/20 SPLIT")
    print(f"Train:      {len(X_train):3d} rows ({len(X_train)/len(X):.1%})")
    print(f"Validation: {len(X_val):3d} rows ({len(X_val)/len(X):.1%})")
    print(f"Test:       {len(X_test):3d} rows ({len(X_test)/len(X):.1%})")
    print("The test set is held aside for final evaluation only.")

    print("\n2. DECISION TREE OVERFITTING EXPERIMENT")
    depths = [1, 3, 5, 10, None]
    train_scores, val_scores = [], []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        tr = accuracy_score(y_train, model.predict(X_train))
        va = accuracy_score(y_val, model.predict(X_val))
        train_scores.append(tr)
        val_scores.append(va)
        print(f"Depth {str(depth):>4}: train accuracy={tr:.4f}, validation accuracy={va:.4f}")
    labels = [str(d) if d is not None else "None" for d in depths]
    plt.figure(figsize=(8, 5))
    plt.plot(labels, train_scores, marker="o", label="Training accuracy")
    plt.plot(labels, val_scores, marker="o", label="Validation accuracy")
    plt.xlabel("Decision tree max_depth")
    plt.ylabel("Accuracy")
    plt.title("Task 1: Decision Tree Overfitting")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "task1_overfitting.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

    print("\n3. LOGISTIC REGRESSION — 5-FOLD CROSS-VALIDATION")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    scores = cross_val_score(pipeline, X_trainval, y_trainval, cv=cv, scoring="accuracy")
    print("Fold accuracies:", np.round(scores, 4))
    print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f} (mean +/- standard deviation)")

    print("\n4. DATA LEAKAGE: INCORRECT VS CORRECT SCALING")
    X_demo_train, X_demo_test = train_test_split(
        X, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    leaky_scaler = StandardScaler()
    X_scaled_before_split = leaky_scaler.fit_transform(X)
    correct_scaler = StandardScaler()
    X_correct_train = correct_scaler.fit_transform(X_demo_train)
    X_correct_test = correct_scaler.transform(X_demo_test)
    print("Incorrect: fit the scaler on the full dataset before splitting.")
    print("Correct: fit the scaler on training data only; transform validation/test data afterward.")
    print(f"Shapes — incorrectly scaled full data: {X_scaled_before_split.shape}; "
          f"correct train: {X_correct_train.shape}; held-out transform: {X_correct_test.shape}")
    print("The incorrect scaler learns feature statistics from held-out rows, causing leakage.")
    print("The correct scaler learns only from training rows and applies those statistics to held-out rows.")
    print("A Pipeline keeps preprocessing inside each cross-validation fold.")

    print("\n5. SUPERVISED VS UNSUPERVISED PROBLEMS")
    problems = [
        ("Spam detection", "Supervised", "Examples are labeled spam or not spam."),
        ("Customer segmentation", "Unsupervised", "Groups customers without predefined target labels."),
        ("House price prediction", "Supervised", "Historical examples include known prices."),
        ("Anomaly detection", "Usually unsupervised", "Finds unusual patterns without labels; labeled anomaly detection can be supervised."),
    ]
    for name, kind, why in problems:
        print(f"- {name}: {kind}. {why}")

    print("\n6. MACHINE LEARNING WORKFLOW — 7 STEPS")
    steps = [
        "Define the problem and success metric.",
        "Collect and understand the data.",
        "Clean the data and perform exploratory data analysis.",
        "Split into training, validation, and test sets.",
        "Preprocess and engineer features using training-only fitting.",
        "Train models; validate, compare, and tune with cross-validation.",
        "Evaluate the chosen model once on the held-out test set, then document, deploy, and monitor.",
    ]
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")
    print("\nTask 1 completed successfully. Test data was not used for model selection.")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()

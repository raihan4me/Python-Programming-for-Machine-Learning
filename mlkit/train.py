"""Train and compare baseline models with stratified cross-validation."""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from mlkit.data_io import read_customers
from mlkit.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, build_pipeline

CLEAN_PATH = "data/processed/customers_clean.csv"


def load_xy(path=CLEAN_PATH):
    df = read_customers(path)
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]


def split(X, y):
    """Hold out 20% as a test set, preserving the churn ratio (stratify)."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    X, y = load_xy()
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"train={X_train.shape} test={X_test.shape} "
          f"train_churn_rate={y_train.mean():.3f} test_churn_rate={y_test.mean():.3f}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                random_state=42, n_jobs=2),
    }
    report = {}
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        report[name] = {"cv_roc_auc_mean": round(float(scores.mean()), 4),
                        "cv_roc_auc_std": round(float(scores.std()), 4),
                        "test": metrics}
        print(f"\n[{name}] cv_roc_auc={scores.mean():.4f} +/- {scores.std():.4f}")
        print(f"[{name}] test: " + ", ".join(f"{k}={metrics[k]}"
                                            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]))
        if name == "logistic_regression":
            names = pipe.named_steps["prep"].get_feature_names_out()
            coefs = pd.Series(pipe.named_steps["clf"].coef_[0], index=names)
            top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(6)
            print("Top coefficients by magnitude:\n" + top.round(3).to_string())
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(report, indent=2))
    print("\nSaved reports/metrics.json")


if __name__ == "__main__":
    main()

"""Grid search, recall-driven threshold selection and model persistence."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from mlkit.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_pipeline
from mlkit.train import evaluate, load_xy, split

TARGET_RECALL = 0.75
MODEL_PATH = Path("models/churn_model.joblib")
META_PATH = Path("models/model_metadata.json")


def choose_threshold(y_true, scores, target_recall):
    """Highest threshold whose recall is >= target_recall."""
    _, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]])


def describe(params):
    clf = type(params["clf"]).__name__
    rest = ", ".join(f"{k.split('__', 1)[1]}={v}" for k, v in sorted(params.items()) if k != "clf")
    return f"{clf}({rest})"


def main():
    X, y = load_xy()
    X_train, X_test, y_train, y_test = split(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = [
        {"clf": [LogisticRegression(max_iter=1000)],
         "clf__C": [0.01, 0.1, 1.0, 10.0],
         "clf__class_weight": [None, "balanced"]},
        {"clf": [RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)],
         "clf__max_depth": [6, None],
         "clf__min_samples_leaf": [1, 10]},
    ]
    search = GridSearchCV(build_pipeline(LogisticRegression(max_iter=1000)), param_grid,
                          scoring="roc_auc", cv=cv, n_jobs=2)
    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    results["candidate"] = results["params"].map(describe)
    top = results.sort_values("rank_test_score")[
        ["rank_test_score", "candidate", "mean_test_score", "std_test_score"]].head(5)
    print(f"Evaluated {len(results)} candidates x {cv.get_n_splits()} folds")
    print(top.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = search.best_estimator_
    print(f"\nBest: {describe(search.best_params_)}  cv_roc_auc={search.best_score_:.4f}")

    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=2)[:, 1]
    threshold = choose_threshold(y_train.to_numpy(), oof, TARGET_RECALL)
    print(f"Threshold for recall >= {TARGET_RECALL} (out-of-fold, train only): {threshold:.4f}")

    test_default = evaluate(best, X_test, y_test, threshold=0.5)
    test_tuned = evaluate(best, X_test, y_test, threshold=threshold)
    for label, m in [("default 0.50", test_default), ("tuned", test_tuned)]:
        print(f"test @ {label:>12}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} roc_auc={m['roc_auc']:.3f} cm={m['confusion_matrix']}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES},
        "best_candidate": describe(search.best_params_),
        "cv_roc_auc": round(float(search.best_score_), 4),
        "target_recall": TARGET_RECALL,
        "threshold": round(float(threshold), 6),
        "test_metrics_default_threshold": test_default,
        "test_metrics_tuned_threshold": test_tuned,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {MODEL_PATH} and {META_PATH}")


if __name__ == "__main__":
    main()

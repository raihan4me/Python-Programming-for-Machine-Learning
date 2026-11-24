"""Batch inference using the persisted pipeline and tuned threshold."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from mlkit.clean import clean_dataframe
from mlkit.data_io import read_customers
from mlkit.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

REQUIRED = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_artifacts(model_path="models/churn_model.joblib", meta_path="models/model_metadata.json"):
    model = joblib.load(model_path)
    meta = json.loads(Path(meta_path).read_text())
    return model, meta


def predict_frame(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    clean = clean_dataframe(df)
    proba = model.predict_proba(clean[REQUIRED])[:, 1]
    out = pd.DataFrame({"customer_id": clean.get("customer_id", pd.Series(range(len(clean))))})
    out["churn_probability"] = proba.round(4)
    out["churn_pred"] = (proba >= threshold).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser(description="Score customers for churn risk")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, meta = load_artifacts()
    preds = predict_frame(model, read_customers(args.input), meta["threshold"])
    preds.to_csv(args.output, index=False)
    print(f"threshold={meta['threshold']:.4f} model={meta['best_candidate']}")
    print(preds.to_string(index=False))
    print(f"Saved {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()

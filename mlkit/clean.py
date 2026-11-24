"""Deterministic cleaning rules for NorthWave customer data."""
from pathlib import Path

import numpy as np
import pandas as pd

from mlkit.data_io import read_customers

CONTRACT_MAP = {"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"}
MONTHLY_MIN, MONTHLY_MAX = 0.0, 200.0
RAW_PATH = Path("data/raw/customers.csv")
CLEAN_PATH = Path("data/processed/customers_clean.csv")


def normalize_contract(s: pd.Series) -> pd.Series:
    """Strip, lower-case and map contract labels to canonical values."""
    out = s.str.strip().str.lower().map(CONTRACT_MAP)
    unknown = s.notna() & out.isna()
    if unknown.any():
        raise ValueError(f"Unknown contract values: {sorted(s[unknown].unique().tolist())}")
    return out


def coerce_numeric(s: pd.Series) -> pd.Series:
    """Convert to float; unparseable values become NaN."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def mask_outliers(s: pd.Series, lower: float, upper: float) -> pd.Series:
    """Replace values outside [lower, upper] with NaN."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return s.where((s >= lower) & (s <= upper))


def encode_target(s: pd.Series) -> pd.Series:
    """Map Yes/No to 1/0."""
    out = s.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if out.isna().any():
        raise ValueError(f"Unknown target values: {sorted(s[out.isna()].astype(str).unique().tolist())}")
    return out.astype(int)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules and return a new DataFrame."""
    out = df.copy().drop_duplicates()
    out["contract"] = normalize_contract(out["contract"])
    out["total_charges"] = coerce_numeric(out["total_charges"])
    out["monthly_charges"] = mask_outliers(out["monthly_charges"], MONTHLY_MIN, MONTHLY_MAX)
    if "churn" in out.columns:
        out["churn"] = encode_target(out["churn"])
    return out.reset_index(drop=True)


def main() -> None:
    raw = read_customers(RAW_PATH)
    clean = clean_dataframe(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)
    na = clean.isna().sum()
    print(f"Raw rows: {len(raw)}")
    print(f"Duplicates removed: {len(raw) - len(clean)}")
    print(f"Rows after cleaning: {len(clean)}")
    print("Missing values after cleaning (imputed later inside the model pipeline):")
    print(na[na > 0].to_string())
    print(f"Contract categories: {sorted(clean['contract'].unique())}")
    print(f"total_charges dtype: {clean['total_charges'].dtype}")
    print(f"Churn rate: {clean['churn'].mean():.3f}")
    print(f"Saved {CLEAN_PATH}")


if __name__ == "__main__":
    main()

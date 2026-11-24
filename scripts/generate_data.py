"""Generate a deterministic, deliberately messy telecom churn dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

N = 5000
OUT = Path("data/raw/customers.csv")


def main() -> None:
    rng = np.random.default_rng(42)

    tenure = rng.integers(1, 73, size=N)
    monthly = np.clip(rng.normal(70, 25, size=N), 18, 150).round(2)
    total = (tenure * monthly * rng.uniform(0.95, 1.05, size=N)).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber optic", "None"], size=N, p=[0.35, 0.45, 0.20])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
                         size=N, p=[0.35, 0.20, 0.225, 0.225])
    support_calls = rng.poisson(2, size=N)
    senior = rng.binomial(1, 0.16, size=N)

    logit = (-1.0
             + 1.5 * (contract == "Month-to-month")
             - 0.8 * (contract == "Two year")
             - 0.04 * tenure
             + 0.015 * (monthly - 70)
             + 0.35 * support_calls
             + 0.6 * (internet == "Fiber optic")
             + 0.3 * senior)
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = np.where(rng.random(N) < prob, "Yes", "No")

    df = pd.DataFrame({
        "customer_id": [f"C{10000 + i}" for i in range(N)],
        "tenure_months": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract": contract,
        "internet_service": internet,
        "payment_method": payment,
        "support_calls": support_calls,
        "senior_citizen": senior,
        "churn": churn,
    })

    # ---- inject data-quality problems (disjoint row sets, exact counts) ----
    perm = rng.permutation(N)
    idx_missing, idx_blank = perm[:150], perm[150:250]
    idx_outlier, idx_case = perm[250:270], perm[270:470]

    df["total_charges"] = df["total_charges"].map(lambda v: f"{v:.2f}")   # numbers as text
    df.loc[idx_blank, "total_charges"] = " "                               # blank strings
    df.loc[idx_missing, "monthly_charges"] = np.nan                       # true missing values
    df.loc[idx_outlier, "monthly_charges"] = 9999.0                       # data-entry errors
    messy = {"Month-to-month": "month-to-month ", "One year": "ONE YEAR", "Two year": " two year"}
    df.loc[idx_case, "contract"] = df.loc[idx_case, "contract"].map(messy)  # inconsistent labels
    dups = df.iloc[perm[470:520]]                                          # 50 exact duplicates
    df = pd.concat([df, dups], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(f"rows={len(df)} columns={df.shape[1]}")
    print("injected: missing_monthly=150 blank_total=100 outliers=20 messy_contract=200 duplicates=50")
    print(f"churn_rate={(df['churn'] == 'Yes').mean():.3f}")


if __name__ == "__main__":
    main()

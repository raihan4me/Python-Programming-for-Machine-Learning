import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlkit.features import build_pipeline


def make_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [f"T{i}" for i in range(n)],
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n),
        "total_charges": rng.uniform(20, 8000, n),
        "support_calls": rng.poisson(2, n),
        "senior_citizen": rng.integers(0, 2, n),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "internet_service": rng.choice(["DSL", "Fiber optic", "None"], n),
        "payment_method": rng.choice(["Electronic check", "Mailed check"], n),
    })
    df.loc[:9, "monthly_charges"] = np.nan
    y = (df["contract"] == "Month-to-month").astype(int)
    return df, y


def test_pipeline_handles_nan_unknown_and_extra_columns():
    X, y = make_frame()
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    X_new = X.head(3).copy()
    X_new.loc[X_new.index[0], "internet_service"] = "Satellite"   # unseen category
    proba = pipe.predict_proba(X_new)
    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert "customer_id" not in " ".join(pipe.named_steps["prep"].get_feature_names_out())


def test_pipeline_learns_signal():
    X, y = make_frame()
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    assert pipe.score(X, y) > 0.95

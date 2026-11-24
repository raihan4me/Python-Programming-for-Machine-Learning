import numpy as np
import pandas as pd
import pytest

from mlkit.clean import (clean_dataframe, coerce_numeric, encode_target,
                         mask_outliers, normalize_contract)


def test_normalize_contract():
    s = pd.Series(["month-to-month ", "ONE YEAR", " two year", "Two year"])
    assert normalize_contract(s).tolist() == ["Month-to-month", "One year", "Two year", "Two year"]


def test_normalize_contract_unknown_raises():
    with pytest.raises(ValueError, match="weekly"):
        normalize_contract(pd.Series(["weekly", "One year"]))


def test_coerce_numeric():
    out = coerce_numeric(pd.Series(["10.5", " ", "3"]))
    assert out.iloc[0] == 10.5 and np.isnan(out.iloc[1]) and out.iloc[2] == 3.0


def test_mask_outliers():
    out = mask_outliers(pd.Series([50.0, 9999.0, np.nan, -5.0]), 0, 200)
    assert out.iloc[0] == 50.0
    assert out.iloc[1:].isna().all()


def test_encode_target():
    assert encode_target(pd.Series(["Yes", "No", " yes "])).tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Maybe"]))


def test_clean_dataframe():
    raw = pd.DataFrame({
        "customer_id": ["C1", "C2", "C2", "C3"],
        "tenure_months": [1, 2, 2, 3],
        "monthly_charges": [50.0, 9999.0, 9999.0, np.nan],
        "total_charges": ["50.00", " ", " ", "300.10"],
        "contract": ["ONE YEAR", "month-to-month ", "month-to-month ", "Two year"],
        "internet_service": ["DSL", "None", "None", "Fiber optic"],
        "payment_method": ["Mailed check"] * 4,
        "support_calls": [0, 1, 1, 2],
        "senior_citizen": [0, 0, 0, 1],
        "churn": ["No", "Yes", "Yes", "No"],
    })
    before = raw.copy()
    out = clean_dataframe(raw)
    pd.testing.assert_frame_equal(raw, before)          # input must not be mutated
    assert len(out) == 3
    assert out["contract"].tolist() == ["One year", "Month-to-month", "Two year"]
    assert out["churn"].tolist() == [0, 1, 0]
    assert out["monthly_charges"].isna().sum() == 2
    assert out["total_charges"].isna().sum() == 1
    assert out["total_charges"].dtype == float

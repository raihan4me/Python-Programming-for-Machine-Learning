"""REST API for churn predictions. Run from the project root."""
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlkit.predict import load_artifacts, predict_frame

MODEL, META = load_artifacts()
app = FastAPI(title="NorthWave Churn API", version="1.0.0")


class Customer(BaseModel):
    customer_id: str
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: Optional[float] = Field(default=None, ge=0)
    total_charges: Optional[float] = Field(default=None, ge=0)
    contract: str
    internet_service: str
    payment_method: str
    support_calls: int = Field(ge=0)
    senior_citizen: int = Field(ge=0, le=1)


@app.get("/health")
def health():
    return {"status": "ok", "model": META["best_candidate"], "threshold": META["threshold"],
            "sklearn": META["sklearn"]}


@app.post("/predict")
def predict(customers: List[Customer]):
    df = pd.DataFrame([c.model_dump() for c in customers])
    try:
        out = predict_frame(MODEL, df, META["threshold"])
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    return {"threshold": META["threshold"], "predictions": out.to_dict(orient="records")}

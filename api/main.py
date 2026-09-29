"""
Run locally:      uvicorn api.main:app --reload --port 8000
Run in Docker:     see Dockerfile / docker-compose.yml
Test it:           curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_loan.json
"""
from __future__ import annotations

import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schema import LoanPayload, PredictionResponse
from src.preprocessing import from_logit

MODEL_PATH = os.environ.get("MODEL_BUNDLE_PATH", "artifacts/model_bundle.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

app = FastAPI(title="ARC Recovery Rate API", version=MODEL_VERSION)

_bundle = None


@app.on_event("startup")
def load_model():
    global _bundle
    _bundle = joblib.load(MODEL_PATH)
    print(f"[startup] loaded {_bundle['model_type']} bundle trained on "
          f"{_bundle['n_training_rows']} rows, CV MAE={_bundle['cv_metrics']['mae']:.4f}")


@app.get("/health")
def health():
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_type": _bundle["model_type"],
            "cv_metrics": _bundle["cv_metrics"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: LoanPayload):
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    row = payload.model_dump()
    df = pd.DataFrame([row])
    X = _bundle["preprocessor"].transform(df)
    logit_pred = _bundle["model"].predict(X)[0]
    pred = float(from_logit(logit_pred))

    return PredictionResponse(
        loan_id=payload.loan_id,
        predicted_recovery_rate=round(pred, 6),
        model_type=_bundle["model_type"],
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(payloads: list[LoanPayload]):
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    df = pd.DataFrame([p.model_dump() for p in payloads])
    X = _bundle["preprocessor"].transform(df)
    logit_preds = _bundle["model"].predict(X)
    preds = from_logit(logit_preds)

    return [
        PredictionResponse(
            loan_id=p.loan_id,
            predicted_recovery_rate=round(float(pred), 6),
            model_type=_bundle["model_type"],
            model_version=MODEL_VERSION,
        )
        for p, pred in zip(payloads, preds)
    ]

# backend/python-service/app.py

import os
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dateutil.parser import isoparse
import joblib
import requests

# ─── FastAPI setup ───────────────────────────────────────────────────────────
app = FastAPI(
    title="GitHub PR Time-to-Merge API",
    description="Given a GitHub repo & PR number, predicts how many hours until merge, or takes manual PR data and estimates merge time",
    version="0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],   # adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── load model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pr_time_model.joblib")
model = joblib.load(MODEL_PATH)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in .env")

# ─── request/response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    owner:     str = Field(..., example="facebook")
    repo:      str = Field(..., example="react")
    pr_number: int = Field(..., example=32812)

class PredictResponse(BaseModel):
    predicted_hours: float
    title:            str
    body:             Optional[str]
    created_at:       datetime
    merged_at:        datetime
    actual_hours:     float

class EstimateRequest(BaseModel):
    title:      str
    body:       Optional[str] = None
    num_labels: int
    is_closed:  bool

class EstimateResponse(BaseModel):
    predicted_hours: float

# ─── health check ─────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def read_root():
    return {
        "message": "✅ API up",
        "endpoints": {
            "predict":  {"method": "POST", "path": "/predict"},
            "estimate": {"method": "POST", "path": "/estimate"}
        },
    }

# ─── prediction endpoint (pull from GitHub) ──────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    url = f"https://api.github.com/repos/{req.owner}/{req.repo}/pulls/{req.pr_number}"
    headers = {
        "Accept":        "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}"
    }
    resp = requests.get(url, headers=headers)
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="PR not found")
    resp.raise_for_status()
    pr = resp.json()

    created = isoparse(pr["created_at"])
    merged  = isoparse(pr["merged_at"])
    actual_hours = (merged - created).total_seconds() / 3600

    title_length = len(pr.get("title") or "")
    body_length  = len(pr.get("body")  or "")
    num_labels   = len(pr.get("labels") or [])
    is_closed    = 1 if pr.get("state", "").lower() == "closed" else 0

    pred = model.predict([[title_length, body_length, num_labels, is_closed]])[0]

    return PredictResponse(
        predicted_hours=float(pred),
        title=pr.get("title", ""),
        body=pr.get("body"),
        created_at=created,
        merged_at=merged,
        actual_hours=actual_hours,
    )

# ─── estimate endpoint (manual PR data) ──────────────────────────────────────
@app.post("/estimate", response_model=EstimateResponse, tags=["prediction"])
def estimate(req: EstimateRequest):
    title_length = len(req.title or "")
    body_length  = len(req.body  or "")
    num_labels   = req.num_labels
    is_closed    = 1 if req.is_closed else 0

    pred = model.predict([[title_length, body_length, num_labels, is_closed]])[0]
    return EstimateResponse(predicted_hours=float(pred))

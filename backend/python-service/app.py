# backend/python-service/app.py

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import requests
import pandas as pd
from dotenv import load_dotenv

# ─── load environment & tokens ────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in .env")

# ─── feature column names must match training ─────────────────────
FEATURE_COLS = ["title_length", "body_length", "num_labels", "is_closed"]

# ─── app & CORS setup ─────────────────────────────────────────────
app = FastAPI(
    title="GitHub PR Time-to-Merge Estimator",
    version="0.2"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ─── load your trained model ──────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pr_time_model.joblib")
model = joblib.load(MODEL_PATH)

# ─── shared response model ────────────────────────────────────────
class PredictResponse(BaseModel):
    predicted_hours: float

# ─── live GitHub‐fetched PR endpoint ──────────────────────────────
class PredictRequest(BaseModel):
    owner: str
    repo: str
    pr_number: int

@app.post("/predict", response_model=PredictResponse, tags=["github"])
def predict_from_github(req: PredictRequest):
    url = f"https://api.github.com/repos/{req.owner}/{req.repo}/pulls/{req.pr_number}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}"
    }
    gh = requests.get(url, headers=headers)
    if gh.status_code == 404:
        raise HTTPException(404, "PR not found")
    gh.raise_for_status()
    pr = gh.json()

    tl = len(pr.get("title") or "")
    bl = len(pr.get("body") or "")
    nl = len(pr.get("labels") or [])
    ic = 1 if pr.get("state", "").lower() == "closed" else 0

    # wrap in DataFrame so feature names match training
    df = pd.DataFrame([[tl, bl, nl, ic]], columns=FEATURE_COLS)
    hours = model.predict(df)[0]
    return PredictResponse(predicted_hours=float(hours))

# ─── manual input endpoint ────────────────────────────────────────
class ManualRequest(BaseModel):
    title: str = Field(..., example="Fix infinite-loop spinner")
    body: str = Field("", example="This PR addresses …")
    num_labels: int = Field(0, example=2)
    is_closed: bool = Field(False, example=False)

@app.post("/estimate", response_model=PredictResponse, tags=["manual"])
def predict_manual(req: ManualRequest):
    tl = len(req.title or "")
    bl = len(req.body or "")
    nl = req.num_labels
    ic = 1 if req.is_closed else 0

    df = pd.DataFrame([[tl, bl, nl, ic]], columns=FEATURE_COLS)
    hours = model.predict(df)[0]
    return PredictResponse(predicted_hours=float(hours))

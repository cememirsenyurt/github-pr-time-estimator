# backend/python-service/app.py

import os
import json
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dateutil.parser import isoparse
import joblib
import requests
import numpy as np

# ─── FastAPI setup ───────────────────────────────────────────────────────────
app = FastAPI(
    title="GitHub PR Time-to-Merge API",
    description="Advanced ML-powered prediction of GitHub PR merge times using XGBoost and comprehensive feature engineering",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── load model and metrics ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pr_time_model.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "model", "model_metrics.json")
COMPARISON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "model", "model_comparison.json")

model_artifact = joblib.load(MODEL_PATH)
model = model_artifact['model']
feature_columns = model_artifact['feature_columns']

# Load metrics
model_metrics = {}
model_comparison = {}
try:
    with open(METRICS_PATH, 'r') as f:
        model_metrics = json.load(f)
except FileNotFoundError:
    print("[WARN] Model metrics file not found")

try:
    with open(COMPARISON_PATH, 'r') as f:
        model_comparison = json.load(f)
except FileNotFoundError:
    print("[WARN] Model comparison file not found")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in .env")

# ─── Feature engineering functions ───────────────────────────────────────────
def extract_features(title: str, body: str, num_labels: int, is_closed: bool,
                     created_at: Optional[datetime] = None, author_pr_count: int = 1) -> Dict[str, Any]:
    """Extract all features needed for prediction."""
    title = title or ""
    body = body or ""
    
    features = {
        'title_length': len(title),
        'body_length': len(body),
        'num_labels': num_labels,
        'is_closed': 1 if is_closed else 0,
        'title_word_count': len(title.split()),
        'body_word_count': len(body.split()),
        'has_body': 1 if len(body) > 0 else 0,
        'title_has_brackets': 1 if re.search(r'\[.*\]', title) else 0,
        'body_has_code': 1 if '```' in body else 0,
        'body_has_links': 1 if re.search(r'https?://', body) else 0,
        'author_pr_count': author_pr_count,
        'body_length_log': np.log1p(len(body)),
    }
    
    # Time-based features
    if created_at:
        features['hour_sin'] = np.sin(2 * np.pi * created_at.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * created_at.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * created_at.weekday() / 7)
        features['day_cos'] = np.cos(2 * np.pi * created_at.weekday() / 7)
    else:
        # Use current time as default
        now = datetime.now()
        features['hour_sin'] = np.sin(2 * np.pi * now.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * now.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * now.weekday() / 7)
        features['day_cos'] = np.cos(2 * np.pi * now.weekday() / 7)
    
    return features

def prepare_feature_vector(features: Dict[str, Any]) -> List[float]:
    """Convert features dict to ordered list matching model's expected input."""
    return [features.get(col, 0) for col in feature_columns]

# ─── request/response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    owner:     str = Field(..., example="facebook")
    repo:      str = Field(..., example="react")
    pr_number: int = Field(..., example=32812)

class PredictResponse(BaseModel):
    predicted_hours: float
    title:           str
    body:            Optional[str]
    created_at:      datetime
    merged_at:       datetime
    actual_hours:    float
    features_used:   Optional[Dict[str, Any]] = None

class EstimateRequest(BaseModel):
    title:      str
    body:       Optional[str] = None
    num_labels: int
    is_closed:  bool

class EstimateResponse(BaseModel):
    predicted_hours: float
    features_used:   Optional[Dict[str, Any]] = None

class ModelMetricsResponse(BaseModel):
    best_model:      str
    training_date:   str
    num_samples:     int
    num_features:    int
    features:        List[str]
    test_rmse:       float
    test_mae:        float
    test_r2:         float
    cv_rmse:         float
    feature_importance: Dict[str, float]
    data_stats:      Dict[str, float]

class ModelComparisonResponse(BaseModel):
    comparison_date: str
    best_model:      str
    models:          List[Dict[str, Any]]

# ─── health check ─────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def read_root():
    return {
        "message": "✅ PR Time Estimator API is running",
        "version": "1.0",
        "model": model_metrics.get("best_model", "Unknown"),
        "endpoints": {
            "predict":  {"method": "POST", "path": "/predict", "description": "Predict merge time for existing GitHub PR"},
            "estimate": {"method": "POST", "path": "/estimate", "description": "Estimate merge time for manual PR data"},
            "metrics":  {"method": "GET", "path": "/metrics", "description": "Get model performance metrics"},
            "comparison": {"method": "GET", "path": "/comparison", "description": "Get model comparison results"}
        },
    }

# ─── model metrics endpoint ──────────────────────────────────────────────────
@app.get("/metrics", response_model=ModelMetricsResponse, tags=["metrics"])
def get_metrics():
    """Return model performance metrics."""
    if not model_metrics:
        raise HTTPException(status_code=404, detail="Model metrics not available")
    
    metrics = model_metrics.get("metrics", {})
    return ModelMetricsResponse(
        best_model=model_metrics.get("best_model", "Unknown"),
        training_date=model_metrics.get("training_date", ""),
        num_samples=model_metrics.get("num_samples", 0),
        num_features=model_metrics.get("num_features", 0),
        features=model_metrics.get("features", []),
        test_rmse=metrics.get("test_rmse", 0),
        test_mae=metrics.get("test_mae", 0),
        test_r2=metrics.get("test_r2", 0),
        cv_rmse=metrics.get("cv_rmse", 0),
        feature_importance=metrics.get("feature_importance", {}),
        data_stats=model_metrics.get("data_stats", {}),
    )

# ─── model comparison endpoint ───────────────────────────────────────────────
@app.get("/comparison", response_model=ModelComparisonResponse, tags=["metrics"])
def get_comparison():
    """Return model comparison results."""
    if not model_comparison:
        raise HTTPException(status_code=404, detail="Model comparison not available")
    
    return ModelComparisonResponse(
        comparison_date=model_comparison.get("comparison_date", ""),
        best_model=model_comparison.get("best_model", "Unknown"),
        models=model_comparison.get("models", []),
    )

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
    merged_str = pr.get("merged_at")
    if not merged_str:
        raise HTTPException(status_code=400, detail="PR has not been merged yet")
    merged = isoparse(merged_str)
    actual_hours = (merged - created).total_seconds() / 3600

    # Extract features
    features = extract_features(
        title=pr.get("title", ""),
        body=pr.get("body", ""),
        num_labels=len(pr.get("labels", [])),
        is_closed=pr.get("state", "").lower() == "closed",
        created_at=created,
        author_pr_count=1  # Could be enhanced with GitHub API lookup
    )
    
    # Prepare feature vector and predict
    feature_vector = prepare_feature_vector(features)
    pred = model.predict([feature_vector])[0]

    return PredictResponse(
        predicted_hours=float(max(0, pred)),  # Ensure non-negative
        title=pr.get("title", ""),
        body=pr.get("body"),
        created_at=created,
        merged_at=merged,
        actual_hours=actual_hours,
        features_used=features,
    )

# ─── estimate endpoint (manual PR data) ──────────────────────────────────────
@app.post("/estimate", response_model=EstimateResponse, tags=["prediction"])
def estimate(req: EstimateRequest):
    # Extract features
    features = extract_features(
        title=req.title,
        body=req.body,
        num_labels=req.num_labels,
        is_closed=req.is_closed,
    )
    
    # Prepare feature vector and predict
    feature_vector = prepare_feature_vector(features)
    pred = model.predict([feature_vector])[0]
    
    return EstimateResponse(
        predicted_hours=float(max(0, pred)),  # Ensure non-negative
        features_used=features,
    )

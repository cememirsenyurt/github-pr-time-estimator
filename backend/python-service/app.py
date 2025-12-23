# backend/python-service/app.py
"""
GitHub PR Time-to-Merge API

Advanced ML-powered prediction of GitHub PR merge times using:
- XGBoost with log-transformed targets
- Semantic text embeddings (sentence-transformers)
- Label meaning extraction (urgency vs complexity)
- Code complexity features
- Prediction confidence intervals
"""

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
    description="Advanced ML-powered prediction of GitHub PR merge times using XGBoost, semantic embeddings, and comprehensive feature engineering",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Label classification constants ──────────────────────────────────────────
# Note: These constants are duplicated from ml/constants.py for deployment simplicity.
# If you update these, also update the shared constants file.
URGENCY_LABELS = {
    'bug', 'bugfix', 'hotfix', 'fix', 'critical', 'urgent', 'p0', 'p1', 
    'security', 'patch', 'regression', 'blocker', 'crash', 'error'
}
COMPLEXITY_LABELS = {
    'feature', 'enhancement', 'refactor', 'breaking-change', 'major',
    'architecture', 'rfc', 'needs-discussion', 'wip', 'draft', 'large',
    'complex', 'review-needed', 'documentation'
}
FAST_MERGE_LABELS = {
    'trivial', 'typo', 'docs', 'documentation', 'chore', 'cleanup',
    'minor', 'small', 'quick-fix', 'dependencies', 'dep', 'deps'
}
CORE_AUTHOR_TYPES = {'OWNER', 'MEMBER', 'COLLABORATOR'}

# ─── load model and metrics ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pr_time_model.joblib")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "model", "model_metrics.json")
COMPARISON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "model", "model_comparison.json")

model_artifact = joblib.load(MODEL_PATH)
model = model_artifact['model']
feature_columns = model_artifact['feature_columns']
use_log_transform = model_artifact.get('use_log_transform', False)
quantile_models = model_artifact.get('quantile_models', None)
embedding_model_name = model_artifact.get('embedding_model_name', None)

# Load embedding model if available
embedding_model = None
if embedding_model_name:
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(embedding_model_name)
        print(f"[INFO] Loaded embedding model: {embedding_model_name}")
    except Exception as e:
        print(f"[WARN] Could not load embedding model: {e}")

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
def extract_label_features(labels: List[str]) -> Dict[str, float]:
    """Extract semantic features from label names."""
    if not labels:
        return {
            'label_urgency_score': 0,
            'label_complexity_score': 0,
            'label_fast_merge_score': 0,
            'has_urgency_label': 0,
            'has_complexity_label': 0,
            'has_fast_merge_label': 0,
        }
    
    label_names_lower = [str(l).lower() for l in labels]
    
    urgency_count = sum(1 for l in label_names_lower if any(u in l for u in URGENCY_LABELS))
    complexity_count = sum(1 for l in label_names_lower if any(c in l for c in COMPLEXITY_LABELS))
    fast_count = sum(1 for l in label_names_lower if any(f in l for f in FAST_MERGE_LABELS))
    
    return {
        'label_urgency_score': urgency_count / max(len(labels), 1),
        'label_complexity_score': complexity_count / max(len(labels), 1),
        'label_fast_merge_score': fast_count / max(len(labels), 1),
        'has_urgency_label': 1 if urgency_count > 0 else 0,
        'has_complexity_label': 1 if complexity_count > 0 else 0,
        'has_fast_merge_label': 1 if fast_count > 0 else 0,
    }


def get_text_for_embedding(title: str, body: str) -> str:
    """Combine title and body for embedding."""
    title = title or ""
    body = body or ""
    
    # Truncate body to avoid very long sequences
    max_body_len = 500
    if len(body) > max_body_len:
        body = body[:max_body_len] + "..."
    
    return f"{title}. {title}. {body}".strip()


def extract_features(
    title: str, 
    body: str, 
    labels: List[str],
    is_closed: bool,
    created_at: Optional[datetime] = None, 
    author_pr_count: int = 1,
    author_association: str = "NONE",
    additions: int = 0,
    deletions: int = 0,
    changed_files: int = 1,
    commits: int = 1,
) -> Dict[str, Any]:
    """Extract all features needed for prediction."""
    title = title or ""
    body = body or ""
    
    # Basic text features
    features = {
        'title_length': len(title),
        'body_length': len(body),
        'title_word_count': len(title.split()),
        'body_word_count': len(body.split()),
        'body_length_log': np.log1p(len(body)),
        'has_body': 1 if len(body) > 0 else 0,
    }
    
    # Label features
    features['num_labels'] = len(labels)
    label_features = extract_label_features(labels)
    features.update(label_features)
    
    # Code complexity features
    total_changes = additions + deletions
    features['total_changes_log'] = np.log1p(total_changes)
    features['additions_log'] = np.log1p(additions)
    features['deletions_log'] = np.log1p(deletions)
    features['changed_files_log'] = np.log1p(changed_files)
    features['change_density'] = total_changes / max(changed_files, 1)
    features['additions_ratio'] = additions / max(total_changes, 1)
    features['commits'] = commits
    
    # Binary features
    features['is_closed'] = 1 if is_closed else 0
    features['title_has_brackets'] = 1 if re.search(r'\[.*\]', title) else 0
    features['body_has_code'] = 1 if '```' in body else 0
    features['body_has_links'] = 1 if re.search(r'https?://', body) else 0
    
    # Title type classification
    title_lower = title.lower()
    features['title_is_fix'] = 1 if re.search(r'\bfix\b|\bbug\b|\bhotfix\b', title_lower) else 0
    features['title_is_feature'] = 1 if re.search(r'\bfeat\b|\bfeature\b|\badd\b|\bnew\b', title_lower) else 0
    features['title_is_refactor'] = 1 if re.search(r'\brefactor\b|\bclean\b|\bimprove\b', title_lower) else 0
    features['title_is_docs'] = 1 if re.search(r'\bdoc\b|\bdocs\b|\breadme\b', title_lower) else 0
    features['title_is_test'] = 1 if re.search(r'\btest\b|\bspec\b', title_lower) else 0
    
    # Author features
    features['author_pr_count_log'] = np.log1p(author_pr_count)
    features['is_core_contributor'] = 1 if author_association in CORE_AUTHOR_TYPES else 0
    
    # Time-based features
    if created_at:
        features['hour_sin'] = np.sin(2 * np.pi * created_at.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * created_at.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * created_at.weekday() / 7)
        features['day_cos'] = np.cos(2 * np.pi * created_at.weekday() / 7)
        features['is_weekend'] = 1 if created_at.weekday() >= 5 else 0
    else:
        now = datetime.now()
        features['hour_sin'] = np.sin(2 * np.pi * now.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * now.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * now.weekday() / 7)
        features['day_cos'] = np.cos(2 * np.pi * now.weekday() / 7)
        features['is_weekend'] = 1 if now.weekday() >= 5 else 0
    
    return features


def add_embedding_features(features: Dict[str, Any], title: str, body: str) -> Dict[str, Any]:
    """Add embedding features if model is available."""
    if embedding_model is None:
        return features
    
    try:
        text = get_text_for_embedding(title, body)
        embedding = embedding_model.encode([text], convert_to_numpy=True)[0]
        
        for i, val in enumerate(embedding):
            features[f'emb_{i}'] = float(val)
    except Exception as e:
        print(f"[WARN] Could not generate embedding: {e}")
    
    return features


def prepare_feature_vector(features: Dict[str, Any]) -> List[float]:
    """Convert features dict to ordered list matching model's expected input."""
    return [features.get(col, 0) for col in feature_columns]


def predict_with_confidence(feature_vector: List[float]) -> Dict[str, float]:
    """
    Make prediction with confidence intervals.
    Returns predicted value and confidence bounds.
    """
    # Main prediction
    pred_log = model.predict([feature_vector])[0]
    
    result = {
        'predicted_log': float(pred_log),
        'predicted_hours': float(np.expm1(pred_log)) if use_log_transform else float(pred_log),
    }
    
    # Quantile predictions for confidence intervals
    if quantile_models:
        try:
            q10 = quantile_models['q10'].predict([feature_vector])[0]
            q50 = quantile_models['q50'].predict([feature_vector])[0]
            q90 = quantile_models['q90'].predict([feature_vector])[0]
            
            if use_log_transform:
                lower = float(np.expm1(q10))
                median = float(np.expm1(q50))
                upper = float(np.expm1(q90))
            else:
                lower = float(q10)
                median = float(q50)
                upper = float(q90)
            
            # Ensure logical ordering of confidence bounds
            result['lower_bound_hours'] = min(lower, median, upper)
            result['median_hours'] = sorted([lower, median, upper])[1]
            result['upper_bound_hours'] = max(lower, median, upper)
        except Exception as e:
            print(f"[WARN] Could not compute quantiles: {e}")
    
    return result


# ─── request/response schemas ─────────────────────────────────────────────────
class PredictRequest(BaseModel):
    owner:     str = Field(..., example="facebook")
    repo:      str = Field(..., example="react")
    pr_number: int = Field(..., example=32812)

class PredictionConfidence(BaseModel):
    lower_bound_hours: Optional[float] = None
    median_hours: Optional[float] = None
    upper_bound_hours: Optional[float] = None

class PredictResponse(BaseModel):
    predicted_hours: float
    confidence: Optional[PredictionConfidence] = None
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
    labels:     Optional[List[str]] = None
    additions:  Optional[int] = 0
    deletions:  Optional[int] = 0
    changed_files: Optional[int] = 1

class EstimateResponse(BaseModel):
    predicted_hours: float
    confidence: Optional[PredictionConfidence] = None
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
    uses_embeddings: Optional[bool] = None
    uses_log_transform: Optional[bool] = None

class ModelComparisonResponse(BaseModel):
    comparison_date: str
    best_model:      str
    models:          List[Dict[str, Any]]

# ─── health check ─────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def read_root():
    return {
        "message": "✅ PR Time Estimator API is running",
        "version": "2.0",
        "model": model_metrics.get("best_model", "Unknown"),
        "uses_embeddings": embedding_model is not None,
        "uses_log_transform": use_log_transform,
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
        uses_embeddings=model_metrics.get("uses_embeddings", False),
        uses_log_transform=model_metrics.get("uses_log_transform", False),
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

    # Extract label names
    labels = [label.get("name", "") for label in pr.get("labels", [])]
    
    # Extract features with code complexity
    features = extract_features(
        title=pr.get("title", ""),
        body=pr.get("body", ""),
        labels=labels,
        is_closed=pr.get("state", "").lower() == "closed",
        created_at=created,
        author_pr_count=1,  # Could be enhanced with GitHub API lookup
        author_association=pr.get("author_association", "NONE"),
        additions=pr.get("additions", 0),
        deletions=pr.get("deletions", 0),
        changed_files=pr.get("changed_files", 1),
        commits=pr.get("commits", 1),
    )
    
    # Add embedding features if available
    features = add_embedding_features(features, pr.get("title", ""), pr.get("body", ""))
    
    # Prepare feature vector and predict
    feature_vector = prepare_feature_vector(features)
    prediction = predict_with_confidence(feature_vector)
    
    # Build confidence response
    confidence = None
    if 'lower_bound_hours' in prediction:
        confidence = PredictionConfidence(
            lower_bound_hours=max(0, prediction.get('lower_bound_hours', 0)),
            median_hours=max(0, prediction.get('median_hours', 0)),
            upper_bound_hours=max(0, prediction.get('upper_bound_hours', 0)),
        )

    return PredictResponse(
        predicted_hours=float(max(0, prediction['predicted_hours'])),
        confidence=confidence,
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
    # Build label list from provided labels or use empty list
    labels = req.labels if req.labels else []
    
    # Extract features
    features = extract_features(
        title=req.title,
        body=req.body,
        labels=labels,
        is_closed=req.is_closed,
        additions=req.additions or 0,
        deletions=req.deletions or 0,
        changed_files=req.changed_files or 1,
    )
    
    # Add embedding features if available
    features = add_embedding_features(features, req.title, req.body or "")
    
    # Prepare feature vector and predict
    feature_vector = prepare_feature_vector(features)
    prediction = predict_with_confidence(feature_vector)
    
    # Build confidence response
    confidence = None
    if 'lower_bound_hours' in prediction:
        confidence = PredictionConfidence(
            lower_bound_hours=max(0, prediction.get('lower_bound_hours', 0)),
            median_hours=max(0, prediction.get('median_hours', 0)),
            upper_bound_hours=max(0, prediction.get('upper_bound_hours', 0)),
        )
    
    return EstimateResponse(
        predicted_hours=float(max(0, prediction['predicted_hours'])),
        confidence=confidence,
        features_used=features,
    )

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import joblib, requests

# ─── point at the repo‐root .env ───────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(ROOT, ".env"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in .env")

app = FastAPI(
    title="GitHub PR Time-to-Merge Estimator",
    description="Given a GitHub repo & PR number, predicts how many hours until merge",
    version="0.1",
)

# ─── allow both localhost and 127.0.0.1 dev UIs ─────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
      "http://localhost:5173",    # Vite default
      "http://127.0.0.1:5173",    # in case you browse via IP
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pr_time_model.joblib")
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    owner: str = Field(..., example="facebook")
    repo: str = Field(..., example="react")
    pr_number: int = Field(..., example=32812)

class PredictResponse(BaseModel):
    predicted_hours: float

@app.get("/", tags=["health"])
def read_root():
    return {
        "message": "✅ API up",
        "endpoints": {
            "predict": {"method": "POST", "path": "/predict"}
        },
    }

@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    # 1) get the PR payload from GitHub
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

    # 2) extract the four numeric features
    tl = len(pr.get("title") or "")
    bl = len(pr.get("body")  or "")
    nl = len(pr.get("labels") or [])
    ic = 1 if (pr.get("state") or "").lower() == "closed" else 0

    # 3) predict
    try:
        hours = model.predict([[tl, bl, nl, ic]])[0]
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")

    return PredictResponse(predicted_hours=float(hours))

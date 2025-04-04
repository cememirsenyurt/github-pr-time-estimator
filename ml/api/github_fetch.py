# ml/api/github_fetch.py

import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables from .env
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is not set in the .env file")

def fetch_pr_data(owner, repo, state="closed", per_page=300):
    base_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}"
    }
    params = {
        "state": state,
        "per_page": per_page
    }

    response = requests.get(base_url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    pr_data = fetch_pr_data("facebook", "react")
    with open("/Users/cememirsenyurt/github-pr-time-estimator/ml/data/github_prs_raw.json", "w") as f:
        json.dump(pr_data, f, indent=2)
    print(f"Fetched {len(pr_data)} PRs")
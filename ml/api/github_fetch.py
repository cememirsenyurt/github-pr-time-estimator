# ml/api/github_fetch.py
"""
Enhanced GitHub PR Data Fetcher

Fetches rich PR data including:
- Basic PR info (title, body, labels)
- Code change statistics (additions, deletions, changed files)
- Author information and history
- Commit count and review information
"""

import os
import time
from dotenv import load_dotenv
import requests
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Load environment variables from .env
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN is not set in the .env file")

# Headers for GitHub API
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}"
}


def fetch_pr_list(owner: str, repo: str, state: str = "closed", per_page: int = 100, page: int = 1) -> List[Dict]:
    """Fetch a list of PRs from a repository."""
    base_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    params = {
        "state": state,
        "per_page": per_page,
        "page": page,
        "sort": "updated",
        "direction": "desc"
    }
    
    response = requests.get(base_url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()


def fetch_pr_details(owner: str, repo: str, pr_number: int) -> Optional[Dict]:
    """
    Fetch detailed information for a single PR including:
    - additions/deletions (lines of code)
    - changed_files count
    - commits count
    - review comments count
    - mergeable state
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def fetch_pr_commits(owner: str, repo: str, pr_number: int) -> List[Dict]:
    """Fetch commits for a PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits"
    params = {"per_page": 100}
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return response.json()


def fetch_pr_reviews(owner: str, repo: str, pr_number: int) -> List[Dict]:
    """Fetch reviews for a PR."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    params = {"per_page": 100}
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return response.json()


def fetch_user_info(username: str) -> Optional[Dict]:
    """Fetch user profile information."""
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def fetch_author_pr_history(owner: str, repo: str, author: str, max_prs: int = 100) -> Dict[str, Any]:
    """
    Fetch an author's PR history in a repository.
    Returns stats about their previous PRs.
    """
    url = "https://api.github.com/search/issues"
    query = f"repo:{owner}/{repo} type:pr author:{author} is:merged"
    params = {
        "q": query,
        "per_page": min(max_prs, 100),
        "sort": "created",
        "order": "desc"
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        return {"total_count": 0, "items": []}
    
    return response.json()


def enrich_pr_data(pr_basic: Dict, owner: str, repo: str, include_reviews: bool = False) -> Dict[str, Any]:
    """
    Enrich basic PR data with additional details.
    
    Returns a dictionary with:
    - Basic PR info
    - Code change statistics (additions, deletions, changed_files)
    - Commit count
    - Author statistics
    - Label names (extracted from label objects)
    """
    pr_number = pr_basic.get("number")
    
    # Fetch detailed PR info
    pr_details = fetch_pr_details(owner, repo, pr_number)
    
    if not pr_details:
        return pr_basic
    
    # Extract label names
    labels = pr_details.get("labels", [])
    label_names = [label.get("name", "") for label in labels]
    
    # Get author info
    author = pr_details.get("user", {}).get("login", "")
    author_type = pr_details.get("author_association", "NONE")
    
    # Build enriched data
    enriched = {
        "number": pr_number,
        "state": pr_details.get("state", ""),
        "title": pr_details.get("title", ""),
        "body": pr_details.get("body", ""),
        "created_at": pr_details.get("created_at", ""),
        "updated_at": pr_details.get("updated_at", ""),
        "closed_at": pr_details.get("closed_at", ""),
        "merged_at": pr_details.get("merged_at", ""),
        "labels": labels,
        "label_names": label_names,
        "user.login": author,
        "author_association": author_type,
        # Code change statistics
        "additions": pr_details.get("additions", 0),
        "deletions": pr_details.get("deletions", 0),
        "changed_files": pr_details.get("changed_files", 0),
        "commits": pr_details.get("commits", 0),
        "review_comments": pr_details.get("review_comments", 0),
        "comments": pr_details.get("comments", 0),
        # Derived metrics
        "total_changes": pr_details.get("additions", 0) + pr_details.get("deletions", 0),
    }
    
    return enriched


def fetch_rich_pr_data(owner: str, repo: str, state: str = "closed", 
                       max_prs: int = 300, include_reviews: bool = False,
                       rate_limit_delay: float = 0.5) -> List[Dict]:
    """
    Fetch rich PR data from a repository with all available details.
    
    Args:
        owner: Repository owner
        repo: Repository name
        state: PR state filter ('closed', 'open', 'all')
        max_prs: Maximum number of PRs to fetch
        include_reviews: Whether to include review data (slower)
        rate_limit_delay: Delay between API calls to avoid rate limiting
    
    Returns:
        List of enriched PR dictionaries
    """
    all_prs = []
    page = 1
    per_page = 100
    
    print(f"[INFO] Fetching PRs from {owner}/{repo}...")
    
    while len(all_prs) < max_prs:
        pr_list = fetch_pr_list(owner, repo, state, per_page, page)
        
        if not pr_list:
            break
            
        for pr in pr_list:
            if len(all_prs) >= max_prs:
                break
                
            # Only include merged PRs for training data
            if state == "closed" and not pr.get("merged_at"):
                continue
                
            enriched = enrich_pr_data(pr, owner, repo, include_reviews)
            all_prs.append(enriched)
            
            # Rate limiting
            time.sleep(rate_limit_delay)
            
            if len(all_prs) % 10 == 0:
                print(f"[INFO] Fetched {len(all_prs)} PRs...")
        
        page += 1
        
        # Check if we got fewer results than requested (last page)
        if len(pr_list) < per_page:
            break
    
    print(f"[INFO] Total PRs fetched: {len(all_prs)}")
    return all_prs


def fetch_from_multiple_repos(repos: List[tuple], max_per_repo: int = 500,
                              state: str = "closed") -> List[Dict]:
    """
    Fetch PR data from multiple repositories.
    
    Args:
        repos: List of (owner, repo) tuples
        max_per_repo: Maximum PRs to fetch per repository
        state: PR state filter
    
    Returns:
        Combined list of PRs from all repositories
    """
    all_prs = []
    
    for owner, repo in repos:
        try:
            prs = fetch_rich_pr_data(owner, repo, state, max_per_repo)
            # Add repo info to each PR
            for pr in prs:
                pr["source_repo"] = f"{owner}/{repo}"
            all_prs.extend(prs)
            print(f"[INFO] Fetched {len(prs)} PRs from {owner}/{repo}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch from {owner}/{repo}: {e}")
            continue
    
    return all_prs


# Legacy function for backward compatibility
def fetch_pr_data(owner: str, repo: str, state: str = "closed", per_page: int = 300) -> List[Dict]:
    """Legacy function - redirects to fetch_rich_pr_data."""
    return fetch_rich_pr_data(owner, repo, state, per_page)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch GitHub PR data")
    parser.add_argument("--owner", default="facebook", help="Repository owner")
    parser.add_argument("--repo", default="react", help="Repository name")
    parser.add_argument("--max-prs", type=int, default=300, help="Maximum PRs to fetch")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()
    
    pr_data = fetch_rich_pr_data(args.owner, args.repo, max_prs=args.max_prs)
    
    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "data", "github_prs_raw.json"
    )
    
    with open(output_path, "w") as f:
        json.dump(pr_data, f, indent=2)
    
    print(f"[INFO] Saved {len(pr_data)} PRs to {output_path}")
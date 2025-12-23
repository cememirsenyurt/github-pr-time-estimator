# ml/constants.py
"""
Shared constants for PR Time Estimator ML Model.

These constants define label classification patterns used to extract
semantic meaning from PR labels.
"""

# Label classification for urgency signals
# These labels indicate PRs that should be reviewed/merged quickly
URGENCY_LABELS = {
    'bug', 'bugfix', 'hotfix', 'fix', 'critical', 'urgent', 'p0', 'p1', 
    'security', 'patch', 'regression', 'blocker', 'crash', 'error'
}

# Label classification for complexity signals
# These labels indicate PRs that may take longer to review/merge
COMPLEXITY_LABELS = {
    'feature', 'enhancement', 'refactor', 'breaking-change', 'major',
    'architecture', 'rfc', 'needs-discussion', 'wip', 'draft', 'large',
    'complex', 'review-needed', 'documentation'
}

# Label classification for fast merge signals
# These labels indicate PRs that should merge very quickly
FAST_MERGE_LABELS = {
    'trivial', 'typo', 'docs', 'documentation', 'chore', 'cleanup',
    'minor', 'small', 'quick-fix', 'dependencies', 'dep', 'deps'
}

# Author association types from GitHub API that indicate core contributors
CORE_AUTHOR_TYPES = {'OWNER', 'MEMBER', 'COLLABORATOR'}

# Default embedding model name (lightweight for fast inference)
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

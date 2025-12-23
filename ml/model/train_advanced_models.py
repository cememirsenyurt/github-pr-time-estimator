# ml/model/train_advanced_models.py
"""
Advanced ML Model Training for PR Time Estimation

This script trains and compares multiple ML models to find the best predictor
for GitHub PR merge time. It implements:
- Semantic text embeddings using sentence-transformers
- Label meaning extraction (urgency vs complexity signals)
- Code complexity features (lines added/deleted, files changed)
- Author context features
- Log-transformed target variable
- Prediction confidence intervals via quantile regression
- Multiple model comparison (Random Forest, XGBoost, Gradient Boosting, Ridge, Elastic Net)
- Cross-validation with proper metrics
- Model persistence and metrics export
"""

import os
import json
import ast
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Try to import sentence-transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("[WARN] sentence-transformers not available. Semantic features disabled.")

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed_pr_data.csv'))
MODEL_PATH = os.path.join(BASE_DIR, 'pr_time_model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'model_metrics.json')
COMPARISON_PATH = os.path.join(BASE_DIR, 'model_comparison.json')

# Label classification for urgency/complexity signals
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

# Author association types (from GitHub API)
CORE_AUTHOR_TYPES = {'OWNER', 'MEMBER', 'COLLABORATOR'}

# Embedding model (lightweight for fast inference)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'


def load_data(path=DATA_PATH):
    """Load and return the processed PR data."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def parse_labels(labels_str):
    """Parse labels from string representation."""
    if pd.isna(labels_str) or labels_str == '':
        return []
    
    if isinstance(labels_str, list):
        return labels_str
    
    if isinstance(labels_str, str):
        if labels_str.startswith('['):
            try:
                labels_list = ast.literal_eval(labels_str)
                # Extract label names if list of dicts
                if labels_list and isinstance(labels_list[0], dict):
                    return [l.get('name', '') for l in labels_list]
                return labels_list
            except (ValueError, SyntaxError):
                return []
        return [labels_str]
    
    return []


def extract_label_features(labels):
    """
    Extract semantic features from labels.
    Returns dict with urgency, complexity, and fast_merge scores.
    """
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
    """Combine title and body for embedding, with title weighted more."""
    title = str(title) if pd.notna(title) else ""
    body = str(body) if pd.notna(body) else ""
    
    # Truncate body to avoid very long sequences
    max_body_len = 500
    if len(body) > max_body_len:
        body = body[:max_body_len] + "..."
    
    # Combine with title repeated for emphasis
    return f"{title}. {title}. {body}".strip()


def generate_embeddings(texts, model_name=EMBEDDING_MODEL_NAME, batch_size=32):
    """
    Generate embeddings for a list of texts using sentence-transformers.
    Returns numpy array of embeddings or None if not available.
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    
    try:
        print(f"[INFO] Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        
        print(f"[INFO] Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"[INFO] Embedding shape: {embeddings.shape}")
        return embeddings, model
    except Exception as e:
        print(f"[WARN] Failed to generate embeddings: {e}")
        return None, None


def engineer_features(df: pd.DataFrame, generate_text_embeddings: bool = True) -> pd.DataFrame:
    """
    Create engineered features from raw PR data.
    
    Features created:
    - Text features: title_length, body_length, word counts
    - Label features: count, urgency score, complexity score
    - Code complexity: additions, deletions, changed_files, commits
    - Author features: pr_count, is_core_contributor
    - Time features: hour, day of week (cyclic encoded)
    - Semantic embeddings (optional): title+body embeddings
    """
    df = df.copy()
    
    # Basic text features
    df['title_length'] = df['title'].fillna('').str.len()
    df['body_length'] = df['body'].fillna('').str.len()
    df['title_word_count'] = df['title'].fillna('').str.split().str.len().fillna(0)
    df['body_word_count'] = df['body'].fillna('').str.split().str.len().fillna(0)
    
    # Parse labels and extract features
    df['parsed_labels'] = df['labels'].apply(parse_labels)
    df['num_labels'] = df['parsed_labels'].apply(len)
    
    # Label semantic features
    label_features = df['parsed_labels'].apply(extract_label_features)
    label_df = pd.DataFrame(label_features.tolist())
    for col in label_df.columns:
        df[col] = label_df[col]
    
    # Code complexity features (if available)
    if 'additions' in df.columns:
        df['additions'] = pd.to_numeric(df['additions'], errors='coerce').fillna(0)
    else:
        df['additions'] = 0
        
    if 'deletions' in df.columns:
        df['deletions'] = pd.to_numeric(df['deletions'], errors='coerce').fillna(0)
    else:
        df['deletions'] = 0
        
    if 'changed_files' in df.columns:
        df['changed_files'] = pd.to_numeric(df['changed_files'], errors='coerce').fillna(1)
    else:
        df['changed_files'] = 1
        
    if 'commits' in df.columns:
        df['commits'] = pd.to_numeric(df['commits'], errors='coerce').fillna(1)
    else:
        df['commits'] = 1
    
    # Derived code metrics
    df['total_changes'] = df['additions'] + df['deletions']
    df['change_density'] = df['total_changes'] / df['changed_files'].replace(0, 1)
    df['additions_ratio'] = df['additions'] / df['total_changes'].replace(0, 1)
    
    # Log transforms for skewed distributions
    df['total_changes_log'] = np.log1p(df['total_changes'])
    df['additions_log'] = np.log1p(df['additions'])
    df['deletions_log'] = np.log1p(df['deletions'])
    df['changed_files_log'] = np.log1p(df['changed_files'])
    df['body_length_log'] = np.log1p(df['body_length'])
    
    # Binary features
    df['is_closed'] = (df['state'].str.lower() == 'closed').astype(int)
    df['has_body'] = (df['body'].fillna('').str.len() > 0).astype(int)
    df['title_has_brackets'] = df['title'].fillna('').str.contains(r'\[.*\]', regex=True).astype(int)
    df['body_has_code'] = df['body'].fillna('').str.contains(r'```', regex=True).astype(int)
    df['body_has_links'] = df['body'].fillna('').str.contains(r'https?://', regex=True).astype(int)
    
    # Title type classification (simple heuristics)
    title_lower = df['title'].fillna('').str.lower()
    df['title_is_fix'] = title_lower.str.contains(r'\bfix\b|\bbug\b|\bhotfix\b', regex=True).astype(int)
    df['title_is_feature'] = title_lower.str.contains(r'\bfeat\b|\bfeature\b|\badd\b|\bnew\b', regex=True).astype(int)
    df['title_is_refactor'] = title_lower.str.contains(r'\brefactor\b|\bclean\b|\bimprove\b', regex=True).astype(int)
    df['title_is_docs'] = title_lower.str.contains(r'\bdoc\b|\bdocs\b|\breadme\b', regex=True).astype(int)
    df['title_is_test'] = title_lower.str.contains(r'\btest\b|\bspec\b', regex=True).astype(int)
    
    # Author features
    author_counts = df['user.login'].value_counts().to_dict()
    df['author_pr_count'] = df['user.login'].map(lambda u: author_counts.get(u, 0)).fillna(0).astype(int)
    df['author_pr_count_log'] = np.log1p(df['author_pr_count'])
    
    # Author type (if available)
    if 'author_association' in df.columns:
        df['is_core_contributor'] = df['author_association'].isin(CORE_AUTHOR_TYPES).astype(int)
    else:
        df['is_core_contributor'] = 0
    
    # Time-based features (if created_at is available)
    if 'created_at' in df.columns:
        try:
            df['created_dt'] = pd.to_datetime(df['created_at'])
            df['hour_of_day'] = df['created_dt'].dt.hour
            df['day_of_week'] = df['created_dt'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Cyclic encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        except Exception as e:
            print(f"[WARN] Could not parse datetime: {e}")
    
    return df


def get_feature_columns(df: pd.DataFrame, include_embeddings: bool = False) -> list:
    """Return list of feature columns to use for training."""
    base_features = [
        # Text features
        'title_length', 'body_length', 'title_word_count', 'body_word_count',
        'body_length_log', 'has_body',
        
        # Label features
        'num_labels', 'label_urgency_score', 'label_complexity_score', 
        'label_fast_merge_score', 'has_urgency_label', 'has_complexity_label',
        'has_fast_merge_label',
        
        # Code complexity features
        'total_changes_log', 'additions_log', 'deletions_log', 
        'changed_files_log', 'change_density', 'additions_ratio', 'commits',
        
        # Binary indicators
        'is_closed', 'title_has_brackets', 'body_has_code', 'body_has_links',
        'title_is_fix', 'title_is_feature', 'title_is_refactor', 
        'title_is_docs', 'title_is_test',
        
        # Author features
        'author_pr_count_log', 'is_core_contributor',
        
        # Time features
        'is_weekend',
    ]
    
    # Add time features if available
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    for feat in time_features:
        if feat in df.columns:
            base_features.append(feat)
    
    # Filter to columns that exist in the dataframe
    available_features = [f for f in base_features if f in df.columns]
    
    return available_features


def get_models():
    """Return dictionary of models to compare."""
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    }
    
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a single model.
    Note: Training is done on log-transformed target.
    Returns dictionary with all metrics.
    """
    print(f"\n[INFO] Training {model_name}...")
    
    # Cross-validation on log-transformed target
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_rmse_std = np.sqrt(-cv_scores).std()
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions (in log space)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics in log space
    metrics = {
        'model_name': model_name,
        'cv_rmse': float(cv_rmse),
        'cv_rmse_std': float(cv_rmse_std),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
        'train_mae': float(mean_absolute_error(y_train, train_pred)),
        'train_r2': float(r2_score(y_train, train_pred)),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
        'test_mae': float(mean_absolute_error(y_test, test_pred)),
        'test_r2': float(r2_score(y_test, test_pred)),
    }
    
    # Convert back to hours for interpretable MAPE
    y_test_hours = np.expm1(y_test)
    test_pred_hours = np.expm1(test_pred)
    
    # Avoid division by zero in MAPE
    valid_mask = y_test_hours > 0.1
    if valid_mask.sum() > 0:
        metrics['test_mape'] = float(
            mean_absolute_percentage_error(y_test_hours[valid_mask], test_pred_hours[valid_mask]) * 100
        )
    else:
        metrics['test_mape'] = float('nan')
    
    print(f"  CV RMSE (log): {metrics['cv_rmse']:.3f} (±{metrics['cv_rmse_std']:.3f})")
    print(f"  Test RMSE (log): {metrics['test_rmse']:.3f}, MAE (log): {metrics['test_mae']:.3f}, R²: {metrics['test_r2']:.3f}")
    
    return model, metrics


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from model if available."""
    importance = {}
    
    try:
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            for name, val in zip(feature_names, imp):
                importance[name] = float(val)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            for name, val in zip(feature_names, coef):
                importance[name] = float(abs(val))
    except Exception as e:
        print(f"[WARN] Could not extract feature importance for {model_name}: {e}")
    
    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance


def train_quantile_models(X_train, y_train, feature_cols):
    """
    Train quantile regression models for prediction intervals.
    Returns models for 10th, 50th (median), and 90th percentiles.
    """
    quantile_models = {}
    
    if HAS_XGB:
        for q in [0.1, 0.5, 0.9]:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='reg:quantileerror',
                quantile_alpha=q,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            quantile_models[f'q{int(q*100)}'] = model
            print(f"[INFO] Trained quantile model for {int(q*100)}th percentile")
    
    return quantile_models


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("PR Time Estimation - Advanced Model Training")
    print("=" * 60)
    
    # Load and prepare data
    df = load_data()
    df = engineer_features(df)
    
    # Filter to valid rows
    df = df.dropna(subset=['time_to_merge_hours'])
    df = df[df['time_to_merge_hours'] > 0]  # Only positive merge times
    print(f"[INFO] After filtering: {len(df)} rows")
    
    # Get features
    feature_cols = get_feature_columns(df)
    print(f"[INFO] Features ({len(feature_cols)}): {feature_cols}")
    
    X = df[feature_cols].fillna(0)
    
    # Log-transform target variable for better distribution
    y_raw = df['time_to_merge_hours']
    y = np.log1p(y_raw)
    
    print(f"[INFO] Target stats (hours): mean={y_raw.mean():.2f}, median={y_raw.median():.2f}, std={y_raw.std():.2f}")
    print(f"[INFO] Target stats (log): mean={y.mean():.2f}, median={y.median():.2f}, std={y.std():.2f}")
    
    # Generate embeddings if available
    embedding_model = None
    if HAS_SENTENCE_TRANSFORMERS:
        texts = df.apply(lambda row: get_text_for_embedding(row['title'], row['body']), axis=1).tolist()
        embeddings, embedding_model = generate_embeddings(texts)
        
        if embeddings is not None:
            # Add embedding columns to X
            embedding_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
            X_with_emb = pd.DataFrame(embeddings, columns=embedding_cols, index=X.index)
            X = pd.concat([X, X_with_emb], axis=1)
            feature_cols = feature_cols + embedding_cols
            print(f"[INFO] Added {len(embedding_cols)} embedding features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[INFO] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train and compare models
    models = get_models()
    results = []
    trained_models = {}
    
    for model_name, model in models.items():
        trained_model, metrics = evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )
        
        # Get feature importance
        metrics['feature_importance'] = get_feature_importance(
            trained_model, feature_cols, model_name
        )
        
        results.append(metrics)
        trained_models[model_name] = trained_model
    
    # Find best model (by test RMSE in log space)
    results_df = pd.DataFrame(results)
    best_idx = results_df['test_rmse'].idxmin()
    best_model_name = results_df.loc[best_idx, 'model_name']
    best_model = trained_models[best_model_name]
    best_metrics = results[best_idx]
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df[['model_name', 'cv_rmse', 'test_rmse', 'test_mae', 'test_r2']].to_string(index=False))
    print(f"\n[INFO] Best Model: {best_model_name}")
    print(f"  Test RMSE (log): {best_metrics['test_rmse']:.3f}")
    print(f"  Test MAE (log):  {best_metrics['test_mae']:.3f}")
    print(f"  Test R²:   {best_metrics['test_r2']:.3f}")
    
    # Train quantile models for prediction intervals
    quantile_models = {}
    if HAS_XGB:
        print("\n[INFO] Training quantile models for prediction intervals...")
        quantile_models = train_quantile_models(X_train, y_train, feature_cols)
    
    # Save best model and artifacts
    model_artifact = {
        'model': best_model,
        'feature_columns': feature_cols,
        'scaler': None,  # No scaling needed for tree models
        'use_log_transform': True,  # Flag to indicate log-transformed target
        'quantile_models': quantile_models if quantile_models else None,
        'embedding_model_name': EMBEDDING_MODEL_NAME if HAS_SENTENCE_TRANSFORMERS else None,
    }
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\n[INFO] Best model saved to {MODEL_PATH}")
    
    # Save metrics
    metrics_output = {
        'best_model': best_model_name,
        'training_date': datetime.now().isoformat(),
        'num_samples': len(df),
        'num_features': len(feature_cols),
        'features': feature_cols,
        'uses_embeddings': HAS_SENTENCE_TRANSFORMERS and embedding_model is not None,
        'uses_log_transform': True,
        'metrics': best_metrics,
        'data_stats': {
            'mean_merge_time_hours': float(y_raw.mean()),
            'median_merge_time_hours': float(y_raw.median()),
            'std_merge_time_hours': float(y_raw.std()),
            'min_merge_time_hours': float(y_raw.min()),
            'max_merge_time_hours': float(y_raw.max()),
        }
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"[INFO] Metrics saved to {METRICS_PATH}")
    
    # Save comparison results
    comparison_output = {
        'comparison_date': datetime.now().isoformat(),
        'models': results,
        'best_model': best_model_name,
    }
    
    with open(COMPARISON_PATH, 'w') as f:
        json.dump(comparison_output, f, indent=2)
    print(f"[INFO] Comparison saved to {COMPARISON_PATH}")
    
    return best_model, best_metrics


if __name__ == "__main__":
    main()

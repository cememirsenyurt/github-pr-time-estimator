# ml/model/train_advanced_models.py
"""
Advanced ML Model Training for PR Time Estimation

This script trains and compares multiple ML models to find the best predictor
for GitHub PR merge time. It implements:
- Feature engineering with multiple approaches
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

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed_pr_data.csv'))
MODEL_PATH = os.path.join(BASE_DIR, 'pr_time_model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'model_metrics.json')
COMPARISON_PATH = os.path.join(BASE_DIR, 'model_comparison.json')


def load_data(path=DATA_PATH):
    """Load and return the processed PR data."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw PR data.
    
    Features created:
    - title_length: Length of PR title
    - body_length: Length of PR body/description
    - num_labels: Number of labels attached
    - is_closed: Whether the PR is closed
    - title_word_count: Number of words in title
    - body_word_count: Number of words in body
    - has_body: Binary indicator if body exists
    - title_has_brackets: Contains [] (common for categorization)
    - body_has_code: Contains code blocks
    - body_has_links: Contains URLs
    - author_pr_count: Historical PR count by author
    - hour_of_day: Hour when PR was created (cyclic)
    - day_of_week: Day of week when PR was created (cyclic)
    """
    df = df.copy()
    
    # Basic text features
    df['title_length'] = df['title'].fillna('').str.len()
    df['body_length'] = df['body'].fillna('').str.len()
    df['title_word_count'] = df['title'].fillna('').str.split().str.len().fillna(0)
    df['body_word_count'] = df['body'].fillna('').str.split().str.len().fillna(0)
    
    # Label features
    df['num_labels'] = df['labels'].apply(
        lambda x: len(ast.literal_eval(x)) if isinstance(x, str) and x.startswith('[') else (len(x) if isinstance(x, list) else 0)
    )
    
    # Binary features
    df['is_closed'] = (df['state'].str.lower() == 'closed').astype(int)
    df['has_body'] = (df['body'].fillna('').str.len() > 0).astype(int)
    df['title_has_brackets'] = df['title'].fillna('').str.contains(r'\[.*\]', regex=True).astype(int)
    df['body_has_code'] = df['body'].fillna('').str.contains(r'```', regex=True).astype(int)
    df['body_has_links'] = df['body'].fillna('').str.contains(r'https?://', regex=True).astype(int)
    
    # Author history feature
    author_counts = df['user.login'].value_counts().to_dict()
    df['author_pr_count'] = df['user.login'].map(lambda u: author_counts.get(u, 0)).fillna(0).astype(int)
    
    # Time-based features (if created_at is available)
    if 'created_at' in df.columns:
        try:
            df['created_dt'] = pd.to_datetime(df['created_at'])
            df['hour_of_day'] = df['created_dt'].dt.hour
            df['day_of_week'] = df['created_dt'].dt.dayofweek
            # Cyclic encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        except Exception as e:
            print(f"[WARN] Could not parse datetime: {e}")
    
    # Log transform of body length (often long-tailed)
    df['body_length_log'] = np.log1p(df['body_length'])
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature columns to use for training."""
    base_features = [
        'title_length', 'body_length', 'num_labels', 'is_closed',
        'title_word_count', 'body_word_count', 'has_body',
        'title_has_brackets', 'body_has_code', 'body_has_links',
        'author_pr_count', 'body_length_log'
    ]
    
    # Add time features if available
    time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    for feat in time_features:
        if feat in df.columns:
            base_features.append(feat)
    
    return base_features


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
    Returns dictionary with all metrics.
    """
    print(f"\n[INFO] Training {model_name}...")
    
    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_rmse_std = np.sqrt(-cv_scores).std()
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    # Note: MAPE can be very high when actual values are close to 0 (fast PRs)
    # This is a known limitation of MAPE metric
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
        'test_mape': float(mean_absolute_percentage_error(y_test, test_pred) * 100),
    }
    
    print(f"  CV RMSE: {metrics['cv_rmse']:.2f} (±{metrics['cv_rmse_std']:.2f})")
    print(f"  Test RMSE: {metrics['test_rmse']:.2f}, MAE: {metrics['test_mae']:.2f}, R²: {metrics['test_r2']:.3f}")
    
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
    print(f"[INFO] Features: {feature_cols}")
    
    X = df[feature_cols].fillna(0)
    y = df['time_to_merge_hours']
    
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
    
    # Find best model (by test RMSE)
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
    print(f"  Test RMSE: {best_metrics['test_rmse']:.2f} hours")
    print(f"  Test MAE:  {best_metrics['test_mae']:.2f} hours")
    print(f"  Test R²:   {best_metrics['test_r2']:.3f}")
    
    # Save best model
    model_artifact = {
        'model': best_model,
        'feature_columns': feature_cols,
        'scaler': None,  # No scaling needed for tree models
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
        'metrics': best_metrics,
        'data_stats': {
            'mean_merge_time_hours': float(y.mean()),
            'median_merge_time_hours': float(y.median()),
            'std_merge_time_hours': float(y.std()),
            'min_merge_time_hours': float(y.min()),
            'max_merge_time_hours': float(y.max()),
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

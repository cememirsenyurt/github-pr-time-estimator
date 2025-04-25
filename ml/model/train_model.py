# =============================================
# train_model.py
# =============================================
# Script to load processed PR data, engineer features, train a RandomForest model
# with cross-validation, evaluate performance, and save the model.

# 1) Imports & Configuration
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed_pr_data.csv'))
MODEL_FILENAME = 'pr_time_model.joblib'
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# =============================================
# 2) Load Processed Data
# =============================================
def load_processed_data(path=DATA_PATH):
    """
    Load the pre-processed GitHub PR data from CSV.
    """
    df = pd.read_csv(path)
    print(f"[INFO] Loaded processed data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# =============================================
# 3) Feature Engineering
# =============================================
def engineer_features(df):
    """
    Create numeric features from raw columns:
      - title_length: number of characters in title
      - body_length: number of characters in body
      - num_labels: count of labels list
      - is_closed: 1 if state is 'closed', else 0
    """
    # Title & body length
    df['title_length'] = df['title'].fillna('').apply(len)
    df['body_length'] = df['body'].fillna('').apply(len)

    # Count labels (assumes labels column is a list-like string or actual list)
    def count_labels(x):
        try:
            # if already list
            return len(x) if isinstance(x, list) else len(eval(x))
        except Exception:
            return 0
    df['num_labels'] = df['labels'].apply(count_labels)

    # Closed flag
    df['is_closed'] = df['state'].fillna('').apply(lambda s: 1 if str(s).lower() == 'closed' else 0)

    return df

# =============================================
# 4) Train and Evaluate
# =============================================
def train_and_evaluate(df):
    """
    Train a Random Forest Regressor using 5-fold cross-validation
    and evaluate on a held-out test set.
    Assumes 'time_to_merge_hours' is present in df.
    """
    # Engineer numeric features
    df = engineer_features(df)

    # Define feature columns and target
    feature_cols = ['title_length', 'body_length', 'num_labels', 'is_closed']
    target_col = 'time_to_merge_hours'

    # Drop missing values in features/target
    df_model = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"[INFO] Final training dataset size: {len(df_model)} rows.")

    X = df_model[feature_cols]
    y = df_model[target_col]

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Cross-validation setup
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        rf, X_train, y_train, cv=kf, scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores.mean())
    print(f"[INFO] Cross-Validation RMSE (5-fold): {cv_rmse:.2f}")

    # Train final model
    rf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print("[INFO] Test Set Performance:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    return rf

# =============================================
# 5) Save Model
# =============================================
def save_model(model, path=MODEL_PATH):
    """
    Save the trained model to disk using joblib.
    """
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")

# =============================================
# 6) Main Execution
# =============================================
if __name__ == '__main__':
    # Load data
    df = load_processed_data()

    # Train and evaluate
    model = train_and_evaluate(df)

    # Save trained model
    save_model(model)


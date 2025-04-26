# ml/model/train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# embedding imports
from transformers import AutoTokenizer, AutoModel
import torch

# embedding setup
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embed_model.eval()

def embed_text(text: str):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        out = embed_model(**inputs)
        # mean-pool token embeddings
        return out.last_hidden_state.mean(dim=1).squeeze().numpy()

# paths
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'data', 'processed_pr_data.csv'))
MODEL_FILENAME = 'pr_time_model.joblib'
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

def load_processed_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"[INFO] Loaded processed data: {df.shape[0]} rows × {df.shape[1]} cols")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # original numeric features
    df['title_length'] = df['title'].fillna('').str.len()
    df['body_length']  = df['body'].fillna('').str.len()
    df['num_labels']   = df['labels'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['is_closed']    = (df['state'].str.lower() == 'closed').astype(int)

    # compute title embeddings once and concatenate
    titles = df['title'].fillna('').tolist()
    embs = [embed_text(t) for t in titles]
    emb_matrix = np.vstack(embs)
    dim = emb_matrix.shape[1]
    emb_cols = [f"emb_{i}" for i in range(dim)]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols, index=df.index)
    df = pd.concat([df, emb_df], axis=1)

    # author history feature
    author_counts = df['user.login'].value_counts().to_dict()
    df['author_pr_count'] = df['user.login'].map(lambda u: author_counts.get(u, 0)).fillna(0).astype(int)

    return df

def train_and_evaluate(df: pd.DataFrame):
    df = engineer_features(df).dropna(subset=['time_to_merge_hours'])
    print(f"[INFO] After cleaning: {len(df)} rows")

    base_feats = ['title_length', 'body_length', 'num_labels', 'is_closed']
    emb_feats = [c for c in df.columns if c.startswith('emb_')]
    feats = base_feats + emb_feats + ['author_pr_count']

    X = df[feats]
    y = pd.to_numeric(df['time_to_merge_hours'], errors='coerce')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = np.sqrt(-cross_val_score(rf, X_train, y_train, cv=cv,
                                       scoring='neg_mean_squared_error').mean())
    print(f"[INFO] CV RMSE: {cv_rmse:.2f}")

    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    print(f"[INFO] Test MAE: {mean_absolute_error(y_test, preds):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")

    # stash features for app.py
    global FEATURE_COLS
    FEATURE_COLS = feats
    return rf

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"[INFO] Model saved to {path}")

if __name__ == "__main__":
    df = load_processed_data()
    model = train_and_evaluate(df)
    save_model(model)
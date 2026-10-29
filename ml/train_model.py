"""
Model Training Script for Movie Recommendation System.
Trains a RandomForestRegressor on real TMDB 5000 movie data,
evaluates on an independent test split, and persists all model artifacts.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.preprocessing import load_and_merge_data, clean_data
from ml.feature_engineering import create_feature_pipeline, NUMERICAL_FEATURES


def train():
    print("========================================")
    print("CINEMATCH ML TRAINING")
    print("========================================\n")

    start_time = time.time()
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load data
    print("Loading movies dataset...")
    movies_path = 'data/tmdb_5000_movies.csv' if os.path.exists('data/tmdb_5000_movies.csv') else 'data/movies_tmdb.csv'
    credits_path = 'data/tmdb_5000_credits.csv' if os.path.exists('data/tmdb_5000_credits.csv') else 'data/credits_tmdb.csv'

    raw_movies = pd.read_csv(movies_path)
    print(f"Movies loaded: {len(raw_movies)}")

    print("\nLoading credits dataset...")
    raw_credits = pd.read_csv(credits_path)
    print(f"Credits loaded: {len(raw_credits)}")

    print("\nMerging datasets...")
    raw_df = load_and_merge_data(movies_path, credits_path)

    # 2. Clean data
    print("\nCleaning data...")
    df = clean_data(raw_df)

    # 3. Feature engineering
    print("\nFeature engineering...")
    X, y, tfidf, scaler, tfidf_matrix = create_feature_pipeline(df, max_tfidf_features=2500)

    # 4. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 5. Train Random Forest Regressor
    print("\nTraining Random Forest...\n")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print(f"Train samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # 6. Model Evaluation
    print("\nEvaluating model...\n")
    y_pred = rf.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # 7. Save model artifacts
    print("\nSaving model...")
    joblib.dump(rf, os.path.join(models_dir, 'random_forest_model.pkl'))

    print("Saving preprocessing artifacts...")
    joblib.dump(tfidf, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    features_pack = {
        'X_all': X,
        'tfidf_matrix': tfidf_matrix,
        'feature_names': NUMERICAL_FEATURES + list(tfidf.get_feature_names_out())
    }
    joblib.dump(features_pack, os.path.join(models_dir, 'movie_features.pkl'))

    save_cols = [
        'id', 'title', 'release_year', 'genres_str', 'genres_list',
        'keywords_str', 'director', 'cast_str', 'cast_list', 'overview',
        'tagline', 'runtime', 'vote_average', 'vote_count', 'popularity',
        'budget', 'revenue'
    ]
    processed_df = df[save_cols].copy()
    joblib.dump(processed_df, os.path.join(models_dir, 'processed_movies.pkl'))

    metrics_info = {
        'algorithm': 'Random Forest Regressor',
        'dataset': 'TMDB 5000 Movie Dataset',
        'total_movies': len(df),
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'features_count': int(X.shape[1]),
        'n_estimators': 100,
        'max_depth': 12
    }
    joblib.dump(metrics_info, os.path.join(models_dir, 'metrics.pkl'))

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================\n")
    return metrics_info


if __name__ == '__main__':
    train()

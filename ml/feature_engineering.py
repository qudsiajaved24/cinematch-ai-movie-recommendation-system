"""
Feature Engineering for Random Forest Regression and Content Similarity.
Handles TF-IDF vectorization on movie text tags and standard scaling on numerical metadata.
"""

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

NUMERICAL_FEATURES = ['budget', 'popularity', 'runtime', 'vote_count', 'release_year']


def create_feature_pipeline(df, max_tfidf_features=2500):
    """
    Fits TF-IDF vectorizer and StandardScaler on dataset.
    Returns:
        X: Combined sparse feature matrix (scipy csr_matrix)
        y: Target array (real TMDB vote_average)
        tfidf: Fitted TfidfVectorizer
        scaler: Fitted StandardScaler
        tfidf_matrix: Document-term matrix for cosine similarity calculations
    """
    # 1. Text feature extraction using TF-IDF
    tfidf = TfidfVectorizer(
        max_features=max_tfidf_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = tfidf.fit_transform(df['tags'])

    # 2. Numerical feature scaling
    scaler = StandardScaler()
    
    # We apply log1p to heavily skewed positive continuous distributions (budget, vote_count, popularity)
    # for numerical stability while preserving monotonicity
    num_data = df[NUMERICAL_FEATURES].copy()
    num_data['budget'] = np.log1p(np.maximum(0, num_data['budget'].values))
    num_data['vote_count'] = np.log1p(np.maximum(0, num_data['vote_count'].values))
    num_data['popularity'] = np.log1p(np.maximum(0, num_data['popularity'].values))

    scaled_num = scaler.fit_transform(num_data)

    # 3. Combine numerical features and TF-IDF features
    X = hstack([scaled_num, tfidf_matrix]).tocsr()

    # 4. Target variable: Real TMDB vote_average
    y = df['vote_average'].values.astype(float)

    return X, y, tfidf, scaler, tfidf_matrix


def transform_features(df, tfidf, scaler):
    """
    Transforms new/candidate movie records using already-fitted tfidf and scaler.
    """
    tfidf_matrix = tfidf.transform(df['tags'])
    num_data = df[NUMERICAL_FEATURES].copy()
    num_data['budget'] = np.log1p(np.maximum(0, num_data['budget'].values))
    num_data['vote_count'] = np.log1p(np.maximum(0, num_data['vote_count'].values))
    num_data['popularity'] = np.log1p(np.maximum(0, num_data['popularity'].values))
    scaled_num = scaler.transform(num_data)
    X = hstack([scaled_num, tfidf_matrix]).tocsr()
    return X, tfidf_matrix

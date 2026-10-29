"""
Model Evaluation and Metrics Inspection Script.
Loads persisted artifacts and verifies model evaluation results and feature importances.
"""

import os
import joblib
import numpy as np


def evaluate():
    metrics_path = 'models/metrics.pkl'
    model_path = 'models/random_forest_model.pkl'
    features_path = 'models/movie_features.pkl'

    if not os.path.exists(metrics_path) or not os.path.exists(model_path):
        print("Model artifacts not found. Please run 'python ml/train_model.py' first.")
        return None

    metrics = joblib.load(metrics_path)
    model = joblib.load(model_path)
    features_pack = joblib.load(features_path)

    print("=" * 50)
    print("   AI MOVIE RECOMMENDER - MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"Algorithm:         {metrics.get('algorithm')}")
    print(f"Dataset:           {metrics.get('dataset')}")
    print(f"Total Movies:      {metrics.get('total_movies'):,}")
    print(f"Training Samples:  {metrics.get('train_samples'):,} (80%)")
    print(f"Testing Samples:   {metrics.get('test_samples'):,} (20%)")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE):     {metrics.get('mae'):.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics.get('rmse'):.4f}")
    print(f"Coefficient of Determ. (R²):    {metrics.get('r2'):.4f}")
    print("-" * 50)
    print(f"Total Features Analyzed:       {metrics.get('features_count'):,}")
    print(f"Trained Timestamp:             {metrics.get('trained_at')}")

    # Top Feature Importances from Random Forest
    if hasattr(model, 'feature_importances_') and 'feature_names' in features_pack:
        importances = model.feature_importances_
        names = features_pack['feature_names']
        top_indices = np.argsort(importances)[::-1][:10]

        print("\nTop 10 Most Influential Features in Quality Rating Prediction:")
        for rank, idx in enumerate(top_indices, 1):
            feat_name = names[idx] if idx < len(names) else f"feature_{idx}"
            print(f"  {rank:2d}. {feat_name:<28} (Importance: {importances[idx]:.4f})")
    print("=" * 50)
    return metrics


if __name__ == '__main__':
    evaluate()

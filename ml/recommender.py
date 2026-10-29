"""
Hybrid Movie Recommendation Engine.
Combines Random Forest predicted quality scores, TF-IDF cosine similarity,
and normalized TMDB ratings across all 4,803 real movies.
"""

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import json

logger = logging.getLogger(__name__)

# Poster cache to avoid repeated API calls
POSTER_CACHE = {}
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # TMDB open public educational key

# Load precomputed seed poster cache if available
poster_cache_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'poster_cache.json')
if os.path.exists(poster_cache_file):
    try:
        with open(poster_cache_file, 'r') as f:
            POSTER_CACHE.update(json.load(f))
    except Exception:
        pass


def get_poster_url(movie_id, title=None):
    """
    Retrieves authentic movie poster from TMDB API with in-memory caching.
    Falls back gracefully if unavailable.
    """
    movie_id_str = str(movie_id)
    if movie_id_str in POSTER_CACHE:
        return POSTER_CACHE[movie_id_str]

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            path = data.get('poster_path')
            if path:
                full_url = f"https://image.tmdb.org/t/p/w500{path}"
                POSTER_CACHE[movie_id_str] = full_url
                return full_url
    except Exception:
        pass

    # Cache None so we don't spam failed queries
    POSTER_CACHE[movie_id_str] = None
    return None


class MovieRecommender:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.rf_model = None
        self.tfidf = None
        self.scaler = None
        self.features_pack = None
        self.processed_movies = None
        self.metrics = None
        self.X_all = None
        self.tfidf_matrix = None
        self._rf_all_predictions = None
        self.loaded = False

        self.load_models()

    def load_models(self):
        """Loads precomputed model artifacts from disk."""
        rf_path = os.path.join(self.models_dir, 'random_forest_model.pkl')
        tfidf_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        features_path = os.path.join(self.models_dir, 'movie_features.pkl')
        movies_path = os.path.join(self.models_dir, 'processed_movies.pkl')
        metrics_path = os.path.join(self.models_dir, 'metrics.pkl')

        required_files = [rf_path, tfidf_path, scaler_path, features_path, movies_path, metrics_path]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print("\n" + "!" * 50)
            print("ERROR: Model artifacts missing from 'models/' directory.")
            print("Please train the model first by running:")
            print("    python ml/train_model.py")
            print("!" * 50 + "\n")
            return False

        print("Loading movie dataset...")
        self.processed_movies = joblib.load(movies_path)

        print("Loading trained Random Forest model...")
        self.rf_model = joblib.load(rf_path)

        print("Loading TF-IDF vectorizer...")
        self.tfidf = joblib.load(tfidf_path)

        print("Loading recommendation data...")
        self.scaler = joblib.load(scaler_path)
        self.features_pack = joblib.load(features_path)
        self.metrics = joblib.load(metrics_path)

        self.X_all = self.features_pack['X_all']
        self.tfidf_matrix = self.features_pack['tfidf_matrix']

        # Precompute Random Forest predictions for all 4,803 movies once on startup
        self._rf_all_predictions = self.rf_model.predict(self.X_all)

        # NEW: genre set of every movie (used for user genre-preference matching)
        self._genre_sets = [{g.lower() for g in gl} for gl in self.processed_movies['genres_list']]

        self.loaded = True

        print("\nCineMatch is ready.")
        return True

    def get_movie_by_id(self, movie_id):
        """Fetches full movie record dictionary by TMDB movie ID."""
        if not self.loaded:
            return None
        matches = self.processed_movies[self.processed_movies['id'] == int(movie_id)]
        if matches.empty:
            return None
        movie = matches.iloc[0].to_dict()
        movie['poster_url'] = get_poster_url(movie['id'], movie.get('title'))
        return movie

    def get_movie_index(self, movie_id):
        """Finds row index in processed_movies for a given movie_id."""
        matches = self.processed_movies.index[self.processed_movies['id'] == int(movie_id)].tolist()
        return matches[0] if matches else None

    def _pref_scores(self, favorite_genres):
        """
        NEW: Genre-preference score (0 to 1) for every movie.
        Score = (number of movie genres that match user's favorite genres) / (number of movie genres).
        Returns zeros when the user has no favorite genres.
        """
        n = len(self.processed_movies)
        if not favorite_genres:
            return np.zeros(n)
        fav = {g.strip().lower() for g in favorite_genres if g and g.strip()}
        if not fav:
            return np.zeros(n)
        return np.array([len(gs & fav) / max(len(gs), 1) for gs in self._genre_sets])

    def recommend(self, movie_id, top_n=12, weight_ml=0.50, weight_content=0.35, weight_rating=0.15,
                  favorite_genres=None, weight_pref=0.0):
        """
        Generates Top N hybrid movie recommendations for a selected movie across ALL 4,803 movies.
        Steps:
            1. Find selected movie and extract its TF-IDF vector.
            2. Compute cosine similarity against all 4,803 movies.
            3. Retrieve Random Forest predicted scores for candidates.
            4. Filter candidate movies with contextual similarity (> 0.02) across all 4,803 records.
            5. Combine normalized scores:
               Final Score = (weight_ml * ML_Norm) + (weight_content * Sim_Norm)
                           + (weight_rating * Rating_Norm) + (weight_pref * Genre_Pref_Norm)
            6. Rank candidates descending and return top recommendations.
        """
        if not self.loaded:
            raise RuntimeError("Recommender models are not loaded. Train the model first.")

        target_idx = self.get_movie_index(movie_id)
        if target_idx is None:
            return None

        total_movies = len(self.processed_movies)

        # 1 & 2. Target vector & Content Cosine Similarity against all 4,803 movies
        target_vec = self.tfidf_matrix[target_idx]
        sim_scores = cosine_similarity(target_vec, self.tfidf_matrix).flatten()

        # 3. Random Forest predictions for candidate movies
        ml_scores = self._rf_all_predictions
        ratings = self.processed_movies['vote_average'].values

        # 4. Form candidate pool across the dataset (excluding target movie itself)
        # Select candidates that have meaningful contextual/genre/thematic similarity
        candidate_indices = np.where(sim_scores > 0.02)[0]
        candidate_indices = [idx for idx in candidate_indices if idx != target_idx]

        # If too few meet threshold, take top 100 similar movies
        if len(candidate_indices) < top_n:
            top_sim_indices = np.argsort(sim_scores)[::-1]
            candidate_indices = [idx for idx in top_sim_indices if idx != target_idx][:100]

        candidate_indices = np.array(candidate_indices)

        # 5. Normalization
        cand_sim = sim_scores[candidate_indices]
        max_sim = cand_sim.max() if len(cand_sim) > 0 and cand_sim.max() > 0 else 1.0
        sim_norm = cand_sim / max_sim

        cand_ml = ml_scores[candidate_indices]
        ml_norm = np.clip(cand_ml / 10.0, 0.0, 1.0)

        cand_ratings = ratings[candidate_indices]
        rating_norm = np.clip(cand_ratings / 10.0, 0.0, 1.0)

        # NEW: user genre preference score for candidates
        pref_norm = self._pref_scores(favorite_genres)[candidate_indices]

        # 6. Hybrid Score calculation
        final_scores = (
            weight_ml * ml_norm +
            weight_content * sim_norm +
            weight_rating * rating_norm +
            weight_pref * pref_norm
        )

        # 7. Rank candidate movies descending
        sorted_order = np.argsort(final_scores)[::-1][:top_n]
        top_ranked_indices = candidate_indices[sorted_order]

        recommendations = []
        for i, idx in enumerate(top_ranked_indices):
            row = self.processed_movies.iloc[idx].to_dict()
            rec_id = row['id']
            order_pos = sorted_order[i]

            rec_item = {
                'id': rec_id,
                'title': row['title'],
                'release_year': row['release_year'],
                'genres_str': row['genres_str'],
                'director': row['director'],
                'cast_str': row['cast_str'],
                'overview': row['overview'],
                'tagline': row.get('tagline', ''),
                'runtime': int(row.get('runtime', 0)),
                'vote_average': round(float(row['vote_average']), 1),
                'vote_count': int(row.get('vote_count', 0)),
                'predicted_score': round(float(cand_ml[order_pos]), 1),
                'similarity_pct': round(float(cand_sim[order_pos]) * 100, 1),
                'hybrid_score': round(float(final_scores[order_pos]), 3),
                'genre_match_pct': round(float(pref_norm[order_pos]) * 100),
                'poster_url': get_poster_url(rec_id, row['title'])
            }
            recommendations.append(rec_item)

        target_movie = self.get_movie_by_id(movie_id)
        return {
            'target_movie': target_movie,
            'recommendations': recommendations,
            'total_candidates_analyzed': total_movies,
            'weights': {
                'ml_prediction': int(weight_ml * 100),
                'content_similarity': int(weight_content * 100),
                'movie_rating': int(weight_rating * 100),
                'genre_preference': int(weight_pref * 100)
            }
        }

    def recommend_from_multiple(self, anchor_movie_ids, top_n=16, weight_ml=0.50, weight_content=0.35, weight_rating=0.15, exclude_ids=None,
                             favorite_genres=None, weight_pref=0.0):
        """
        Generates recommendations based on a collection of movies (e.g. Watchlist or Top-Rated).
        Aggregates content similarities across all provided anchor films, excludes anchors & exclude_ids,
        and scores candidates using the hybrid ML model.
        """
        if not self.loaded:
            raise RuntimeError("Recommender models are not loaded. Train the model first.")

        if not anchor_movie_ids:
            return None

        exclude_set = set(int(x) for x in anchor_movie_ids)
        if exclude_ids:
            exclude_set.update(int(x) for x in exclude_ids)

        anchor_indices = [self.get_movie_index(mid) for mid in anchor_movie_ids]
        anchor_indices = [idx for idx in anchor_indices if idx is not None]

        if not anchor_indices:
            return None

        # Compute cosine similarity for each anchor against all movies
        sim_matrix = cosine_similarity(self.tfidf_matrix[anchor_indices], self.tfidf_matrix)
        combined_sim = 0.65 * np.max(sim_matrix, axis=0) + 0.35 * np.mean(sim_matrix, axis=0)

        ml_scores = self._rf_all_predictions
        ratings = self.processed_movies['vote_average'].values
        total_movies = len(self.processed_movies)

        # Candidates: movies not in exclude_set
        candidate_indices = [idx for idx in range(total_movies) if int(self.processed_movies.iloc[idx]['id']) not in exclude_set]
        candidate_indices = np.array(candidate_indices)

        cand_sim = combined_sim[candidate_indices]
        max_sim = cand_sim.max() if len(cand_sim) > 0 and cand_sim.max() > 0 else 1.0
        sim_norm = cand_sim / max_sim

        cand_ml = ml_scores[candidate_indices]
        ml_norm = np.clip(cand_ml / 10.0, 0.0, 1.0)

        cand_ratings = ratings[candidate_indices]
        rating_norm = np.clip(cand_ratings / 10.0, 0.0, 1.0)

        # NEW: user genre preference score for candidates
        pref_norm = self._pref_scores(favorite_genres)[candidate_indices]

        final_scores = (
            weight_ml * ml_norm +
            weight_content * sim_norm +
            weight_rating * rating_norm +
            weight_pref * pref_norm
        )

        sorted_order = np.argsort(final_scores)[::-1][:top_n]
        top_ranked_indices = candidate_indices[sorted_order]

        recommendations = []
        for i, idx in enumerate(top_ranked_indices):
            row = self.processed_movies.iloc[idx].to_dict()
            rec_id = row['id']
            order_pos = sorted_order[i]

            rec_item = {
                'id': rec_id,
                'title': row['title'],
                'release_year': row['release_year'],
                'genres_str': row['genres_str'],
                'director': row['director'],
                'cast_str': row['cast_str'],
                'overview': row['overview'],
                'runtime': int(row['runtime']) if row['runtime'] and not pd.isna(row['runtime']) else None,
                'vote_average': round(float(row['vote_average']), 1),
                'vote_count': int(row['vote_count']),
                'popularity': round(float(row['popularity']), 2),
                'poster_url': get_poster_url(rec_id, row['title']),
                'predicted_score': round(float(cand_ml[order_pos]), 2),
                'similarity_pct': round(float(cand_sim[order_pos] * 100), 1),
                'hybrid_score': round(float(final_scores[order_pos]), 4),
                'genre_match_pct': round(float(pref_norm[order_pos]) * 100)
            }
            recommendations.append(rec_item)

        anchor_movies = [self.get_movie_by_id(mid) for mid in anchor_movie_ids[:6]]
        anchor_movies = [m for m in anchor_movies if m is not None]

        return {
            'anchor_movies': anchor_movies,
            'recommendations': recommendations,
            'total_candidates_analyzed': total_movies,
            'weights': {
                'ml_prediction': int(weight_ml * 100),
                'content_similarity': int(weight_content * 100),
                'movie_rating': int(weight_rating * 100),
                'genre_preference': int(weight_pref * 100)
            }
        }

    def search_movies(self, query, limit=20):
        """Searches movies across Title, Director, Cast, and Genre."""
        if not self.loaded or not query or not query.strip():
            return []

        q = query.strip().lower()
        df = self.processed_movies

        # Search matching in title, director, cast, and genres
        title_match = df['title'].str.lower().str.contains(q, regex=False, na=False)
        director_match = df['director'].str.lower().str.contains(q, regex=False, na=False)
        cast_match = df['cast_str'].str.lower().str.contains(q, regex=False, na=False)
        genre_match = df['genres_str'].str.lower().str.contains(q, regex=False, na=False)

        # Prioritize exact and title matches
        matches = df[title_match | director_match | cast_match | genre_match].copy()
        if matches.empty:
            return []

        # Sort by popularity or vote_count
        matches.sort_values(by=['vote_count', 'popularity'], ascending=[False, False], inplace=True)
        results = []
        for _, row in matches.head(limit).iterrows():
            item = row.to_dict()
            item['poster_url'] = get_poster_url(item['id'], item['title'])
            item['vote_average'] = round(float(item['vote_average']), 1)
            results.append(item)
        return results

    def explore_movies(self, genre='All', year='All', min_rating=0.0, sort_by='popularity', page=1, per_page=18):
        """
        Filters and browses the complete real dataset of 4,803 movies.
        Supports genre filtering, release decade/year filtering, rating threshold, and sorting.
        """
        if not self.loaded:
            return {'movies': [], 'total': 0, 'page': page, 'pages': 0}

        df = self.processed_movies.copy()

        if genre and genre != 'All':
            df = df[df['genres_str'].str.contains(genre, case=False, na=False)]

        if year and year != 'All':
            try:
                if '-' in year:
                    y_start, y_end = map(int, year.split('-'))
                    df = df[(df['release_year'] >= y_start) & (df['release_year'] <= y_end)]
                elif year.endswith('s'):
                    decade = int(year[:-1])
                    df = df[(df['release_year'] >= decade) & (df['release_year'] < decade + 10)]
                else:
                    y_val = int(year)
                    df = df[df['release_year'] == y_val]
            except Exception:
                pass

        if min_rating and min_rating > 0:
            df = df[df['vote_average'] >= float(min_rating)]

        if sort_by == 'rating':
            df.sort_values(by=['vote_average', 'vote_count'], ascending=[False, False], inplace=True)
        elif sort_by == 'year_desc':
            df.sort_values(by='release_year', ascending=False, inplace=True)
        elif sort_by == 'year_asc':
            df.sort_values(by='release_year', ascending=True, inplace=True)
        elif sort_by == 'title':
            df.sort_values(by='title', ascending=True, inplace=True)
        else:  # default popularity
            df.sort_values(by=['popularity', 'vote_count'], ascending=[False, False], inplace=True)

        total = len(df)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        subset = df.iloc[start_idx:end_idx]

        movie_list = []
        for _, row in subset.iterrows():
            item = row.to_dict()
            item['poster_url'] = get_poster_url(item['id'], item['title'])
            item['vote_average'] = round(float(item['vote_average']), 1)
            movie_list.append(item)

        return {
            'movies': movie_list,
            'total': total,
            'page': page,
            'pages': total_pages
        }

    def get_featured_movies(self, count=12):
        """Returns popular and acclaimed movies for home page showcase."""
        if not self.loaded:
            return []
        # Curate high vote count and popularity classics from real dataset
        popular = self.processed_movies[
            (self.processed_movies['vote_count'] > 2000) &
            (self.processed_movies['vote_average'] >= 7.2)
        ].sort_values(by='popularity', ascending=False).head(count)

        featured = []
        for _, row in popular.iterrows():
            item = row.to_dict()
            item['poster_url'] = get_poster_url(item['id'], item['title'])
            item['vote_average'] = round(float(item['vote_average']), 1)
            featured.append(item)
        return featured
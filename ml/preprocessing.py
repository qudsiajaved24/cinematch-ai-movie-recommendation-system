"""
Data Preprocessing Pipeline for TMDB 5000 Movie Dataset.
Handles loading, merging, cleaning, parsing JSON metadata,
handling missing values, and formatting text representations.
"""

import ast
import json
import logging
import os
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_json_names(val, max_items=5):
    """Safely extracts names from JSON-like strings in TMDB dataset."""
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        items = ast.literal_eval(val)
        if isinstance(items, list):
            names = []
            for item in items:
                if isinstance(item, dict) and 'name' in item:
                    names.append(item['name'])
                    if len(names) >= max_items:
                        break
            return names
        return []
    except Exception:
        return []


def parse_director(val):
    """Extracts the director's name from crew JSON string."""
    if not isinstance(val, str) or not val.strip():
        return 'Unknown'
    try:
        items = ast.literal_eval(val)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get('job') == 'Director':
                    return item.get('name', 'Unknown')
        return 'Unknown'
    except Exception:
        return 'Unknown'


def load_and_merge_data(movies_path=None, credits_path=None):
    """
    Loads raw TMDB movies and credits CSV files and merges on movie ID.
    Supports both tmdb_5000_*.csv and *_tmdb.csv naming conventions.
    Returns merged raw DataFrame.
    """
    if movies_path is None:
        if os.path.exists('data/tmdb_5000_movies.csv'):
            movies_path = 'data/tmdb_5000_movies.csv'
        elif os.path.exists('data/movies_tmdb.csv'):
            movies_path = 'data/movies_tmdb.csv'
        else:
            raise FileNotFoundError("Movies dataset not found. Checked data/tmdb_5000_movies.csv and data/movies_tmdb.csv")

    if credits_path is None:
        if os.path.exists('data/tmdb_5000_credits.csv'):
            credits_path = 'data/tmdb_5000_credits.csv'
        elif os.path.exists('data/credits_tmdb.csv'):
            credits_path = 'data/credits_tmdb.csv'
        else:
            raise FileNotFoundError("Credits dataset not found. Checked data/tmdb_5000_credits.csv and data/credits_tmdb.csv")

    logger.info(f"Loading movies from {movies_path}...")
    movies = pd.read_csv(movies_path)
    logger.info(f"Movies loaded: {len(movies)} rows.")

    logger.info(f"Loading credits from {credits_path}...")
    credits = pd.read_csv(credits_path)
    logger.info(f"Credits loaded: {len(credits)} rows.")

    # Standardize ID column for merging
    credits_clean = credits.copy()
    credits_clean.rename(columns={'movie_id': 'id'}, inplace=True)
    if 'title' in credits_clean.columns and 'title' in movies.columns:
        credits_clean.drop(columns=['title'], inplace=True)

    merged = movies.merge(credits_clean, on='id')
    logger.info(f"Merged dataset shape: {merged.shape}")
    return merged


def clean_data(df):
    """
    Cleans merged TMDB dataframe:
    - Removes duplicates
    - Parses JSON columns (genres, keywords, cast, crew)
    - Extracts director and top cast members
    - Imputes and validates numerical values (vote_average, vote_count, runtime, popularity, budget)
    - Extracts release_year
    - Constructs rich content 'tags' for NLP & TF-IDF
    """
    logger.info("Cleaning data and parsing metadata...")
    df = df.copy()

    # Step 3: Remove duplicate records based on id and title
    initial_count = len(df)
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Removed {initial_count - len(df)} duplicates. Remaining: {len(df)} movies.")

    # Step 4-8: JSON extraction & cleaning
    df['genres_list'] = df['genres'].apply(lambda x: parse_json_names(x, max_items=10))
    df['genres_str'] = df['genres_list'].apply(lambda x: ', '.join(x) if x else 'General')
    
    df['keywords_list'] = df['keywords'].apply(lambda x: parse_json_names(x, max_items=12))
    df['keywords_str'] = df['keywords_list'].apply(lambda x: ', '.join(x) if x else '')

    df['cast_list'] = df['cast'].apply(lambda x: parse_json_names(x, max_items=5))
    df['cast_str'] = df['cast_list'].apply(lambda x: ', '.join(x) if x else 'Not listed')

    df['director'] = df['crew'].apply(parse_director)

    # Clean text columns
    df['title'] = df['title'].fillna('Untitled Movie').astype(str).str.strip()
    df['overview'] = df['overview'].fillna('').astype(str).str.strip()
    df['tagline'] = df['tagline'].fillna('').astype(str).str.strip()

    # Step 5: Extract release year
    parsed_dates = pd.to_datetime(df['release_date'], errors='coerce')
    median_year = int(parsed_dates.dt.year.dropna().median()) if not parsed_dates.dt.year.dropna().empty else 2005
    df['release_year'] = parsed_dates.dt.year.fillna(median_year).astype(int)

    # Numerical imputation
    median_runtime = float(df['runtime'].dropna().median()) if not df['runtime'].dropna().empty else 105.0
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(median_runtime).astype(float)

    mean_vote = float(df['vote_average'].dropna().mean()) if not df['vote_average'].dropna().empty else 6.0
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(mean_vote).astype(float)
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0).astype(int)
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0.0).astype(float)
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce').fillna(0.0).astype(float)
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0.0).astype(float)

    # Step 9: Clean and construct content tags for ML & NLP
    # Combine genres, keywords, overview, cast, and director with weighted repetition
    # to appropriately balance entity signal with overview narrative text
    def build_rich_tags(row):
        genres_str = (' ' + ' '.join([g.replace(' ', '_').lower() for g in row['genres_list']])) * 3
        keywords_str = (' ' + ' '.join([k.replace(' ', '_').lower() for k in row['keywords_list']])) * 2
        cast_str = (' ' + ' '.join([c.replace(' ', '_').lower() for c in row['cast_list']])) * 2
        director_str = (f" {row['director'].replace(' ', '_').lower()}") * 3 if row['director'] != 'Unknown' else ''
        overview_clean = ' ' + row['overview'].lower().replace('\n', ' ')
        return (genres_str + keywords_str + cast_str + director_str + overview_clean).strip()

    df['tags'] = df.apply(build_rich_tags, axis=1)

    logger.info(f"Preprocessing completed successfully for {len(df)} movies.")
    return df


if __name__ == '__main__':
    df = load_and_merge_data()
    cleaned_df = clean_data(df)
    print("Cleaned DataFrame sample:")
    print(cleaned_df[['id', 'title', 'release_year', 'director', 'vote_average']].head())

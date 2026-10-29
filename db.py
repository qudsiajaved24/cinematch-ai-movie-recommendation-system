"""
Database module for CineMatch AI using Python's built-in sqlite3.
Handles user authentication (signup, login, profile), watchlists, and user interactions.
"""

import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cinematch.db')


def get_db():
    """Returns a SQLite connection with row_factory set to sqlite3.Row."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_db()
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            favorite_genres TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Watchlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(user_id, movie_id)
        )
    ''')

    # Movie Ratings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating INTEGER NOT NULL,
            rated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            UNIQUE(user_id, movie_id)
        )
    ''')

    conn.commit()
    conn.close()


def create_user(username, email, password):
    """
    Registers a new user.
    Returns (user_id, None) on success or (None, error_message) on failure.
    """
    username = username.strip()
    email = email.strip().lower()

    if not username or len(username) < 3:
        return None, "Username must be at least 3 characters long."
    if not email or '@' not in email:
        return None, "Please provide a valid email address."
    if not password or len(password) < 6:
        return None, "Password must be at least 6 characters long."

    pwd_hash = generate_password_hash(password)

    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
            (username, email, pwd_hash)
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id, None
    except sqlite3.IntegrityError as e:
        conn.close()
        err_str = str(e).lower()
        if 'username' in err_str:
            return None, "This username is already taken. Please choose another."
        elif 'email' in err_str:
            return None, "An account with this email already exists."
        return None, "Username or email is already registered."


def authenticate_user(identifier, password):
    """
    Authenticates by username or email.
    Returns user dict on success, None on invalid credentials.
    """
    identifier = identifier.strip().lower()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        'SELECT * FROM users WHERE LOWER(username) = ? OR LOWER(email) = ?',
        (identifier, identifier)
    )
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None


def get_user_by_id(user_id):
    """Fetches user details by user ID."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, favorite_genres, created_at FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def update_user_profile(user_id, favorite_genres):
    """Updates favorite genres for user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET favorite_genres = ? WHERE id = ?', (favorite_genres, user_id))
    conn.commit()
    conn.close()


def add_to_watchlist(user_id, movie_id):
    """Adds a movie to the user's watchlist."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT OR IGNORE INTO watchlist (user_id, movie_id) VALUES (?, ?)',
            (user_id, movie_id)
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.close()
        return False


def remove_from_watchlist(user_id, movie_id):
    """Removes a movie from the user's watchlist."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'DELETE FROM watchlist WHERE user_id = ? AND movie_id = ?',
        (user_id, movie_id)
    )
    conn.commit()
    conn.close()
    return True


def get_user_watchlist_ids(user_id):
    """Returns list of movie IDs currently in user's watchlist."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT movie_id FROM watchlist WHERE user_id = ? ORDER BY added_at DESC', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [r['movie_id'] for r in rows]


def is_movie_in_watchlist(user_id, movie_id):
    """Checks if a given movie is in the user's watchlist."""
    if not user_id:
        return False
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM watchlist WHERE user_id = ? AND movie_id = ?', (user_id, movie_id))
    found = cursor.fetchone() is not None
    conn.close()
    return found


def save_user_rating(user_id, movie_id, rating):
    """Saves or updates user's 1-5 star rating for a movie."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ratings (user_id, movie_id, rating)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id, movie_id) DO UPDATE SET rating = excluded.rating, rated_at = CURRENT_TIMESTAMP
    ''', (user_id, movie_id, rating))
    conn.commit()
    conn.close()
    return True


def get_user_rated_movie_ids(user_id, min_rating=3):
    """Returns list of (movie_id, rating) tuples where user gave at least min_rating."""
    if not user_id:
        return []
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT movie_id, rating FROM ratings WHERE user_id = ? AND rating >= ? ORDER BY rating DESC, rated_at DESC',
        (user_id, min_rating)
    )
    rows = cursor.fetchall()
    conn.close()
    return [(r['movie_id'], r['rating']) for r in rows]


def get_user_stats(user_id):
    """Returns total watchlist count and total ratings given by user."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) as cnt FROM watchlist WHERE user_id = ?', (user_id,))
    watchlist_cnt = cursor.fetchone()['cnt']

    cursor.execute('SELECT COUNT(*) as cnt FROM ratings WHERE user_id = ?', (user_id,))
    ratings_cnt = cursor.fetchone()['cnt']
    conn.close()

    return {
        'watchlist_count': watchlist_cnt,
        'ratings_count': ratings_cnt
    }


# Automatically initialize tables on import
init_db()

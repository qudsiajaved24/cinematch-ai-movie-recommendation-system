"""
CineMatch AI - Flask Web Application for Machine Learning Movie Recommendation.
Pages: Home, Explore, Recommendations, Watchlist, About, plus User Profile & Authentication.
"""

import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ml.recommender import MovieRecommender
from ml.train_model import train
import db

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'cinematch-secret-user-session-2026'

# Ensure models exist or train on startup
models_dir = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(os.path.join(models_dir, 'random_forest_model.pkl')):
    logger.info("Trained model artifacts not found. Starting training pipeline...")
    train()

# Initialize Recommender engine
recommender = MovieRecommender(models_dir=models_dir)

# All standard TMDB Genres for filtering
GENRES_LIST = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
    'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
    'TV Movie', 'Thriller', 'War', 'Western'
]


@app.context_processor
def inject_global_data():
    """Injects user authentication status, watchlist count, and metrics into all templates."""
    current_user = None
    watchlist_count = 0
    user_id = session.get('user_id')
    if user_id:
        current_user = db.get_user_by_id(user_id)
        if current_user:
            stats = db.get_user_stats(user_id)
            watchlist_count = stats['watchlist_count']
        else:
            session.pop('user_id', None)

    metrics = recommender.metrics or {
        'total_movies': 4803,
        'mae': 0.4929,
        'rmse': 0.6863,
        'r2': 0.6876,
        'algorithm': 'Random Forest Regressor'
    }
    return {
        'current_user': current_user,
        'watchlist_count': watchlist_count,
        'metrics': metrics,
        'genres_list': GENRES_LIST,
        'app_name': 'CineMatch'
    }


# ===================== CORE PAGES =====================

@app.route('/')
def home():
    """Home Page - Hero banner, Search, Featured Movies & ML highlights."""
    featured_movies = recommender.get_featured_movies(count=12)
    sample_movies = [
        {'id': 27205, 'title': 'Inception'},
        {'id': 157336, 'title': 'Interstellar'},
        {'id': 155, 'title': 'The Dark Knight'},
        {'id': 680, 'title': 'Pulp Fiction'},
        {'id': 597, 'title': 'Titanic'},
        {'id': 603, 'title': 'The Matrix'}
    ]

    # Check watchlist status for featured movies if user logged in
    user_id = session.get('user_id')
    user_watchlist = set(db.get_user_watchlist_ids(user_id)) if user_id else set()
    for m in featured_movies:
        m['in_watchlist'] = m['id'] in user_watchlist

    return render_template(
        'index.html',
        featured_movies=featured_movies,
        sample_movies=sample_movies
    )


@app.route('/explore')
def explore():
    """Explore Page - Browse complete dataset of 4,803 real movies with filters."""
    genre = request.args.get('genre', 'All')
    year = request.args.get('year', 'All')
    min_rating = float(request.args.get('rating', 0.0))
    sort_by = request.args.get('sort', 'popularity')
    page = int(request.args.get('page', 1))
    search_query = request.args.get('q', '').strip()

    user_id = session.get('user_id')
    user_watchlist = set(db.get_user_watchlist_ids(user_id)) if user_id else set()

    if search_query:
        matched = recommender.search_movies(search_query, limit=60)
        for m in matched:
            m['in_watchlist'] = m['id'] in user_watchlist
        return render_template(
            'explore.html',
            movies=matched,
            total_count=len(matched),
            search_query=search_query,
            current_genre='All',
            current_year='All',
            current_rating=0.0,
            current_sort=sort_by,
            current_page=1,
            total_pages=1
        )

    result = recommender.explore_movies(
        genre=genre,
        year=year,
        min_rating=min_rating,
        sort_by=sort_by,
        page=page,
        per_page=18
    )

    for m in result['movies']:
        m['in_watchlist'] = m['id'] in user_watchlist

    return render_template(
        'explore.html',
        movies=result['movies'],
        total_count=result['total'],
        search_query='',
        current_genre=genre,
        current_year=year,
        current_rating=min_rating,
        current_sort=sort_by,
        current_page=result['page'],
        total_pages=result['pages']
    )


@app.route('/recommendations')
@app.route('/recommendations/<int:movie_id>')
@app.route('/recommend/<int:movie_id>')
def recommendations(movie_id=None):
    """
    Recommendations Page - Generates hybrid ML recommendations.
    Supports 3 modes:
      1. Based on Selected Movie ('source=movie' or URL movie_id)
      2. Based on User's Watchlist ('source=watchlist')
      3. Based on User's High Ratings ('source=rated')
    """
    user_id = session.get('user_id')
    user_watchlist_ids = db.get_user_watchlist_ids(user_id) if user_id else []
    user_watchlist_set = set(user_watchlist_ids)
    user_rated_tuples = db.get_user_rated_movie_ids(user_id, min_rating=3) if user_id else []
    user_rated_ids = [t[0] for t in user_rated_tuples]

    source = request.args.get('source')
    if not source:
        if movie_id is not None:
            source = 'movie'
        else:
            source = 'movie'

    # NEW: user's favorite genres (saved in profile)
    favorite_genres = []
    if user_id:
        u = db.get_user_by_id(user_id)
        if u and u.get('favorite_genres'):
            favorite_genres = [g.strip() for g in u['favorite_genres'].split(',') if g.strip()]

    # Weights: 4 factors when the user has favorite genres, otherwise original 3 factors
    if favorite_genres:
        weight_ml = float(request.args.get('w_ml', 0.45))
        weight_sim = float(request.args.get('w_sim', 0.30))
        weight_rating = float(request.args.get('w_rating', 0.15))
        weight_pref = float(request.args.get('w_pref', 0.10))
    else:
        weight_ml = float(request.args.get('w_ml', 0.50))
        weight_sim = float(request.args.get('w_sim', 0.35))
        weight_rating = float(request.args.get('w_rating', 0.15))
        weight_pref = 0.0

    total_w = weight_ml + weight_sim + weight_rating + weight_pref
    if total_w > 0:
        weight_ml /= total_w
        weight_sim /= total_w
        weight_rating /= total_w
        weight_pref /= total_w

    rec_result = None
    anchor_mode = source
    target_movie = None
    anchor_movies = []

    if source == 'watchlist' and user_id and user_watchlist_ids:
        rec_result = recommender.recommend_from_multiple(
            anchor_movie_ids=user_watchlist_ids,
            top_n=16,
            weight_ml=weight_ml,
            weight_content=weight_sim,
            weight_rating=weight_rating,
            favorite_genres=favorite_genres,
            weight_pref=weight_pref,
            exclude_ids=user_watchlist_ids
        )
        if rec_result:
            anchor_movies = rec_result.get('anchor_movies', [])
    elif source == 'rated' and user_id and user_rated_ids:
        rec_result = recommender.recommend_from_multiple(
            anchor_movie_ids=user_rated_ids,
            top_n=16,
            weight_ml=weight_ml,
            weight_content=weight_sim,
            weight_rating=weight_rating,
            favorite_genres=favorite_genres,
            weight_pref=weight_pref,
            exclude_ids=user_rated_ids
        )
        if rec_result:
            anchor_movies = rec_result.get('anchor_movies', [])
    else:
        # Movie anchor mode
        anchor_mode = 'movie'
        if movie_id is None:
            movie_id_arg = request.args.get('movie_id')
            if movie_id_arg and movie_id_arg.isdigit():
                movie_id = int(movie_id_arg)
            else:
                movie_id = user_watchlist_ids[0] if user_watchlist_ids else 27205  # Default: Inception

        movie = recommender.get_movie_by_id(movie_id)
        if not movie:
            return render_template('error.html', message=f"Movie with ID {movie_id} was not found."), 404

        target_movie = movie
        target_movie['in_watchlist'] = target_movie['id'] in user_watchlist_set

        rec_result = recommender.recommend(
            movie_id=movie_id,
            top_n=16,
            weight_ml=weight_ml,
            weight_content=weight_sim,
            weight_rating=weight_rating,
            favorite_genres=favorite_genres,
            weight_pref=weight_pref
        )

    if not rec_result or not rec_result.get('recommendations'):
        return render_template(
            'error.html',
            message="No recommendations could be generated for the selected criteria. Please add films to your watchlist or pick a movie anchor."
        )

    for r in rec_result['recommendations']:
        r['in_watchlist'] = r['id'] in user_watchlist_set

    # Quick selection list of popular movies for easy switching on Recommendations page
    quick_anchors = [
        {'id': 27205, 'title': 'Inception'},
        {'id': 157336, 'title': 'Interstellar'},
        {'id': 155, 'title': 'The Dark Knight'},
        {'id': 680, 'title': 'Pulp Fiction'},
        {'id': 597, 'title': 'Titanic'},
        {'id': 603, 'title': 'The Matrix'},
        {'id': 120, 'title': 'The Fellowship of the Ring'},
        {'id': 98, 'title': 'Gladiator'}
    ]

    return render_template(
        'recommendations.html',
        target_movie=target_movie,
        anchor_movies=anchor_movies,
        anchor_mode=anchor_mode,
        recommendations=rec_result['recommendations'],
        total_candidates=rec_result['total_candidates_analyzed'],
        weights=rec_result['weights'],
        quick_anchors=quick_anchors,
        watchlist_total=len(user_watchlist_ids),
        rated_total=len(user_rated_ids),
        favorite_genres=favorite_genres
    )


# Alias recommend route to match existing links
@app.route('/recommend/<int:movie_id>')
def recommend(movie_id):
    return recommendations(movie_id)


@app.route('/watchlist')
def watchlist():
    """Watchlist Page - Displays user's saved movies with options to get recommendations or remove."""
    user_id = session.get('user_id')
    if not user_id:
        return render_template('watchlist_guest.html')

    user = db.get_user_by_id(user_id)
    movie_ids = db.get_user_watchlist_ids(user_id)
    watchlist_movies = []
    for mid in movie_ids:
        m = recommender.get_movie_by_id(mid)
        if m:
            m['in_watchlist'] = True
            watchlist_movies.append(m)

    return render_template(
        'watchlist.html',
        user=user,
        movies=watchlist_movies,
        total_movies=len(watchlist_movies)
    )


@app.route('/about')
def about():
    """Basic About Page - Explains CineMatch AI, TMDB dataset, and recommendation mechanism."""
    return render_template('about.html')


@app.route('/about-ml')
def about_ml_redirect():
    """Redirects older about-ml link to /about."""
    return redirect(url_for('about'))


@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    """
    Movie Details Page - Full metadata, ratings, watchlist button,
    AND directly displays full recommendations right below the opened movie!
    """
    movie = recommender.get_movie_by_id(movie_id)
    if not movie:
        return render_template('error.html', message=f"Movie with ID {movie_id} could not be found."), 404

    user_id = session.get('user_id')
    user_watchlist_set = set(db.get_user_watchlist_ids(user_id)) if user_id else set()
    movie['in_watchlist'] = movie['id'] in user_watchlist_set

    # Generate full recommendations directly below the opened movie
    recommendations_list = []
    try:
        rec_data = recommender.recommend(movie_id, top_n=12)
        if rec_data and rec_data.get('recommendations'):
            recommendations_list = rec_data['recommendations']
            for r in recommendations_list:
                r['in_watchlist'] = r['id'] in user_watchlist_set
    except Exception as e:
        logger.warning(f"Error computing recommendations for movie {movie_id}: {e}")

    return render_template(
        'movie.html',
        movie=movie,
        recommendations=recommendations_list
    )


@app.route('/search')
def search():
    """Search endpoint."""
    query = request.args.get('q', '').strip()
    if not query:
        return redirect(url_for('explore'))
    return redirect(url_for('explore', q=query))


# ===================== USER AUTHENTICATION & PROFILE =====================

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User Sign Up Page."""
    if session.get('user_id'):
        return redirect(url_for('profile'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if password != confirm_password:
            error = "Passwords do not match. Please verify your password."
        else:
            user_id, err = db.create_user(username, email, password)
            if err:
                error = err
            else:
                session['user_id'] = user_id
                session['username'] = username
                flash("Welcome to CineMatch! Your profile has been created.", "success")
                return redirect(url_for('profile'))

    return render_template('signup.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User Log In Page."""
    if session.get('user_id'):
        return redirect(url_for('profile'))

    error = None
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '')

        user = db.authenticate_user(identifier, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f"Welcome back, {user['username']}!", "success")
            next_url = request.args.get('next') or url_for('profile')
            return redirect(next_url)
        else:
            error = "Invalid username/email or password. Please try again."

    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    """Clears user session and logs out."""
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for('home'))


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    """User Profile Page."""
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login', next=url_for('profile')))

    user = db.get_user_by_id(user_id)
    if not user:
        session.clear()
        return redirect(url_for('login'))

    if request.method == 'POST':
        favorite_genres = request.form.getlist('genres')
        genres_str = ', '.join(favorite_genres)
        db.update_user_profile(user_id, genres_str)
        user['favorite_genres'] = genres_str
        flash("Profile updated successfully!", "success")

    stats = db.get_user_stats(user_id)
    return render_template(
        'profile.html',
        user=user,
        stats=stats
    )


# ===================== WATCHLIST & INTERACTIVE APIs =====================

@app.route('/api/watchlist/toggle', methods=['POST'])
def api_toggle_watchlist():
    """Toggles movie addition or removal from user's watchlist."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({
            'status': 'unauthorized',
            'message': 'Please log in or sign up to add movies to your personal watchlist.'
        }), 401

    data = request.get_json(silent=True) or {}
    movie_id = data.get('movie_id')
    if not movie_id:
        return jsonify({'status': 'error', 'message': 'Missing movie_id'}), 400

    movie_id = int(movie_id)
    is_in = db.is_movie_in_watchlist(user_id, movie_id)
    if is_in:
        db.remove_from_watchlist(user_id, movie_id)
        in_watchlist = False
        msg = "Movie removed from watchlist"
    else:
        db.add_to_watchlist(user_id, movie_id)
        in_watchlist = True
        msg = "Movie added to your watchlist"

    stats = db.get_user_stats(user_id)
    return jsonify({
        'status': 'success',
        'in_watchlist': in_watchlist,
        'watchlist_count': stats['watchlist_count'],
        'message': msg
    })


@app.route('/watchlist/remove/<int:movie_id>', methods=['POST'])
def watchlist_remove(movie_id):
    """Direct form POST route to remove from watchlist."""
    user_id = session.get('user_id')
    if user_id:
        db.remove_from_watchlist(user_id, movie_id)
        flash("Movie removed from your watchlist.", "info")
    return redirect(url_for('watchlist'))


@app.route('/api/search')
def api_search():
    """Instant JSON search for live search dropdown suggestions."""
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify([])

    results = recommender.search_movies(query, limit=8)
    clean_results = []
    for r in results:
        clean_results.append({
            'id': r['id'],
            'title': r['title'],
            'release_year': r['release_year'],
            'genres_str': r['genres_str'],
            'director': r['director'],
            'vote_average': r['vote_average'],
            'poster_url': r.get('poster_url')
        })
    return jsonify(clean_results)


@app.route('/api/recommend/<int:movie_id>')
def api_recommend(movie_id):
    top_n = int(request.args.get('top_n', 10))
    rec = recommender.recommend(movie_id, top_n=top_n)
    if not rec:
        return jsonify({'error': 'Movie not found or failed to generate recommendations'}), 404
    return jsonify(rec)


@app.route('/api/metrics')
def api_metrics():
    return jsonify(recommender.metrics or {})


@app.route('/api/rate', methods=['POST'])
def api_rate():
    data = request.get_json(silent=True) or {}
    movie_id = data.get('movie_id')
    user_rating = data.get('rating')

    if not movie_id or user_rating is None:
        return jsonify({'status': 'error', 'message': 'Missing movie_id or rating'}), 400

    user_id = session.get('user_id')
    if user_id:
        db.save_user_rating(user_id, int(movie_id), int(user_rating))

    return jsonify({
        'status': 'success',
        'message': f'Rating of {user_rating} stars saved',
        'movie_id': movie_id,
        'user_rating': user_rating
    })


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page or movie not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="An internal server error occurred."), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run CineMatch AI Flask Server")
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)))
    parser.add_argument('--host', type=str, default=os.environ.get('HOST', '127.0.0.1'))
    args, _ = parser.parse_known_args()

    bind_host = '0.0.0.0' if os.environ.get('PORT') or args.host == '0.0.0.0' else args.host
    port = args.port

    print(f"\nRunning on:\nhttp://{args.host}:{port}\n")
    app.run(host=bind_host, port=port, debug=False)
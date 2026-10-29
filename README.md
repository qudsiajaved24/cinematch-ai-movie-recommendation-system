# AI Movie Recommendation System (CineMatch)

A simple, robust, and fully functional **AI / Machine Learning Movie Recommendation System** built with **Python, Flask, Scikit-learn, Pandas, NumPy, HTML, CSS, and Vanilla JavaScript**.

Trained on the **real TMDB 5000 movie dataset (4,803 real movies)**, this project strictly avoids synthetic user profiles, fake random ratings, and dummy data. It demonstrates a genuine academic Machine Learning pipeline from raw CSV data to live dynamic recommendations.

---

## 1. Project Objective

Traditional movie recommendation systems often rely on either hardcoded lists, unweighted content filters that recommend obscure titles, or synthetic collaborative filtering models. 

This project solves this problem by training a real **Random Forest Regressor** to predict audience movie quality scores from multi-modal metadata, combined with **TF-IDF Cosine Similarity** to measure thematic relevance. When a user selects any film, the engine dynamically ranks all 4,803 candidates in the dataset using a verifiable hybrid formula.

---

## 2. Technology Stack

* **Backend & Web Server**: Python 3.10+, Flask
* **Data Processing & Manipulation**: Pandas, NumPy
* **Machine Learning & NLP**: Scikit-Learn (RandomForestRegressor, TfidfVectorizer, StandardScaler, train_test_split, evaluation metrics)
* **Model Serialization**: Joblib
* **Frontend Structure & Styling**: Semantic HTML5, Custom Dark Cinematic CSS (no heavy frameworks, zero-pill discipline)
* **Client-side Interactivity**: Lightweight Vanilla JavaScript (search autocomplete, star rating feedback)

---

## 3. Dataset Information

The system uses the canonical **TMDB 5000 Movie Dataset**:

* `data/movies_tmdb.csv` (4,803 rows): Budget, genres, homepage, ID, keywords, overview, popularity, release date, revenue, runtime, title, vote average, vote count.
* `data/credits_tmdb.csv` (4,803 rows): Movie ID, title, JSON-serialized cast, and JSON-serialized crew.

The datasets are merged on movie `id == movie_id`. After deduplication, all **4,803 authentic movies** are cleaned and indexed.

---

## 4. Machine Learning Algorithm: Random Forest Regressor

The core regression model is Scikit-learn's `RandomForestRegressor`. 

A Random Forest is an ensemble learning algorithm that constructs 100 decorrelated decision trees during training. For quality score prediction:
* Each tree is trained on a bootstrap sample of the dataset.
* At each split, a random subset of features is evaluated, reducing variance and avoiding overfitting.
* Predictions are aggregated by averaging tree outputs:
  $$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)$$

### Target Variable ($y$)
* The real TMDB audience `vote_average` (0.0 to 10.0 scale). No random numbers or synthetic profiles are generated.

---

## 5. Feature Engineering

1. **Numerical Features** (`budget`, `popularity`, `runtime`, `vote_count`, `release_year`):
   * Heavily skewed continuous variables (`budget`, `vote_count`, `popularity`) are transformed via $\log(1 + x)$ for numerical stability.
   * Features are standardized using `StandardScaler` ($\mu = 0, \sigma = 1$).
2. **Text & Content Features** (`tags`):
   * Entity-weighted tags are assembled:
     * Director (3× weight)
     * Genres (3× weight)
     * Key Actors / Cast (2× weight)
     * Plot Keywords (2× weight)
     * Overview narrative (1× weight)
   * Vectorized using `TfidfVectorizer(max_features=2500, stop_words='english', min_df=2)`.
3. **Combined Feature Matrix**:
   * Scipy sparse horizontal stacking: `X = hstack([scaled_numerical, tfidf_matrix]).tocsr()`, resulting in 2,505 analytical features.

---

## 6. Training & Evaluation Results

The dataset is partitioned into an **80% training split** and an **independent 20% testing split** (`random_state=42`):

* **Total Samples**: 4,803 movies
* **Training Partition**: 3,842 samples (80%)
* **Testing Partition**: 961 samples (20%)

### Evaluation Metrics on Independent Test Data:
* **Mean Absolute Error (MAE)**: `0.4929`
  *(On a 10-point scale, the model's quality prediction deviates by less than 0.5 points on average)*
* **Root Mean Squared Error (RMSE)**: `0.6863`
* **Coefficient of Determination ($R^2$ Score)**: `0.6876`
  *(Explains nearly 69% of the rating variance based on movie features alone)*

---

## 7. Hybrid Recommendation Scoring Formula

When a user selects a movie:
1. The target film's TF-IDF vector is compared against all 4,803 movies using `cosine_similarity`.
2. Candidate movies with contextual similarity are evaluated.
3. Random Forest predicts expected quality scores.
4. Final ranking score is computed:

$$\text{Final Score} = 0.50 \times \text{ML\_Score}_{\text{norm}} + 0.35 \times \text{Similarity}_{\text{norm}} + 0.15 \times \text{Rating}_{\text{norm}}$$

5. The target movie is excluded and the Top $N$ candidates are returned.

---

## 8. Project Structure & Pages

The application is structured into the following pages:

* **Home (`/`)**: Hero discovery banner, instant search across 4,803 real movies, curated acclaimed titles, and quick recommendation demo shortcuts.
* **Explore (`/explore`)**: Browse and filter the complete 4,803 real movie catalog by genre, release decade, rating threshold, and sorting with pagination.
* **Recommendations (`/recommendations`)**: Dedicated hybrid recommendation engine allowing selection of any anchor movie to dynamically rank all 4,803 candidates with ML explainability metrics (ML quality score, content similarity %, TMDB rating, hybrid score).
* **Watchlist (`/watchlist`)**: Personalized library for authenticated users to bookmark movies, track films they want to watch, and trigger recommendations directly from saved items.
* **About (`/about`)**: Clean, accessible overview explaining CineMatch AI, the TMDB dataset, the Random Forest Regressor, and academic model evaluation metrics (MAE 0.4929, RMSE 0.6863, R² 0.6876).
* **User Authentication & Profile (`/signup`, `/login`, `/profile`, `/logout`)**: Secure user registration, credential hashing with Werkzeug, personalized favorite genres selection, and watchlist management backed by SQLite (`data/cinematch.db`).
├── data/
│   ├── movies_tmdb.csv        # Real TMDB movies dataset (4,803 records)
│   └── credits_tmdb.csv       # Real TMDB credits dataset (4,803 records)
├── ml/
│   ├── __init__.py
│   ├── preprocessing.py       # Data cleaning, JSON parsing, and tag creation
│   ├── feature_engineering.py # TF-IDF vectorization and StandardScaler
│   ├── train_model.py         # 80/20 split, Random Forest training & evaluation
│   ├── evaluate_model.py      # Independent evaluation & feature importance inspector
│   └── recommender.py         # Hybrid candidate ranking and search engine
├── models/
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   ├── movie_features.pkl
│   ├── processed_movies.pkl
│   └── metrics.pkl
├── templates/
│   ├── base.html              # Cinematic header, search, and footer layout
│   ├── index.html             # Hero banner, featured movies, ML dashboard
│   ├── explore.html           # 4,803 movie browser with filters and pagination
│   ├── movie.html             # Movie details, metadata, and rating widget
│   ├── recommendations.html   # Ranked recommendations with ML explainability
│   ├── about_ml.html          # Academic documentation and viva examination guide
│   └── error.html             # Error handling fallback
├── static/
│   ├── css/
│   │   └── style.css          # Dark cinematic styling (pure CSS)
│   └── js/
│       └── script.js          # Live search autocomplete and rating interaction
├── requirements.txt           # Minimal dependencies
└── README.md                  # Project documentation
```

---

## 9. How to Run Locally

### Prerequisites
* Python 3.10+
* pip

### Installation & Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the Random Forest Regressor & build model artifacts
python ml/train_model.py

# 3. View evaluation report (optional)
python ml/evaluate_model.py

# 4. Start the Flask server
python app.py
```

Open your browser at:
```text
http://127.0.0.1:3000   (or http://127.0.0.1:5000 if PORT=5000)
```

---

## 10. Academic Integrity Declaration

* **No synthetic users** (no fake `scifi_fan` / `drama_fan` profiles).
* **No synthetic ratings** (no `np.random.uniform` or `np.random.normal`).
* **No hardcoded recommendations**.
* All metrics, predictions, and recommendations are computed directly from the real 4,803 TMDB movie records.

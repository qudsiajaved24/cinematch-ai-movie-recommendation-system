/**
 * CineMatch AI - Lightweight Vanilla JavaScript for Interactive UI
 * Handles live search autocomplete, user rating feedback, and watchlist toggles.
 */

document.addEventListener('DOMContentLoaded', () => {
    initSearchAutocomplete('headerSearchInput', 'searchResultsDropdown');
    initSearchAutocomplete('heroSearchInput', 'heroSearchDropdown');
    initRatingWidget();
});

/**
 * Toggles a movie in or out of the user's personal watchlist.
 */
function toggleWatchlist(btn, movieId) {
    if (!btn || !movieId) return;

    fetch('/api/watchlist/toggle', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ movie_id: movieId })
    })
    .then(res => {
        if (res.status === 401) {
            showToast("Please log in or sign up to add movies to your personal Watchlist.", "warning");
            // Highlight auth buttons in nav
            const signupBtn = document.querySelector('.nav-cta-btn');
            if (signupBtn) {
                signupBtn.style.boxShadow = "0 0 12px var(--gold-primary)";
                setTimeout(() => signupBtn.style.boxShadow = "", 2000);
            }
            return null;
        }
        return res.json();
    })
    .then(data => {
        if (!data || data.status !== 'success') return;

        const isIn = data.in_watchlist;
        const count = data.watchlist_count;

        // Update all buttons on the page with this movieId
        const allBtns = document.querySelectorAll(`[data-movie-id="${movieId}"]`);
        allBtns.forEach(b => {
            if (b.classList.contains('watchlist-card-btn')) {
                b.textContent = isIn ? '✓ In Watchlist' : '🔖 Watchlist';
                if (isIn) {
                    b.classList.add('active');
                } else {
                    b.classList.remove('active');
                }
            } else if (b.classList.contains('watchlist-btn') || b.classList.contains('detail-watchlist-btn')) {
                const textSpan = b.querySelector('.btn-text');
                if (textSpan) {
                    textSpan.textContent = isIn ? 'In Watchlist' : 'Add to Watchlist';
                }
                if (isIn) {
                    b.classList.add('active');
                } else {
                    b.classList.remove('active');
                }
            }
        });

        // Update navigation badge count
        const badge = document.getElementById('navWatchlistBadge');
        if (badge) {
            badge.textContent = count;
            badge.style.display = count > 0 ? 'inline-flex' : 'none';
        }

        showToast(data.message, "success");
    })
    .catch(err => {
        console.error('Watchlist toggle error:', err);
    });
}

/**
 * Toast notifications for interactive feedback.
 */
function showToast(message, type = "info") {
    let container = document.querySelector('.flash-messages-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'flash-messages-container';
        document.body.prepend(container);
    }

    const toast = document.createElement('div');
    toast.className = `flash-toast flash-${type}`;
    toast.innerHTML = `
        <span>${message}</span>
        <button type="button" class="flash-close-btn" onclick="this.parentElement.remove()">✕</button>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-10px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

/**
 * Initializes instant search dropdown autocomplete.
 */
function initSearchAutocomplete(inputId, dropdownId) {
    const input = document.getElementById(inputId);
    const dropdown = document.getElementById(dropdownId);
    if (!input || !dropdown) return;

    let debounceTimer = null;

    input.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        clearTimeout(debounceTimer);

        if (query.length < 2) {
            dropdown.innerHTML = '';
            dropdown.classList.remove('active');
            return;
        }

        debounceTimer = setTimeout(() => {
            fetch(`/api/search?q=${encodeURIComponent(query)}`)
                .then(res => res.json())
                .then(items => {
                    if (!items || items.length === 0) {
                        dropdown.innerHTML = `
                            <div class="dropdown-item" style="color: var(--text-muted); font-size: 0.85rem;">
                                No matching movies found.
                            </div>
                        `;
                        dropdown.classList.add('active');
                        return;
                    }

                    let html = '';
                    items.forEach(movie => {
                        const thumbHtml = movie.poster_url 
                            ? `<img src="${movie.poster_url}" alt="${movie.title}" class="dropdown-thumb" onerror="this.outerHTML='<div class=\\'dropdown-thumb-empty\\'>🎬</div>'">`
                            : `<div class="dropdown-thumb-empty">🎬</div>`;

                        html += `
                            <a href="/movie/${movie.id}" class="dropdown-item">
                                ${thumbHtml}
                                <div class="dropdown-info">
                                    <div class="dropdown-title">${movie.title}</div>
                                    <div class="dropdown-meta">
                                        <span>${movie.release_year}</span> · 
                                        <span>${movie.genres_str.split(',')[0]}</span> · 
                                        <span class="text-gold">★ ${movie.vote_average}</span>
                                    </div>
                                </div>
                            </a>
                        `;
                    });

                    dropdown.innerHTML = html;
                    dropdown.classList.add('active');
                })
                .catch(err => {
                    console.error('Search error:', err);
                });
        }, 220);
    });

    // Dismiss search dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!input.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.classList.remove('active');
        }
    });

    // Reopen on focus if query exists
    input.addEventListener('focus', () => {
        if (dropdown.children.length > 0) {
            dropdown.classList.add('active');
        }
    });
}

/**
 * Initializes interactive 5-star rating widget on movie detail page.
 */
function initRatingWidget() {
    const widget = document.querySelector('.user-rating-widget');
    if (!widget) return;

    const movieId = widget.getAttribute('data-movie-id');
    const stars = widget.querySelectorAll('.star-btn');
    const feedbackMsg = document.getElementById('ratingFeedbackMsg');

    // Check if user previously rated this movie in localStorage
    const savedRating = localStorage.getItem(`cinematch_rating_${movieId}`);
    if (savedRating) {
        highlightStars(parseInt(savedRating, 10));
        if (feedbackMsg) {
            feedbackMsg.innerHTML = `Your rating: <strong class="text-gold">${savedRating}/5 stars</strong>`;
        }
    }

    stars.forEach(star => {
        star.addEventListener('mouseenter', () => {
            const val = parseInt(star.getAttribute('data-value'), 10);
            previewStars(val);
        });

        star.addEventListener('click', () => {
            const val = parseInt(star.getAttribute('data-value'), 10);
            localStorage.setItem(`cinematch_rating_${movieId}`, val);
            highlightStars(val);

            if (feedbackMsg) {
                feedbackMsg.innerHTML = `Thank you! Rated <strong class="text-gold">${val}/5 stars</strong>.`;
            }

            fetch('/api/rate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    movie_id: movieId,
                    rating: val
                })
            }).catch(e => console.log('Rating sync:', e));
        });
    });

    widget.addEventListener('mouseleave', () => {
        const currentSaved = localStorage.getItem(`cinematch_rating_${movieId}`);
        if (currentSaved) {
            highlightStars(parseInt(currentSaved, 10));
        } else {
            clearStars();
        }
    });

    function previewStars(val) {
        stars.forEach(s => {
            const sVal = parseInt(s.getAttribute('data-value'), 10);
            s.style.color = sVal <= val ? 'var(--gold-primary)' : 'var(--border-card)';
        });
    }

    function highlightStars(val) {
        stars.forEach(s => {
            const sVal = parseInt(s.getAttribute('data-value'), 10);
            if (sVal <= val) {
                s.classList.add('active');
                s.style.color = 'var(--gold-primary)';
            } else {
                s.classList.remove('active');
                s.style.color = 'var(--border-card)';
            }
        });
    }

    function clearStars() {
        stars.forEach(s => {
            s.classList.remove('active');
            s.style.color = 'var(--border-card)';
        });
    }
}

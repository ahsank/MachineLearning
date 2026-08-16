"""A minimal SVD-based collaborative-filtering recommender.

Illustrates the classic "latent factor" recommendation technique: treat
users and items as living in a shared low-dimensional space, discovered by
factoring the (mostly missing) user-item rating matrix with SVD. See
notes/concepts/singular-value-decomposition.md for the write-up this
accompanies.

Usage:
    python svd_recommender.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

USERS = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
SONGS = ["rock_a", "rock_b", "rock_c", "pop_a", "pop_b", "pop_c"]

# Rows = users, columns = songs. Ratings are 1-5; 0 means "not rated yet".
# Hand-crafted so "rock" fans (u1-u4) and "pop" fans (u5-u8) each rate
# their own genre highly and barely touch the other -- the latent taste
# structure SVD is expected to recover from the *observed* ratings alone.
RATINGS = np.array([
    [5, 4, 5, 1, 0, 2],
    [4, 5, 4, 0, 1, 0],
    [5, 5, 0, 2, 0, 1],
    [4, 4, 5, 0, 2, 0],
    [1, 0, 2, 5, 4, 5],
    [0, 1, 0, 4, 5, 4],
    [2, 0, 1, 5, 5, 0],
    [0, 2, 0, 4, 4, 5],
], dtype=float)

N_FACTORS = 2  # latent dimensions kept from the truncated SVD


def predict_ratings(ratings: np.ndarray, k: int) -> np.ndarray:
    """Fill missing (0) entries with each user's mean rating, mean-center
    the matrix, and reconstruct it from a rank-k SVD.

    SVD needs a dense matrix, so the mean-fill gives it something to
    factor; centering keeps the "no information yet" fill value from
    distorting the factorization. The top-k singular directions capture
    shared taste patterns (here, "likes rock" vs "likes pop"), and
    reconstructing through only those directions turns a user's missing
    ratings into values consistent with the users who rate similarly.
    """
    observed = ratings > 0
    user_means = np.array([
        ratings[u, observed[u]].mean() if observed[u].any() else 0.0
        for u in range(ratings.shape[0])
    ])

    filled = np.where(observed, ratings, user_means[:, None])
    centered = filled - user_means[:, None]

    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    reconstructed = u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]

    predicted = reconstructed + user_means[:, None]
    return np.clip(predicted, 1.0, 5.0)


def top_recommendations(
    user_idx: int, ratings: np.ndarray, predicted: np.ndarray, songs: list[str], n: int = 2
) -> list[tuple[str, float]]:
    """Rank the user's *unrated* songs by predicted rating."""
    unrated = np.where(ratings[user_idx] == 0)[0]
    ranked = sorted(unrated, key=lambda i: predicted[user_idx, i], reverse=True)
    return [(songs[i], predicted[user_idx, i]) for i in ranked[:n]]


def print_ratings_table(title: str, matrix: np.ndarray) -> None:
    print(title)
    print("        " + " ".join(f"{s:>8s}" for s in SONGS))
    for user, row in zip(USERS, matrix):
        print(f"{user:>6s}: " + " ".join(f"{v:8.2f}" for v in row))
    print()


def sanity_check_recovery() -> None:
    """Hide one already-known rating, re-run the pipeline, and check that
    the prediction lands close to the true value -- a quick, informal
    stand-in for the train/test evaluation a real system would do."""
    user_idx, song_idx = 0, 1  # u1's real rating for rock_b
    true_value = RATINGS[user_idx, song_idx]
    assert true_value > 0, "pick a cell that actually has a rating to hide"

    hidden = RATINGS.copy()
    hidden[user_idx, song_idx] = 0
    predicted = predict_ratings(hidden, N_FACTORS)

    print(
        f"Sanity check: hid {USERS[user_idx]}'s real rating of "
        f"{SONGS[song_idx]} ({true_value:.0f}) and predicted "
        f"{predicted[user_idx, song_idx]:.2f} from the rest of the matrix"
    )
    print()


def plot_ratings(ratings: np.ndarray, predicted: np.ndarray, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, matrix, title in zip(
        axes,
        [ratings, predicted],
        ["Observed ratings\n(blank = not rated)", "Predicted ratings\n(SVD reconstruction)"],
    ):
        display = np.where(matrix == 0, np.nan, matrix) if matrix is ratings else matrix
        im = ax.imshow(display, cmap="YlOrRd", vmin=1, vmax=5)
        ax.set_xticks(range(len(SONGS)), SONGS, rotation=45, ha="right")
        ax.set_yticks(range(len(USERS)), USERS)
        ax.set_title(title)
        for i in range(len(USERS)):
            for j in range(len(SONGS)):
                if not np.isnan(display[i, j]):
                    ax.text(j, i, f"{display[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="rating")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


def main() -> None:
    predicted = predict_ratings(RATINGS, N_FACTORS)

    print_ratings_table("Observed ratings (0 = not rated):", RATINGS)
    print_ratings_table("Predicted ratings (SVD reconstruction):", predicted)

    print("Top recommendations for unrated songs:")
    for i, user in enumerate(USERS):
        recs = top_recommendations(i, RATINGS, predicted, SONGS)
        recs_str = ", ".join(f"{song} ({score:.2f})" for song, score in recs)
        print(f"  {user}: {recs_str}")
    print()

    sanity_check_recovery()
    plot_ratings(RATINGS, predicted, out_path="svd_recommender.png")


if __name__ == "__main__":
    main()

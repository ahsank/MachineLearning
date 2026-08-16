"""Word vectors from a hand-built co-occurrence matrix, via SVD.

A minimal, from-scratch illustration of "count-based" word embeddings:
tokenize a small corpus, count how often words appear near each other
(a word-word co-occurrence matrix), then factor that matrix with SVD and
keep the top singular directions as low-dimensional word vectors. See
notes/concepts/singular-value-decomposition.md for the write-up this
accompanies.

Usage:
    python svd_word_vectors.py
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np

# Five short, hand-written documents spanning two loose topics (beverages,
# winter weather). Words like "coffee", "tea", "rain", "snow" and "winter"
# repeat across documents within a topic, which is what links each
# document's other words together once the matrix is factored.
DOCUMENTS = [
    "coffee and tea are popular morning drinks",
    "many people drink coffee to start their day at work",
    "tea can also be served cold on a hot afternoon",
    "heavy rain and snow often arrive together during winter storms",
    "the winter storms bring strong wind with rain and snow overnight",
]

# Small stopword list so the plot isn't dominated by function words.
STOPWORDS = {
    "the", "a", "an", "and", "are", "many", "to", "their", "at", "also",
    "be", "on", "often", "during", "together", "can", "with",
}

WINDOW_SIZE = 3  # how many words to each side count as "co-occurring"


def tokenize(document: str) -> list[str]:
    words = re.findall(r"[a-z]+", document.lower())
    return [word for word in words if word not in STOPWORDS]


def build_vocabulary(tokenized_docs: list[list[str]]) -> list[str]:
    return sorted({word for doc in tokenized_docs for word in doc})


def build_co_occurrence_matrix(
    tokenized_docs: list[list[str]], vocab: list[str], window_size: int
) -> np.ndarray:
    """Count how often each pair of words appears within `window_size`
    tokens of each other, document by document (windows never cross a
    document boundary)."""
    index = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)))

    for doc in tokenized_docs:
        for center_pos, center_word in enumerate(doc):
            start = max(0, center_pos - window_size)
            end = min(len(doc), center_pos + window_size + 1)
            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue
                matrix[index[center_word], index[doc[context_pos]]] += 1

    return matrix


def word_vectors_via_svd(matrix: np.ndarray, k: int = 2) -> np.ndarray:
    """Truncated SVD: keep the top-k left singular vectors, scaled by
    their singular values, as k-dimensional word vectors."""
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :k] * s[:k]


def plot_word_vectors(vocab: list[str], vectors: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(vectors[:, 0], vectors[:, 1], color="tab:blue")
    for word, (x, y) in zip(vocab, vectors):
        ax.annotate(word, (x, y), textcoords="offset points", xytext=(4, 4))
    ax.set_title("Word vectors from SVD of a co-occurrence matrix")
    ax.set_xlabel("singular direction 1")
    ax.set_ylabel("singular direction 2")
    ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
    ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    tokenized_docs = [tokenize(doc) for doc in DOCUMENTS]
    vocab = build_vocabulary(tokenized_docs)
    co_occurrence = build_co_occurrence_matrix(tokenized_docs, vocab, WINDOW_SIZE)
    vectors = word_vectors_via_svd(co_occurrence, k=2)

    print(f"Vocabulary ({len(vocab)} words): {vocab}\n")
    print("2D word vectors:")
    for word, vec in zip(vocab, vectors):
        print(f"  {word:>10s}: [{vec[0]:+.3f}, {vec[1]:+.3f}]")

    plot_word_vectors(vocab, vectors, out_path="svd_word_vectors.png")


if __name__ == "__main__":
    main()

# Code

Standalone Python scripts used to experiment with concepts covered in the
notes — each file is self-contained (hardcoded inputs, no shared modules)
and runnable on its own.

## Setup

Each script only needs `requirements.txt` from this folder — there's no
shared virtualenv across scripts.

    cd code
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Run a script

With the venv active:

    python svd_word_vectors.py

The venv persists after the script exits (`.venv/` is gitignored), so you
can rerun scripts or `pip install` more packages without repeating setup.
Next time, just `source .venv/bin/activate` again — no need to recreate it
or reinstall dependencies.

## Clean up

Leave the venv in place if you expect to run scripts again. To remove it
entirely:

    deactivate  # if currently active
    rm -rf .venv

Scripts that save output (e.g. `svd_word_vectors.png`) write it into this
folder — delete those files too if you don't want to keep them around.

## Scripts

- [`svd_word_vectors.py`](svd_word_vectors.py) — builds a word-word
  co-occurrence matrix from a handful of hardcoded example documents,
  factors it with SVD, and plots the resulting word vectors in 2D.
  Accompanies
  [Singular Value Decomposition — Example](../notes/concepts/singular-value-decomposition.md#example-word-vectors-from-a-co-occurrence-matrix).

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
  [Singular Value Decomposition — Example: word vectors](../notes/concepts/singular-value-decomposition.md#example-word-vectors-from-a-co-occurrence-matrix).
- [`svd_recommender.py`](svd_recommender.py) — factors a small hardcoded
  user-song rating matrix with SVD to predict missing ratings and
  recommend unrated songs. Accompanies
  [Singular Value Decomposition — Example: recommender systems](../notes/concepts/singular-value-decomposition.md#example-recommender-systems-via-svd-matrix-factorization).
- [`svd_page_matrix_stock_forecast.py`](svd_page_matrix_stock_forecast.py) —
  downloads one stock's price history (via `yfinance`, needs network
  access), denoises it with SVD on its "Page matrix", and forecasts
  future prices. `python svd_page_matrix_stock_forecast.py [TICKER]
  --period 2y --horizon 20`. Add `--backtest-days 40` to also hold out
  the last 40 trading days, forecast them from the preceding history, and
  plot predicted vs. actual for that window (with RMSE/MAPE printed).
  Accompanies
  [Multivariate Singular Spectrum Analysis — single stock](../notes/papers/mssa-page-matrix-svd.md#worked-example-single-stock).
- [`svd_page_matrix_multi_stock_forecast.py`](svd_page_matrix_multi_stock_forecast.py) —
  same idea across several stocks at once, pooled into one shared model
  via a stacked Page matrix (needs network access). `python
  svd_page_matrix_multi_stock_forecast.py AAPL MSFT GOOG --period 2y`.
  Also supports `--backtest-days`. Accompanies
  [Multivariate Singular Spectrum Analysis — multiple stocks](../notes/papers/mssa-page-matrix-svd.md#worked-example-multiple-stocks).
- [`svd_denoised_ar_stock_forecast.py`](svd_denoised_ar_stock_forecast.py) —
  simplified variant of the single-stock script: denoise with SVD, then
  fit a plain autoregression on *overlapping* windows of the denoised
  series (instead of the paper's one-example-per-page regression). Same
  CLI as `svd_page_matrix_stock_forecast.py`, plus `--window` to set the
  AR lag count independently of the denoising length. Accompanies
  [SVD-Denoised Autoregressive Forecasting](../notes/concepts/svd-denoised-autoregressive-forecasting.md).
- [`svd_denoised_ar_multi_stock_forecast.py`](svd_denoised_ar_multi_stock_forecast.py) —
  same variant across several stocks, pooling overlapping windows from
  every stock's denoised series into one shared regression. Same CLI as
  `svd_page_matrix_multi_stock_forecast.py`. Accompanies
  [SVD-Denoised Autoregressive Forecasting](../notes/concepts/svd-denoised-autoregressive-forecasting.md).

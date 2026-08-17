"""Forecast several stocks jointly via SVD on a stacked Page matrix (mSSA).

Extends the single-series technique in svd_page_matrix_stock_forecast.py
to N series at once, following the "stacked Page matrix" construction in
Agarwal, Alomar & Shah, "On Multivariate Singular Spectrum Analysis"
(arXiv:2006.13448, Section 1.1): build each stock's own Page matrix, glue
them together column-wise, and fit *one* denoising/forecasting model
across all of them. Pooling stocks this way lets the model borrow
strength from shared structure (sector trends, correlated moves) across
series, rather than fitting each one in isolation. See
notes/papers/mssa-page-matrix-svd.md for the write-up this accompanies.

This is a from-scratch, simplified implementation of the technique
described in the paper (not the reference `mSSA` package).

Usage:
    python svd_page_matrix_multi_stock_forecast.py [TICKERS...] [--period 2y] [--horizon 20]
    python svd_page_matrix_multi_stock_forecast.py AAPL MSFT GOOG --period 2y
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ENERGY_THRESHOLD = 0.9  # fraction of singular-value energy to keep when picking rank k
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG"]


def download_prices(tickers: list[str], period: str) -> pd.DataFrame:
    """Closing prices for all tickers, aligned to the trading days every
    ticker has data for."""
    data = yf.download(tickers, period=period, progress=False)["Close"]
    return data.dropna()


def normalize(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each stock's own price series. Stocks trade at very
    different price levels (a $300 vs. $30 stock), so pooling raw prices
    into one shared regression would just have the model fit whichever
    series has the largest scale. Normalizing puts every series'
    day-to-day dynamics on the same footing before pooling."""
    mean = prices.mean(axis=0)
    std = prices.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (prices - mean) / std, mean, std


def page_matrix(x: np.ndarray, L: int) -> np.ndarray:
    """Reshape the most recent multiple-of-L values of `x` into an
    L x (T/L) matrix whose j-th column is the j-th consecutive,
    non-overlapping length-L block of the series."""
    usable = (len(x) // L) * L
    return x[-usable:].reshape(usable // L, L).T


def stacked_page_matrix(prices: np.ndarray, L: int) -> np.ndarray:
    """Column-wise concatenation of each series' own Page matrix, shape
    L x (N * T/L). All series must share the same T (use download_prices,
    which aligns them to common trading days)."""
    return np.hstack([page_matrix(prices[:, n], L) for n in range(prices.shape[1])])


def choose_rank(singular_values: np.ndarray, energy_threshold: float = ENERGY_THRESHOLD) -> int:
    """Smallest k whose top-k singular values capture `energy_threshold`
    of the total squared singular value ("spectral energy")."""
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return int(np.searchsorted(energy, energy_threshold) + 1)


def hard_singular_value_threshold(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k = min(k, len(s))
    return u[:, :k], s[:k], vt[:k, :]


def denoise_all(prices: np.ndarray, L: int) -> tuple[np.ndarray, int]:
    """De-noise every series at once by SVD-ing the *stacked* Page
    matrix: the shared low-rank structure across stocks is what a
    per-stock SVD would miss."""
    stacked = stacked_page_matrix(prices, L)
    _, s, _ = np.linalg.svd(stacked, full_matrices=False)
    k = choose_rank(s)
    u_k, s_k, vt_k = hard_singular_value_threshold(stacked, k)
    stacked_hat = u_k @ np.diag(s_k) @ vt_k

    per_stock = np.hsplit(stacked_hat, prices.shape[1])
    denoised = np.column_stack([block.T.reshape(-1) for block in per_stock])
    return denoised, k


def fit_forecast_model(prices: np.ndarray, L: int, k: int) -> np.ndarray:
    """One shared linear model, fit by pooling training pairs from every
    stock's every page: predict a page's true last value from the
    *denoised* first L-1 rows of that page (the last row is zeroed out
    before denoising so features never leak the target)."""
    stacked = stacked_page_matrix(prices, L)
    stacked_masked = stacked.copy()
    stacked_masked[-1, :] = 0.0
    u_k, s_k, vt_k = hard_singular_value_threshold(stacked_masked, k)
    stacked_hat = u_k @ np.diag(s_k) @ vt_k

    features = stacked_hat[:-1, :].T  # (N * num_pages, L-1)
    targets = stacked[-1, :]  # (N * num_pages,)
    beta, *_ = np.linalg.lstsq(features, targets, rcond=None)
    return beta


def forecast_all(prices: np.ndarray, beta: np.ndarray, horizon: int) -> np.ndarray:
    """Apply the shared one-step model to each stock independently,
    rolling forward `horizon` steps (using each stock's own forecasts to
    extend its own window)."""
    window_size = len(beta)
    num_series = prices.shape[1]
    predictions = np.zeros((horizon, num_series))
    for n in range(num_series):
        history = list(prices[-window_size:, n])
        for h in range(horizon):
            pred = np.dot(history[-window_size:], beta)
            predictions[h, n] = pred
            history.append(pred)
    return predictions


def plot_forecasts(dates, prices, denoised, forecast_dates, predictions, tickers, L, k, out_path):
    fig, axes = plt.subplots(len(tickers), 1, figsize=(10, 3 * len(tickers)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, ticker) in enumerate(zip(axes, tickers)):
        ax.plot(dates, prices[:, i], label="Observed close", color="tab:gray", linewidth=1)
        ax.plot(dates[-len(denoised):], denoised[:, i], label="Denoised", color="tab:blue")
        ax.plot(forecast_dates, predictions[:, i], label="Forecast", color="tab:red", linestyle="--", marker="o", markersize=3)
        ax.axvline(dates[-1], color="lightgray", linewidth=0.8)
        ax.set_title(ticker)
        ax.set_ylabel("price")
    axes[0].legend()
    fig.suptitle(f"Stacked Page-matrix SVD denoising + forecast (L={L}, k={k}, shared model)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--period", default="2y", help="yfinance history window, e.g. 1y, 2y, 5y")
    parser.add_argument("--horizon", type=int, default=20, help="trading days to forecast")
    args = parser.parse_args()

    df = download_prices(args.tickers, args.period)
    prices = df.values.astype(float)
    tickers = list(df.columns)
    prices_norm, mean, std = normalize(prices)

    L = int(round(np.sqrt(len(tickers) * len(prices))))  # paper's suggested L = sqrt(min(N,T)*T)
    print(f"{len(tickers)} tickers x {len(prices)} trading days, page length L={L}")

    denoised_norm, k = denoise_all(prices_norm, L)
    denoised = denoised_norm * std + mean
    print(f"Kept top {k} singular directions (>{ENERGY_THRESHOLD:.0%} of pooled spectral energy)")

    beta = fit_forecast_model(prices_norm, L, k)
    predictions = forecast_all(prices_norm, beta, args.horizon) * std + mean

    print(f"\nLast observed close ({df.index[-1].date()}):")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {prices[-1, i]:.2f}")

    print(f"\nForecast for the next {args.horizon} trading days:")
    header = "        " + " ".join(f"{t:>10s}" for t in tickers)
    print(header)
    for h in range(args.horizon):
        row = " ".join(f"{predictions[h, i]:10.2f}" for i in range(len(tickers)))
        print(f"  t+{h+1:<4d}: {row}")

    forecast_dates = pd.bdate_range(df.index[-1], periods=args.horizon + 1)[1:]
    plot_forecasts(
        df.index, prices, denoised, forecast_dates, predictions,
        tickers, L, k, out_path="svd_page_matrix_multi_stock.png",
    )


if __name__ == "__main__":
    main()

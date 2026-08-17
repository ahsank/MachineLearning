"""Forecast a single stock's closing price via SVD on its Page matrix.

Implements the univariate case (N=1, i.e. plain SSA) of the Page-matrix
technique from Agarwal, Alomar & Shah, "On Multivariate Singular Spectrum
Analysis" (arXiv:2006.13448): reshape the price series into a Page matrix
(non-overlapping length-L segments as columns), denoise it by keeping only
the top-k singular directions (Hard Singular Value Thresholding), then
forecast future prices with the linear model the paper fits on top of that
low-rank structure. See notes/papers/mssa-page-matrix-svd.md for the
write-up this accompanies.

This is a from-scratch, simplified implementation of the technique
described in the paper (not the reference `mSSA` package) -- no missing
values, no confidence intervals, just the core page-matrix + SVD +
regression pipeline applied to real data.

Usage:
    python svd_page_matrix_stock_forecast.py [TICKER] [--period 2y] [--horizon 20]
    python svd_page_matrix_stock_forecast.py [TICKER] --backtest-days 40
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

ENERGY_THRESHOLD = 0.9  # fraction of singular-value energy to keep when picking rank k


def download_prices(ticker: str, period: str) -> pd.Series:
    data = yf.download(ticker, period=period, progress=False)
    return data["Close"][ticker]


def page_matrix(x: np.ndarray, L: int) -> np.ndarray:
    """Reshape the most recent multiple-of-L values of `x` into an
    L x (T/L) matrix whose j-th column is the j-th consecutive,
    non-overlapping length-L block of the series."""
    usable = (len(x) // L) * L
    return x[-usable:].reshape(usable // L, L).T


def choose_rank(singular_values: np.ndarray, energy_threshold: float = ENERGY_THRESHOLD) -> int:
    """Smallest k whose top-k singular values capture `energy_threshold`
    of the total squared singular value ("spectral energy") -- a simple
    stand-in for the paper's cross-validated rank selection."""
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return int(np.searchsorted(energy, energy_threshold) + 1)


def hard_singular_value_threshold(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k = min(k, len(s))
    return u[:, :k], s[:k], vt[:k, :]


def denoise_series(x: np.ndarray, L: int) -> tuple[np.ndarray, int]:
    """The paper's imputation/de-noising step: SVD the Page matrix, keep
    the top-k directions, and read the reconstruction back off as a
    denoised version of the original series."""
    P = page_matrix(x, L)
    u, s, vt = np.linalg.svd(P, full_matrices=False)
    k = choose_rank(s)
    u_k, s_k, vt_k = hard_singular_value_threshold(P, k)
    P_hat = u_k @ np.diag(s_k) @ vt_k
    return P_hat.T.reshape(-1), k  # undo page_matrix's reshape/transpose


def fit_forecast_model(x: np.ndarray, L: int, k: int) -> np.ndarray:
    """The paper's forecasting model: predict the last row of each page
    (the true, next-step value) from the *denoised* first L-1 rows of
    that same page, fit by least squares across all historical pages.

    The last row is zeroed out before denoising so the L-1 features for
    a page never leak information about that page's own target value.
    """
    P = page_matrix(x, L)
    P_masked = P.copy()
    P_masked[-1, :] = 0.0
    u_k, s_k, vt_k = hard_singular_value_threshold(P_masked, k)
    P_hat = u_k @ np.diag(s_k) @ vt_k

    features = P_hat[:-1, :].T  # (num_pages, L-1): denoised history within each page
    targets = P[-1, :]  # (num_pages,): actual next-step value for each page
    beta, *_ = np.linalg.lstsq(features, targets, rcond=None)
    return beta


def forecast(x: np.ndarray, beta: np.ndarray, horizon: int) -> np.ndarray:
    """Roll the one-step model forward `horizon` times: predict the next
    value from the last L-1 known/forecast values, append it, repeat.
    This recursive extension isn't spelled out in the paper (its formula
    only covers one step past the observed data) but is the standard way
    to turn a one-step model into a multi-step forecast."""
    window_size = len(beta)
    history = list(x[-window_size:])
    predictions = []
    for _ in range(horizon):
        pred = np.dot(history[-window_size:], beta)
        predictions.append(pred)
        history.append(pred)
    return np.array(predictions)


def backtest(x: np.ndarray, backtest_days: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Hold out the last `backtest_days` values, fit the model on the
    preceding history only, and forecast that many steps forward -- so
    the forecast can be checked against prices we already know actually
    happened, instead of just projecting into the unknown future."""
    train, actual = x[:-backtest_days], x[-backtest_days:]
    L = int(round(np.sqrt(len(train))))
    _, k = denoise_series(train, L)
    beta = fit_forecast_model(train, L, k)
    predictions = forecast(train, beta, backtest_days)
    return predictions, actual, L, k


def forecast_errors(predictions: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((predictions - actual) ** 2)))
    mape = float(np.mean(np.abs((predictions - actual) / actual)) * 100)
    return rmse, mape


def plot_forecast(
    dates, prices, denoised, forecast_dates, predictions, ticker, L, k, out_path,
    backtest_dates=None, backtest_predictions=None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, prices, label="Observed close", color="tab:gray", linewidth=1)
    ax.plot(dates[-len(denoised):], denoised, label=f"Denoised (rank k={k})", color="tab:blue")
    if backtest_dates is not None:
        ax.plot(backtest_dates, backtest_predictions, label="Backtest forecast", color="tab:green", linestyle="--", marker="o", markersize=3)
    ax.plot(forecast_dates, predictions, label="Forecast", color="tab:red", linestyle="--", marker="o", markersize=3)
    ax.axvline(dates[-1], color="lightgray", linewidth=0.8)
    ax.set_title(f"{ticker}: Page-matrix SVD denoising + forecast (L={L})")
    ax.set_ylabel("price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--period", default="2y", help="yfinance history window, e.g. 1y, 2y, 5y")
    parser.add_argument("--horizon", type=int, default=20, help="trading days to forecast")
    parser.add_argument(
        "--backtest-days", type=int, default=0,
        help="hold out this many trailing trading days, forecast them from the "
             "preceding history, and plot predicted vs. actual for that window "
             "(e.g. 40 for ~2 months); 0 disables",
    )
    args = parser.parse_args()

    series = download_prices(args.ticker, args.period)
    x = series.values.astype(float)

    L = int(round(np.sqrt(len(x))))  # paper's suggested optimal window: L = sqrt(T) for a single series
    print(f"{args.ticker}: {len(x)} trading days, page length L={L}")

    denoised, k = denoise_series(x, L)
    print(f"Kept top {k} singular directions (>{ENERGY_THRESHOLD:.0%} of spectral energy)")

    beta = fit_forecast_model(x, L, k)
    predictions = forecast(x, beta, args.horizon)

    print(f"\nLast observed close ({series.index[-1].date()}): {x[-1]:.2f}")
    print(f"Forecast for the next {args.horizon} trading days:")
    for i, price in enumerate(predictions, start=1):
        print(f"  t+{i:<3d}: {price:.2f}")

    backtest_dates = backtest_predictions = None
    if args.backtest_days > 0:
        bt_predictions, bt_actual, bt_L, bt_k = backtest(x, args.backtest_days)
        rmse, mape = forecast_errors(bt_predictions, bt_actual)
        print(
            f"\nBacktest: held out the last {args.backtest_days} trading days "
            f"(trained on the preceding {len(x) - args.backtest_days}, L={bt_L}, k={bt_k})"
        )
        print(f"  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
        backtest_dates = series.index[-args.backtest_days:]
        backtest_predictions = bt_predictions

    forecast_dates = pd.bdate_range(series.index[-1], periods=args.horizon + 1)[1:]
    plot_forecast(
        series.index, x, denoised, forecast_dates, predictions,
        args.ticker, L, k, out_path=f"svd_page_matrix_{args.ticker.lower()}.png",
        backtest_dates=backtest_dates, backtest_predictions=backtest_predictions,
    )


if __name__ == "__main__":
    main()

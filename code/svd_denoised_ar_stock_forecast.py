"""Forecast a single stock by fitting an autoregressive model on an
SVD-denoised price series.

A simplified alternative to svd_page_matrix_stock_forecast.py: use the
Page-matrix SVD purely to denoise the price history (same as before), but
then fit the forecasting model as an ordinary autoregression -- v[t] from
v[t-w..t-1] -- by least squares over *overlapping* windows of the
denoised series, instead of the paper's page-wise regression over
non-overlapping pages. This trades the paper's exact machinery for a much
larger effective training set (T-w overlapping windows vs. T/L
non-overlapping pages) and simpler code, at the cost of no longer being
the paper's algorithm. See notes/concepts/svd-denoised-autoregressive-forecasting.md
for the write-up and discussion of this technique this accompanies.

Key design choice: the AR coefficients are fit on the *denoised* series
(so they benefit from noise reduction, same as classical SSA's linear
recurrence forecasting), but the actual forecast is seeded with *raw*
recent prices, not denoised ones -- because the last few points of any
Page-matrix denoising are unreliable (there's no future data to smooth
them with yet), so using denoised values there would feed the model
inputs unlike anything it saw during training.

Usage:
    python svd_denoised_ar_stock_forecast.py [TICKER] [--period 2y] [--horizon 20] [--window 20]
    python svd_denoised_ar_stock_forecast.py [TICKER] --backtest-days 40
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
    of the total squared singular value ("spectral energy")."""
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return int(np.searchsorted(energy, energy_threshold) + 1)


def denoise_series(x: np.ndarray, L: int) -> tuple[np.ndarray, int]:
    """SVD the Page matrix, keep the top-k directions, and read the
    reconstruction back off as a denoised version of the series."""
    P = page_matrix(x, L)
    u, s, vt = np.linalg.svd(P, full_matrices=False)
    k = choose_rank(s)
    P_hat = u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]
    return P_hat.T.reshape(-1), k  # undo page_matrix's reshape/transpose


def fit_ar_model(denoised: np.ndarray, window: int) -> np.ndarray:
    """Ordinary least squares autoregression on the denoised series:
    predict v[t] from v[t-window..t-1], using every overlapping window as
    a training example (far more samples than the paper's one-per-page
    scheme, since consecutive windows share all but one point)."""
    windows = np.lib.stride_tricks.sliding_window_view(denoised, window + 1)
    features, targets = windows[:, :-1], windows[:, -1]
    beta, *_ = np.linalg.lstsq(features, targets, rcond=None)
    return beta


def forecast(x: np.ndarray, beta: np.ndarray, horizon: int) -> np.ndarray:
    """Roll the one-step AR model forward `horizon` times, seeded with
    the *raw* (not denoised) most recent values -- see module docstring
    for why."""
    window_size = len(beta)
    history = list(x[-window_size:])
    predictions = []
    for _ in range(horizon):
        pred = np.dot(history[-window_size:], beta)
        predictions.append(pred)
        history.append(pred)
    return np.array(predictions)


def backtest(x: np.ndarray, backtest_days: int, window: int | None) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Hold out the last `backtest_days` values, denoise and fit the AR
    model on the preceding history only, and forecast that many steps
    forward -- so the forecast can be checked against prices we already
    know actually happened."""
    train, actual = x[:-backtest_days], x[-backtest_days:]
    L = int(round(np.sqrt(len(train))))
    w = window or L
    denoised, k = denoise_series(train, L)
    beta = fit_ar_model(denoised, w)
    predictions = forecast(train, beta, backtest_days)
    return predictions, actual, L, w, k


def forecast_errors(predictions: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((predictions - actual) ** 2)))
    mape = float(np.mean(np.abs((predictions - actual) / actual)) * 100)
    return rmse, mape


def plot_forecast(
    dates, prices, denoised, forecast_dates, predictions, ticker, L, window, k, out_path,
    backtest_dates=None, backtest_predictions=None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, prices, label="Observed close", color="tab:gray", linewidth=1)
    ax.plot(dates[-len(denoised):], denoised, label=f"Denoised (rank k={k})", color="tab:blue")
    if backtest_dates is not None:
        ax.plot(backtest_dates, backtest_predictions, label="Backtest forecast", color="tab:green", linestyle="--", marker="o", markersize=3)
    ax.plot(forecast_dates, predictions, label="Forecast", color="tab:red", linestyle="--", marker="o", markersize=3)
    ax.axvline(dates[-1], color="lightgray", linewidth=0.8)
    ax.set_title(f"{ticker}: AR({window}) on SVD-denoised series (L={L})")
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
    parser.add_argument("--window", type=int, default=None, help="AR window (lags); defaults to round(sqrt(T))")
    parser.add_argument(
        "--backtest-days", type=int, default=0,
        help="hold out this many trailing trading days, forecast them from the "
             "preceding history, and plot predicted vs. actual for that window "
             "(e.g. 40 for ~2 months); 0 disables",
    )
    args = parser.parse_args()

    series = download_prices(args.ticker, args.period)
    x = series.values.astype(float)

    L = int(round(np.sqrt(len(x))))  # Page matrix length, used only for denoising
    window = args.window or L
    print(f"{args.ticker}: {len(x)} trading days, page length L={L}, AR window={window}")

    denoised, k = denoise_series(x, L)
    print(f"Kept top {k} singular directions (>{ENERGY_THRESHOLD:.0%} of spectral energy)")

    beta = fit_ar_model(denoised, window)
    predictions = forecast(x, beta, args.horizon)

    print(f"\nLast observed close ({series.index[-1].date()}): {x[-1]:.2f}")
    print(f"Forecast for the next {args.horizon} trading days:")
    for i, price in enumerate(predictions, start=1):
        print(f"  t+{i:<3d}: {price:.2f}")

    backtest_dates = backtest_predictions = None
    if args.backtest_days > 0:
        bt_predictions, bt_actual, bt_L, bt_w, bt_k = backtest(x, args.backtest_days, args.window)
        rmse, mape = forecast_errors(bt_predictions, bt_actual)
        print(
            f"\nBacktest: held out the last {args.backtest_days} trading days "
            f"(trained on the preceding {len(x) - args.backtest_days}, L={bt_L}, window={bt_w}, k={bt_k})"
        )
        print(f"  RMSE={rmse:.2f}  MAPE={mape:.2f}%")
        backtest_dates = series.index[-args.backtest_days:]
        backtest_predictions = bt_predictions

    forecast_dates = pd.bdate_range(series.index[-1], periods=args.horizon + 1)[1:]
    plot_forecast(
        series.index, x, denoised, forecast_dates, predictions,
        args.ticker, L, window, k, out_path=f"svd_denoised_ar_{args.ticker.lower()}.png",
        backtest_dates=backtest_dates, backtest_predictions=backtest_predictions,
    )


if __name__ == "__main__":
    main()

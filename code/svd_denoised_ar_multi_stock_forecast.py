"""Forecast several stocks jointly by pooling an autoregression fit on
their SVD-denoised, normalized price series.

A simplified alternative to svd_page_matrix_multi_stock_forecast.py: keep
the stacked Page-matrix SVD for *denoising* every stock jointly (so the
shared low-rank structure across stocks still helps smooth each one), but
fit the forecasting model as a single ordinary-least-squares
autoregression -- v[t] from v[t-w..t-1] -- pooling *overlapping* windows
from every stock's denoised series, instead of the paper's page-wise
regression over non-overlapping stacked pages. See
notes/concepts/svd-denoised-autoregressive-forecasting.md for the
write-up and discussion of this technique this accompanies.

As in the single-stock version, the AR coefficients are fit on the
*denoised* series but the actual forecast rollout is seeded with *raw*
recent prices -- see that script's docstring for why.

Usage:
    python svd_denoised_ar_multi_stock_forecast.py [TICKERS...] [--period 2y] [--horizon 20]
    python svd_denoised_ar_multi_stock_forecast.py AAPL MSFT GOOG --backtest-days 40
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
    """Z-score each stock's own price series so pooling doesn't just fit
    whichever stock has the largest scale."""
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
    L x (N * T/L). All series must share the same T."""
    return np.hstack([page_matrix(prices[:, n], L) for n in range(prices.shape[1])])


def choose_rank(singular_values: np.ndarray, energy_threshold: float = ENERGY_THRESHOLD) -> int:
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    return int(np.searchsorted(energy, energy_threshold) + 1)


def denoise_all(prices: np.ndarray, L: int) -> tuple[np.ndarray, int]:
    """De-noise every series at once by SVD-ing the *stacked* Page
    matrix, so the shared low-rank structure across stocks informs each
    one's denoising, not just its own history."""
    stacked = stacked_page_matrix(prices, L)
    u, s, vt = np.linalg.svd(stacked, full_matrices=False)
    k = choose_rank(s)
    stacked_hat = u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]

    per_stock = np.hsplit(stacked_hat, prices.shape[1])
    denoised = np.column_stack([block.T.reshape(-1) for block in per_stock])
    return denoised, k


def fit_ar_model(denoised: np.ndarray, window: int) -> np.ndarray:
    """Ordinary least squares autoregression, pooling overlapping windows
    from every (denoised, normalized) stock's series into one shared fit."""
    all_features, all_targets = [], []
    for n in range(denoised.shape[1]):
        windows = np.lib.stride_tricks.sliding_window_view(denoised[:, n], window + 1)
        all_features.append(windows[:, :-1])
        all_targets.append(windows[:, -1])
    features = np.vstack(all_features)
    targets = np.concatenate(all_targets)
    beta, *_ = np.linalg.lstsq(features, targets, rcond=None)
    return beta


def forecast_all(prices: np.ndarray, beta: np.ndarray, horizon: int) -> np.ndarray:
    """Apply the shared AR model to each (raw-seeded) stock independently,
    rolling forward `horizon` steps."""
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


def backtest(prices: np.ndarray, backtest_days: int, window: int | None) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Hold out the last `backtest_days` rows, denoise and fit on the
    preceding history only (normalized using only that history), and
    forecast that many steps forward for every stock."""
    train, actual = prices[:-backtest_days], prices[-backtest_days:]
    train_norm, mean, std = normalize(train)
    L = int(round(np.sqrt(train.shape[1] * train.shape[0])))
    w = window or L
    denoised_norm, k = denoise_all(train_norm, L)
    beta = fit_ar_model(denoised_norm, w)
    predictions = forecast_all(train_norm, beta, backtest_days) * std + mean
    return predictions, actual, L, w, k


def forecast_errors(predictions: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rmse = np.sqrt(np.mean((predictions - actual) ** 2, axis=0))
    mape = np.mean(np.abs((predictions - actual) / actual), axis=0) * 100
    return rmse, mape


def plot_forecasts(
    dates, prices, denoised, forecast_dates, predictions, tickers, L, window, k, out_path,
    backtest_dates=None, backtest_predictions=None,
):
    fig, axes = plt.subplots(len(tickers), 1, figsize=(10, 3 * len(tickers)), sharex=True)
    axes = np.atleast_1d(axes)
    for i, (ax, ticker) in enumerate(zip(axes, tickers)):
        ax.plot(dates, prices[:, i], label="Observed close", color="tab:gray", linewidth=1)
        ax.plot(dates[-len(denoised):], denoised[:, i], label="Denoised", color="tab:blue")
        if backtest_dates is not None:
            ax.plot(backtest_dates, backtest_predictions[:, i], label="Backtest forecast", color="tab:green", linestyle="--", marker="o", markersize=3)
        ax.plot(forecast_dates, predictions[:, i], label="Forecast", color="tab:red", linestyle="--", marker="o", markersize=3)
        ax.axvline(dates[-1], color="lightgray", linewidth=0.8)
        ax.set_title(ticker)
        ax.set_ylabel("price")
    axes[0].legend()
    fig.suptitle(f"Pooled AR({window}) on stacked SVD-denoised series (L={L}, k={k}, shared model)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--period", default="2y", help="yfinance history window, e.g. 1y, 2y, 5y")
    parser.add_argument("--horizon", type=int, default=20, help="trading days to forecast")
    parser.add_argument("--window", type=int, default=None, help="AR window (lags); defaults to round(sqrt(N*T))")
    parser.add_argument(
        "--backtest-days", type=int, default=0,
        help="hold out this many trailing trading days, forecast them from the "
             "preceding history, and plot predicted vs. actual for that window "
             "(e.g. 40 for ~2 months); 0 disables",
    )
    args = parser.parse_args()

    df = download_prices(args.tickers, args.period)
    prices = df.values.astype(float)
    tickers = list(df.columns)
    prices_norm, mean, std = normalize(prices)

    L = int(round(np.sqrt(len(tickers) * len(prices))))  # Page matrix length, used only for denoising
    window = args.window or L
    print(f"{len(tickers)} tickers x {len(prices)} trading days, page length L={L}, AR window={window}")

    denoised_norm, k = denoise_all(prices_norm, L)
    denoised = denoised_norm * std + mean
    print(f"Kept top {k} singular directions (>{ENERGY_THRESHOLD:.0%} of pooled spectral energy)")

    beta = fit_ar_model(denoised_norm, window)
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

    backtest_dates = backtest_predictions = None
    if args.backtest_days > 0:
        bt_predictions, bt_actual, bt_L, bt_w, bt_k = backtest(prices, args.backtest_days, args.window)
        rmse, mape = forecast_errors(bt_predictions, bt_actual)
        print(
            f"\nBacktest: held out the last {args.backtest_days} trading days "
            f"(trained on the preceding {len(prices) - args.backtest_days}, L={bt_L}, window={bt_w}, k={bt_k})"
        )
        for i, ticker in enumerate(tickers):
            print(f"  {ticker}: RMSE={rmse[i]:.2f}  MAPE={mape[i]:.2f}%")
        backtest_dates = df.index[-args.backtest_days:]
        backtest_predictions = bt_predictions

    forecast_dates = pd.bdate_range(df.index[-1], periods=args.horizon + 1)[1:]
    plot_forecasts(
        df.index, prices, denoised, forecast_dates, predictions,
        tickers, L, window, k, out_path="svd_denoised_ar_multi_stock.png",
        backtest_dates=backtest_dates, backtest_predictions=backtest_predictions,
    )


if __name__ == "__main__":
    main()

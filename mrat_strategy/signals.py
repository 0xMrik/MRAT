"""
signals.py — Compute MRAT values and MAD signals.

Signal construction (Avramov, Kaplansky & Subrahmanyam):
  MRAT   = MA(21) / MA(200)  on adjusted close prices
  sigma  = cross-sectional std of MRAT across all stocks in a given month
  Long   (+1): stock in top decile  AND MRAT > 1 + sigma_multiplier * sigma
  Short  (-1): stock in bot decile  AND MRAT < 1 - sigma_multiplier * sigma
  Neutral (0): everything else

No look-ahead bias: at month-end date t, only data up to and including t
is used. Forward returns are computed in backtest.py over [t, t+1].
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resampling frequency — last calendar day of each month; .last() picks the
# last trading day inside that bucket, so market-closed days produce no NaN.
_RESAMPLE_FREQ = "ME"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_signals(
    prices_df: pd.DataFrame,
    ma_short: int = 21,
    ma_long: int = 200,
    min_price: float = 5.0,
    sigma_multiplier: float = 1.0,
    min_positions: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute monthly MAD signals for every stock.

    Parameters
    ----------
    prices_df       : Daily adjusted close prices (dates × tickers).
    ma_short        : Short moving-average window (default 21 days).
    ma_long         : Long  moving-average window (default 200 days).
    min_price       : Monthly price filter — stocks below this are excluded.
    sigma_multiplier: Multiplier applied to cross-sectional sigma (default 1).
    min_positions   : Minimum required positions on EACH side per month.
                      Months below this threshold are skipped (warning logged).

    Returns
    -------
    signals_df : (months × tickers) DataFrame with values −1 / 0 / +1.
                 NaN means the stock had no valid signal that month
                 (insufficient history or price filter).
    mrat_df    : (months × tickers) DataFrame with raw MRAT values.
    """
    if prices_df.empty:
        raise ValueError("prices_df is empty")

    logger.info("Computing MA(%d) and MA(%d) on daily data…", ma_short, ma_long)

    # Rolling windows — min_periods enforced so early rows are NaN, not biased
    ma_s = prices_df.rolling(ma_short, min_periods=ma_short).mean()
    ma_l = prices_df.rolling(ma_long, min_periods=ma_long).mean()

    # MRAT on a daily frequency
    mrat_daily: pd.DataFrame = ma_s / ma_l

    # Downsample to month-end: take the last observed value in each month
    mrat_monthly: pd.DataFrame = mrat_daily.resample(_RESAMPLE_FREQ).last()
    prices_monthly: pd.DataFrame = prices_df.resample(_RESAMPLE_FREQ).last()

    n_months = len(mrat_monthly)
    logger.info("Processing %d month-end dates…", n_months)

    signals_rows: list[pd.Series] = []
    mrat_rows: list[pd.Series] = []

    for date in mrat_monthly.index:
        mrat_row = mrat_monthly.loc[date]
        price_row = prices_monthly.loc[date]

        # Keep only stocks with valid MRAT and valid price
        valid = mrat_row.notna() & price_row.notna()
        mrat_row = mrat_row[valid]
        price_row = price_row[valid]

        # Dynamic price filter — avoids look-ahead (uses price at date t)
        above_min = price_row >= min_price
        mrat_row = mrat_row[above_min]

        n_stocks = len(mrat_row)
        if n_stocks < 50:
            logger.warning(
                "%s: only %d stocks pass price filter — skipping (need ≥ 50)",
                date.date(),
                n_stocks,
            )
            continue

        # ------------------------------------------------------------------ #
        # Cross-sectional statistics (computed across stocks, not over time)  #
        # ------------------------------------------------------------------ #
        sigma = float(mrat_row.std())

        upper_threshold = 1.0 + sigma_multiplier * sigma
        lower_threshold = 1.0 - sigma_multiplier * sigma

        q90 = float(mrat_row.quantile(0.90))
        q10 = float(mrat_row.quantile(0.10))

        # MAD signal conditions
        long_mask  = (mrat_row >= q90) & (mrat_row > upper_threshold)
        short_mask = (mrat_row <= q10) & (mrat_row < lower_threshold)

        n_long  = int(long_mask.sum())
        n_short = int(short_mask.sum())

        if n_long < min_positions or n_short < min_positions:
            logger.warning(
                "%s: insufficient positions (long=%d, short=%d, min=%d) — skipping",
                date.date(),
                n_long,
                n_short,
                min_positions,
            )
            continue

        signal = pd.Series(0.0, index=mrat_row.index, name=date)
        signal[long_mask]  = 1.0
        signal[short_mask] = -1.0

        signals_rows.append(signal)
        mrat_rows.append(mrat_row.rename(date))

    if not signals_rows:
        raise RuntimeError(
            "No signals generated. Check data quality and parameter ranges."
        )

    signals_df = pd.DataFrame(signals_rows).sort_index()
    mrat_df    = pd.DataFrame(mrat_rows).sort_index()

    avg_long  = (signals_df == 1.0).sum(axis=1).mean()
    avg_short = (signals_df == -1.0).sum(axis=1).mean()
    logger.info(
        "Signals ready — %d months | avg long: %.0f | avg short: %.0f",
        len(signals_df),
        avg_long,
        avg_short,
    )

    return signals_df, mrat_df

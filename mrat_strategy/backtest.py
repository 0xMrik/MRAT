"""
backtest.py — Manual vectorised backtest for the MAD strategy.

Portfolio mechanics:
  - Rebalancing : monthly (at month-end signal date)
  - Weighting   : value-weighted via price × avg-volume as market-cap proxy.
                  Falls back to equal-weight if market-cap data is missing.
  - Long leg    : stocks with signal == +1
  - Short leg   : stocks with signal == -1 (returns are inverted)
  - Long-Short  : combined dollar-neutral portfolio (long + short legs)

No third-party backtest library is used; all logic is implemented with pandas.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RESAMPLE_FREQ = "ME"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_weights(
    signal_row: pd.Series,
    cap_row: pd.Series,
    side: int,
) -> pd.Series:
    """
    Build normalised portfolio weights for one side (+1 long / -1 short).

    Tries value-weighting by market cap proxy; falls back to equal-weight
    with a warning if cap data is unavailable.
    """
    mask = signal_row == side
    tickers = signal_row[mask].index

    if len(tickers) == 0:
        return pd.Series(dtype=float)

    cap = cap_row.reindex(tickers)

    if cap.isna().all() or cap.sum() == 0:
        logger.warning(
            "Market-cap proxy unavailable for %d tickers — using equal-weight",
            len(tickers),
        )
        return pd.Series(1.0 / len(tickers), index=tickers)

    cap = cap.clip(lower=0).fillna(0)
    return cap / cap.sum()


def _one_way_turnover(prev: pd.Series, curr: pd.Series) -> float:
    """
    Compute one-way portfolio turnover between two weight vectors.
    Turnover = 0.5 * sum(|w_new - w_old|).
    """
    all_tickers = prev.index.union(curr.index)
    w_prev = prev.reindex(all_tickers).fillna(0.0)
    w_curr = curr.reindex(all_tickers).fillna(0.0)
    return float((w_curr - w_prev).abs().sum() / 2.0)


def _performance_metrics(ret: pd.Series, rf_monthly: float = 0.0) -> dict:
    """
    Compute standard performance metrics from a monthly return series.

    Returns
    -------
    dict with keys: cagr, sharpe, max_dd, cumulative, returns
    """
    if ret.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan,
                "cumulative": pd.Series(dtype=float), "returns": ret}

    n_months = len(ret)
    n_years  = n_months / 12.0

    total_return = (1.0 + ret).prod()
    cagr         = total_return ** (1.0 / n_years) - 1.0

    excess    = ret - rf_monthly
    sharpe    = excess.mean() / excess.std(ddof=1) * np.sqrt(12) if excess.std(ddof=1) > 0 else np.nan

    cumulative  = (1.0 + ret).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = float(drawdown.min())

    return {
        "cagr":       cagr,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "cumulative": cumulative,
        "returns":    ret,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    prices_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Run the full MAD strategy backtest.

    Parameters
    ----------
    prices_df     : Daily adjusted close prices (dates × tickers).
    volumes_df    : Daily volume (dates × tickers).
    signals_df    : Monthly signals (months × tickers), values −1/0/+1.
    risk_free_rate: Annual risk-free rate (default 0).

    Returns
    -------
    dict containing performance metrics for 'long', 'short', 'long_short',
    plus summary stats (avg positions, avg turnover per leg).
    """
    rf_monthly = risk_free_rate / 12.0

    # Month-end aggregation
    prices_monthly  = prices_df.resample(_RESAMPLE_FREQ).last()
    # Use average daily volume within each month as liquidity proxy
    volumes_monthly = volumes_df.resample(_RESAMPLE_FREQ).mean()

    # Market-cap proxy: price × avg volume
    mktcap_monthly = prices_monthly.multiply(volumes_monthly)

    # Monthly returns: price_t / price_{t-1} - 1
    monthly_ret = prices_monthly.pct_change()

    # ------------------------------------------------------------------ #
    # Main loop — iterate over signal dates                               #
    # ------------------------------------------------------------------ #
    long_returns:  list[tuple[pd.Timestamp, float]] = []
    short_returns: list[tuple[pd.Timestamp, float]] = []
    ls_returns:    list[tuple[pd.Timestamp, float]] = []

    long_positions:  list[int]   = []
    short_positions: list[int]   = []
    long_turnovers:  list[float] = []
    short_turnovers: list[float] = []

    prev_long_w:  pd.Series = pd.Series(dtype=float)
    prev_short_w: pd.Series = pd.Series(dtype=float)

    signal_dates = signals_df.index

    for date in signal_dates:
        # Find the NEXT available month-end in prices (= forward return date)
        pos = prices_monthly.index.searchsorted(date)
        next_pos = pos + 1
        if next_pos >= len(prices_monthly.index):
            # No forward return available for the last signal date
            break
        next_date = prices_monthly.index[next_pos]

        signal_row = signals_df.loc[date].dropna()

        # Market-cap proxy at signal date (no look-ahead)
        if date in mktcap_monthly.index:
            cap_row = mktcap_monthly.loc[date]
        else:
            cap_row = pd.Series(dtype=float)

        # Compute weights for each leg
        long_w  = _compute_weights(signal_row, cap_row, side=+1)
        short_w = _compute_weights(signal_row, cap_row, side=-1)

        # Forward returns (from t to t+1)
        fwd_ret = monthly_ret.loc[next_date]

        # Long leg return
        if not long_w.empty:
            r_long = float((long_w * fwd_ret.reindex(long_w.index).fillna(0.0)).sum())
        else:
            r_long = 0.0

        # Short leg return (inverted: profit when stocks fall)
        if not short_w.empty:
            r_short = float(-(short_w * fwd_ret.reindex(short_w.index).fillna(0.0)).sum())
        else:
            r_short = 0.0

        # Dollar-neutral long-short: each leg is fully funded independently.
        # Combined return = average of both legs (normalised to 1× notional).
        r_ls = (r_long + r_short) / 2.0

        long_returns.append((next_date, r_long))
        short_returns.append((next_date, r_short))
        ls_returns.append((next_date, r_ls))

        long_positions.append(len(long_w))
        short_positions.append(len(short_w))

        # Turnover
        to_long  = _one_way_turnover(prev_long_w, long_w)  if not prev_long_w.empty  else 1.0
        to_short = _one_way_turnover(prev_short_w, short_w) if not prev_short_w.empty else 1.0
        long_turnovers.append(to_long)
        short_turnovers.append(to_short)

        prev_long_w  = long_w
        prev_short_w = short_w

    # ------------------------------------------------------------------ #
    # Assemble return series                                              #
    # ------------------------------------------------------------------ #
    def _to_series(pairs: list[tuple]) -> pd.Series:
        if not pairs:
            return pd.Series(dtype=float)
        idx, vals = zip(*pairs)
        return pd.Series(vals, index=pd.DatetimeIndex(idx))

    long_ret_s  = _to_series(long_returns)
    short_ret_s = _to_series(short_returns)
    ls_ret_s    = _to_series(ls_returns)

    logger.info(
        "Backtest complete — %d monthly observations | "
        "avg long pos: %.0f | avg short pos: %.0f",
        len(long_ret_s),
        np.mean(long_positions) if long_positions else 0,
        np.mean(short_positions) if short_positions else 0,
    )

    return {
        "long":       _performance_metrics(long_ret_s,  rf_monthly),
        "short":      _performance_metrics(short_ret_s, rf_monthly),
        "long_short": _performance_metrics(ls_ret_s,    rf_monthly),
        # Summary stats
        "avg_long_positions":  float(np.mean(long_positions))  if long_positions  else 0.0,
        "avg_short_positions": float(np.mean(short_positions)) if short_positions else 0.0,
        "avg_long_turnover":   float(np.mean(long_turnovers))  if long_turnovers  else 0.0,
        "avg_short_turnover":  float(np.mean(short_turnovers)) if short_turnovers else 0.0,
    }

"""
main.py — Orchestrates data download → signal computation → backtest → output.

All key parameters are configurable below.
"""

import logging
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# CONFIGURABLE PARAMETERS
# ============================================================
MA_SHORT         = 21       # Short moving-average window (days)
MA_LONG          = 200      # Long  moving-average window (days)
MIN_PRICE        = 5.0      # Minimum stock price filter ($)
START_DATE       = "2015-01-01"
END_DATE         = "2024-12-31"
SIGMA_MULTIPLIER = 1.0      # Multiplier on cross-sectional sigma for thresholds
RISK_FREE_RATE   = 0.0      # Annual risk-free rate (0 = no risk-free adjustment)
MIN_POSITIONS    = 20       # Min positions per side to generate a signal
# ============================================================

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SPY benchmark
# ---------------------------------------------------------------------------

def _download_spy(start: str, end: str) -> pd.Series:
    """Download SPY and return a monthly return series."""
    spy_raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    spy_monthly = spy_raw["Close"].resample("ME").last()
    return spy_monthly.pct_change().dropna()


def _spy_metrics(spy_ret: pd.Series, rf_annual: float = 0.0) -> dict:
    rf_monthly  = rf_annual / 12.0
    n_years     = len(spy_ret) / 12.0
    total       = (1.0 + spy_ret).prod()
    cagr        = total ** (1.0 / n_years) - 1.0
    excess      = spy_ret - rf_monthly
    sharpe      = excess.mean() / excess.std(ddof=1) * np.sqrt(12)
    cumulative  = (1.0 + spy_ret).cumprod()
    rolling_max = cumulative.cummax()
    max_dd      = float(((cumulative - rolling_max) / rolling_max).min())
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "cumulative": cumulative}


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _print_summary(
    metrics: dict,
    spy: dict,
    start: str,
    end: str,
    n_universe: int,
) -> None:
    w = 64
    sep = "=" * w

    print()
    print(sep)
    print("       MRAT / MAD Strategy Backtest")
    print(sep)
    print(f"  Période     : {start[:7]} → {end[:7]}")
    print(f"  Univers     : {n_universe} actions (S&P 500 filtré)")
    print(f"  ⚠ Survivorship bias : liste Wikipedia = composition actuelle")
    print()
    print(
        f"  Positions moy.  long  : {metrics['avg_long_positions']:.0f}"
        f"  |  short : {metrics['avg_short_positions']:.0f}"
    )
    print()

    header = f"  {'':16s}  {'CAGR':>8s}  {'Sharpe':>8s}  {'Max DD':>8s}  {'Turnover':>10s}"
    print(header)
    print("  " + "-" * (w - 2))

    def _row(name: str, m: dict, turnover: float | None = None) -> None:
        to_str = f"{turnover:.1%}" if turnover is not None else "  --"
        cagr_s   = f"{m['cagr']:>8.1%}"   if not np.isnan(m.get("cagr",   np.nan)) else "    N/A "
        sharpe_s = f"{m['sharpe']:>8.2f}" if not np.isnan(m.get("sharpe", np.nan)) else "    N/A "
        maxdd_s  = f"{m['max_dd']:>8.1%}" if not np.isnan(m.get("max_dd", np.nan)) else "    N/A "
        print(f"  {name:16s}  {cagr_s}  {sharpe_s}  {maxdd_s}  {to_str:>10s}")

    _row("Long-only",  metrics["long"],       metrics["avg_long_turnover"])
    _row("Short-only", metrics["short"],      metrics["avg_short_turnover"])
    _row("Long-Short", metrics["long_short"])
    _row("SPY B&H",    spy)
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _plot_performance(metrics: dict, spy: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    series = {
        "Long-only":  (metrics["long"]["cumulative"],       "#27ae60", 1.8),
        "Short-only": (metrics["short"]["cumulative"],      "#e74c3c", 1.4),
        "Long-Short": (metrics["long_short"]["cumulative"], "#2980b9", 2.0),
        "SPY B&H":    (spy["cumulative"],                   "#f39c12", 1.6),
    }

    for label, (s, color, lw) in series.items():
        if s is not None and not s.empty:
            ax.plot(s.index, s.values, label=label, color=color, linewidth=lw)

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(
        f"MRAT/MAD Strategy — Cumulative Performance\n"
        f"MA({MA_SHORT}) / MA({MA_LONG})  |  σ×{SIGMA_MULTIPLIER}  |  "
        f"{START_DATE[:7]} → {END_DATE[:7]}",
        fontsize=12,
        pad=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return (1 = initial capital)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Performance chart saved → %s", path)


def _plot_mrat_distribution(
    mrat_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Plots two panels:
      Top : latest month MRAT cross-sectional histogram + sigma thresholds
      Bot : time series of avg MRAT (long, neutral, short) across months
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ---- Left panel: cross-sectional distribution for latest month ---- #
    ax = axes[0]
    last_date = mrat_df.index[-1]
    mrat_row  = mrat_df.loc[last_date].dropna()
    sigma     = float(mrat_row.std())

    ax.hist(mrat_row.values, bins=50, alpha=0.65, color="#3498db",
            edgecolor="white", linewidth=0.4, label="All stocks")

    # Highlight long and short positions
    if last_date in signals_df.index:
        sig = signals_df.loc[last_date]
        for side, color, lbl in [(1, "#e74c3c", "Long"), (-1, "#27ae60", "Short")]:
            tks = sig[sig == side].index
            vals = mrat_row.reindex(tks).dropna().values
            if len(vals):
                ax.hist(vals, bins=20, alpha=0.75, color=color,
                        edgecolor="white", linewidth=0.4, label=f"{lbl} ({len(vals)})")

    for x, color, lbl in [
        (1.0,                              "black",   "MRAT = 1"),
        (1.0 + SIGMA_MULTIPLIER * sigma,   "#c0392b", f"1 + {SIGMA_MULTIPLIER}σ"),
        (1.0 - SIGMA_MULTIPLIER * sigma,   "#1e8449", f"1 − {SIGMA_MULTIPLIER}σ"),
    ]:
        ax.axvline(x, color=color, linestyle="--", linewidth=1.5, label=lbl)

    ax.set_title(f"MRAT distribution — {last_date.date()}", fontsize=11, pad=10)
    ax.set_xlabel("MRAT  =  MA(21) / MA(200)")
    ax.set_ylabel("Nombre d'actions")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # ---- Right panel: evolution of avg MRAT per bucket over time ---- #
    ax2 = axes[1]

    avg_long    = []
    avg_short   = []
    avg_neutral = []

    for date in mrat_df.index:
        mrat_row = mrat_df.loc[date].dropna()
        if date not in signals_df.index:
            continue
        sig = signals_df.loc[date]
        for bucket, lst in [(1, avg_long), (-1, avg_short), (0, avg_neutral)]:
            tks  = sig[sig == bucket].index.intersection(mrat_row.index)
            vals = mrat_row.reindex(tks).dropna()
            lst.append((date, vals.mean() if not vals.empty else np.nan))

    for pairs, color, label in [
        (avg_long,    "#e74c3c", "Avg MRAT — Long"),
        (avg_short,   "#27ae60", "Avg MRAT — Short"),
        (avg_neutral, "#95a5a6", "Avg MRAT — Neutral"),
    ]:
        if pairs:
            idx, vals = zip(*pairs)
            ax2.plot(idx, vals, color=color, linewidth=1.4, label=label)

    ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_title("Average MRAT by portfolio bucket", fontsize=11, pad=10)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Average MRAT")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.suptitle(
        f"MRAT Cross-Sectional Analysis  |  σ×{SIGMA_MULTIPLIER}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("MRAT distribution chart saved → %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from data import download_prices, get_sp500_tickers
    from signals import compute_signals
    from backtest import run_backtest

    logger.info("=" * 60)
    logger.info("MRAT/MAD Strategy — pipeline start")
    logger.info("  MA_SHORT=%d  MA_LONG=%d  MIN_PRICE=$%.0f", MA_SHORT, MA_LONG, MIN_PRICE)
    logger.info("  Period: %s → %s", START_DATE, END_DATE)
    logger.info("  SIGMA_MULTIPLIER=%.1f  MIN_POSITIONS=%d", SIGMA_MULTIPLIER, MIN_POSITIONS)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Data
    # ------------------------------------------------------------------
    logger.info("[1/3] Fetching price data…")
    tickers = get_sp500_tickers()
    prices_df, volumes_df = download_prices(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        min_price=MIN_PRICE,
    )
    n_universe = len(prices_df.columns)
    logger.info("Universe: %d stocks × %d trading days", n_universe, len(prices_df))

    # ------------------------------------------------------------------
    # Step 2 — Signals
    # ------------------------------------------------------------------
    logger.info("[2/3] Computing MRAT signals…")
    signals_df, mrat_df = compute_signals(
        prices_df=prices_df,
        ma_short=MA_SHORT,
        ma_long=MA_LONG,
        min_price=MIN_PRICE,
        sigma_multiplier=SIGMA_MULTIPLIER,
        min_positions=MIN_POSITIONS,
    )

    # ------------------------------------------------------------------
    # Step 3 — Backtest
    # ------------------------------------------------------------------
    logger.info("[3/3] Running backtest…")
    metrics = run_backtest(
        prices_df=prices_df,
        volumes_df=volumes_df,
        signals_df=signals_df,
        risk_free_rate=RISK_FREE_RATE,
    )

    logger.info("Downloading SPY benchmark…")
    spy_ret = _download_spy(START_DATE, END_DATE)
    spy     = _spy_metrics(spy_ret, RISK_FREE_RATE)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    _print_summary(metrics, spy, START_DATE, END_DATE, n_universe)

    _plot_performance(
        metrics, spy,
        OUTPUT_DIR / "portfolio_performance.png",
    )
    _plot_mrat_distribution(
        mrat_df, signals_df,
        OUTPUT_DIR / "mrat_distribution.png",
    )

    logger.info("All done. Results in %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()

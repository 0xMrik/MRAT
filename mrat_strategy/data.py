"""
data.py — Download and cache S&P 500 adjusted prices.

NOTE: This module uses the CURRENT S&P 500 composition from Wikipedia,
which introduces survivorship bias. Historical index composition data
requires paid data sources (e.g. CRSP, Compustat). All results should
be interpreted with this caveat in mind.
"""

import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "output"
PRICES_PATH = CACHE_DIR / "prices.parquet"
VOLUMES_PATH = CACHE_DIR / "volumes.parquet"

BATCH_SIZE = 50
BATCH_DELAY = 2  # seconds between batches to avoid yfinance throttling


# ---------------------------------------------------------------------------
# Ticker discovery
# ---------------------------------------------------------------------------

def get_sp500_tickers() -> list[str]:
    """
    Fetch current S&P 500 tickers from Wikipedia.

    ⚠ Survivorship bias: returns the CURRENT list, not the historical one.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
    return tickers


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_batch(
    batch: list[str], start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download Close and Volume for a list of tickers via yfinance."""
    data = yf.download(
        batch,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        volume = data["Volume"]
    else:
        # Single-ticker download — wrap in DataFrame
        close = data[["Close"]].rename(columns={"Close": batch[0]})
        volume = data[["Volume"]].rename(columns={"Volume": batch[0]})

    return close, volume


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    min_price: float = 5.0,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download (or load from cache) adjusted Close prices and Volume.

    Parameters
    ----------
    tickers : List of ticker symbols.
    start   : Start date string, e.g. "2015-01-01".
    end     : End date string, e.g. "2024-12-31".
    min_price : Tickers whose MEDIAN price over the period is below this
                threshold are dropped (rough pre-filter; dynamic per-month
                filtering is applied in signals.py).
    force_download : Ignore cache and re-download.

    Returns
    -------
    prices_df  : DataFrame (dates × tickers) of adjusted close prices.
    volumes_df : DataFrame (dates × tickers) of volume.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_download and PRICES_PATH.exists() and VOLUMES_PATH.exists():
        logger.info("Loading prices from cache (%s)", PRICES_PATH)
        prices = pd.read_parquet(PRICES_PATH)
        volumes = pd.read_parquet(VOLUMES_PATH)
        logger.info("Cache loaded: %d tickers, %d days", len(prices.columns), len(prices))
        return prices, volumes

    all_close: list[pd.DataFrame] = []
    all_volume: list[pd.DataFrame] = []
    n_batches = (len(tickers) - 1) // BATCH_SIZE + 1

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info(
            "Batch %d/%d — downloading %d tickers (%s … %s)",
            batch_num,
            n_batches,
            len(batch),
            batch[0],
            batch[-1],
        )

        try:
            close, volume = _download_batch(batch, start, end)
            if not close.empty:
                all_close.append(close)
                all_volume.append(volume)
            else:
                logger.warning("Batch %d returned empty data", batch_num)
        except Exception as exc:
            logger.warning("Batch %d failed: %s", batch_num, exc)

        if i + BATCH_SIZE < len(tickers):
            time.sleep(BATCH_DELAY)

    if not all_close:
        raise RuntimeError("No data downloaded — check network connection or ticker list.")

    prices = pd.concat(all_close, axis=1)
    volumes = pd.concat(all_volume, axis=1)

    # Remove duplicate columns (can occur if a ticker appears twice in the list)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    volumes = volumes.loc[:, ~volumes.columns.duplicated()]

    # Align both DataFrames to the same columns
    common = prices.columns.intersection(volumes.columns)
    prices = prices[common]
    volumes = volumes[common]

    logger.info("Raw download: %d tickers over %d trading days", len(prices.columns), len(prices))

    # Pre-filter: drop tickers whose median price is below threshold
    median_prices = prices.median()
    valid = median_prices[median_prices >= min_price].index
    prices = prices[valid]
    volumes = volumes[valid]
    dropped = len(common) - len(valid)
    logger.info(
        "After median-price filter (>= $%.0f): %d tickers retained, %d dropped",
        min_price,
        len(prices.columns),
        dropped,
    )

    prices.to_parquet(PRICES_PATH)
    volumes.to_parquet(VOLUMES_PATH)
    logger.info("Saved to %s and %s", PRICES_PATH, VOLUMES_PATH)

    return prices, volumes

"""
generate_synthetic_data.py — Génère des données synthétiques réalistes
pour tester la pipeline MRAT/MAD sans connexion internet.

Les prix suivent un mouvement brownien géométrique avec :
  - rendements corrélés via un facteur de marché commun (réalisme cross-sectionnel)
  - retours à la moyenne par rapport à un trend long terme (régimes haussiers/baissiers)
  - volatilités hétérogènes entre actions (sigma entre 15% et 45% annualisé)

⚠ Ces données sont purement fictives — les résultats ne reflètent PAS la
   performance réelle de la stratégie MAD sur le S&P 500.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PRICES_PATH = OUTPUT_DIR / "prices.parquet"
VOLUMES_PATH = OUTPUT_DIR / "volumes.parquet"

# ------------------------------------------------------------------ #
# Paramètres de simulation                                            #
# ------------------------------------------------------------------ #
N_STOCKS    = 480        # ~S&P 500 filtré
START_DATE  = "2015-01-01"
END_DATE    = "2024-12-31"
SEED        = 42

# Distribution des prix initiaux (~réaliste pour le S&P 500)
PRICE_MEAN  = 80.0
PRICE_STD   = 60.0
PRICE_MIN   = 10.0

# Paramètres du mouvement brownien géométrique
MKT_DRIFT   = 0.12       # drift annuel du facteur de marché (12 %)
MKT_VOL     = 0.16       # vol annuelle du facteur de marché

# Paramètres idiosyncratiques par action
ALPHA_MEAN  = 0.0        # drift idiosyncratique moyen (neutre)
ALPHA_STD   = 0.04       # dispersion des alphas individuels (±4% ann.)
IDIO_VOL_MIN = 0.15      # vol idiosyncratique min (annualisée)
IDIO_VOL_MAX = 0.45      # vol idiosyncratique max (annualisée)
BETA_MIN    = 0.5        # bêta de marché min
BETA_MAX    = 1.8        # bêta de marché max

# Factor de corrélation intra-secteur (5 secteurs simulés)
N_SECTORS   = 11
SECTOR_VOL  = 0.08


def generate(
    n_stocks: int = N_STOCKS,
    start: str = START_DATE,
    end: str = END_DATE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    dt = 1 / 252

    tickers = [f"SIM{i:04d}" for i in range(n_stocks)]

    # ---------------------------------------------------------------- #
    # Facteur de marché commun                                          #
    # ---------------------------------------------------------------- #
    mkt_shocks = rng.normal(0, 1, size=T)
    mkt_returns = (MKT_DRIFT - 0.5 * MKT_VOL**2) * dt + MKT_VOL * np.sqrt(dt) * mkt_shocks
    mkt_cumret = np.cumprod(1 + mkt_returns)

    # ---------------------------------------------------------------- #
    # Facteurs sectoriels (11 secteurs GICS)                            #
    # ---------------------------------------------------------------- #
    sector_ids = rng.integers(0, N_SECTORS, size=n_stocks)
    sector_shocks = rng.normal(0, 1, size=(T, N_SECTORS))
    sector_returns = SECTOR_VOL * np.sqrt(dt) * sector_shocks

    # ---------------------------------------------------------------- #
    # Paramètres par action                                             #
    # ---------------------------------------------------------------- #
    betas  = rng.uniform(BETA_MIN, BETA_MAX, size=n_stocks)
    alphas = rng.normal(ALPHA_MEAN, ALPHA_STD, size=n_stocks) * dt  # daily
    idio_vols = rng.uniform(IDIO_VOL_MIN, IDIO_VOL_MAX, size=n_stocks) * np.sqrt(dt)

    # Prix initiaux (log-normal, min clippé)
    p0 = np.clip(
        rng.lognormal(mean=np.log(PRICE_MEAN), sigma=0.8, size=n_stocks),
        PRICE_MIN,
        None,
    )

    # ---------------------------------------------------------------- #
    # Simulation des prix                                               #
    # ---------------------------------------------------------------- #
    # log_prices[t, i] = log(P_t / P_0)
    log_prices = np.zeros((T, n_stocks))
    idio_shocks = rng.normal(0, 1, size=(T, n_stocks))

    for t in range(1, T):
        mkt_contrib  = betas * mkt_returns[t]
        sect_contrib = sector_returns[t, sector_ids]
        idio_contrib = alphas + idio_vols * idio_shocks[t]
        log_prices[t] = log_prices[t - 1] + mkt_contrib + sect_contrib + idio_contrib

    prices_arr = p0[np.newaxis, :] * np.exp(log_prices)

    # ---------------------------------------------------------------- #
    # Simulation des volumes (log-normal, corrélé à la vol)            #
    # ---------------------------------------------------------------- #
    base_volumes = rng.lognormal(mean=np.log(2_000_000), sigma=1.0, size=n_stocks)
    vol_factor   = np.abs(np.diff(prices_arr, axis=0, prepend=prices_arr[:1]) / prices_arr)
    volume_arr   = (
        base_volumes[np.newaxis, :]
        * (1 + 5 * vol_factor)
        * rng.lognormal(0, 0.3, size=(T, n_stocks))
    )

    prices_df  = pd.DataFrame(prices_arr,  index=dates, columns=tickers)
    volumes_df = pd.DataFrame(volume_arr,  index=dates, columns=tickers)

    return prices_df, volumes_df


def main() -> None:
    print("=" * 55)
    print("  Génération de données synthétiques MRAT")
    print("=" * 55)
    print(f"  Période   : {START_DATE} → {END_DATE}")
    print(f"  Actions   : {N_STOCKS}")
    print(f"  Graine    : {SEED}")
    print()

    prices, volumes = generate()

    prices.to_parquet(PRICES_PATH)
    volumes.to_parquet(VOLUMES_PATH)

    print(f"  ✓ prices.parquet  → {PRICES_PATH}  ({prices.shape})")
    print(f"  ✓ volumes.parquet → {VOLUMES_PATH}  ({volumes.shape})")
    print()
    print("  Prix finaux (5 actions) :")
    print(prices.iloc[-1, :5].to_string())
    print()
    print("  ⚠  Données fictives — pour test uniquement.")
    print("=" * 55)


if __name__ == "__main__":
    main()

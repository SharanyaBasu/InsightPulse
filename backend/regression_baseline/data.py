"""
Data layer for the baseline regression model.

Fetches raw market data (yfinance) and macro data (FRED), caches both to
parquet, and exposes a single function `load_monthly_panel()` that returns
an aligned monthly panel of LEVELS (prices, rates, indicators).

This module is intentionally ignorant of features/targets/returns. All
transformations live in pipeline.py. If you change the ticker list or FRED
series dict below, delete cache/ to force a re-fetch.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv


CACHE_DIR = Path(__file__).parent / "cache"
MARKET_CACHE = CACHE_DIR / "market_monthly.parquet"
MACRO_CACHE = CACHE_DIR / "macro_monthly.parquet"
CACHE_TTL_HOURS = 24

YEARS_OF_HISTORY = 10

YFINANCE_TICKERS = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "gold": "GC=F",
    "oil": "CL=F",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
}

FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "gdp": "GDPC1",
    "unrate": "UNRATE",
    "cfnai": "CFNAI",
    "dgs10": "DGS10",
}


def _is_cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def fetch_market_data(years: int = YEARS_OF_HISTORY) -> pd.DataFrame:
    if _is_cache_fresh(MARKET_CACHE):
        print(f"Using cached market data ({MARKET_CACHE.name})")
        return pd.read_parquet(MARKET_CACHE)

    end = datetime.now()
    start = end - timedelta(days=365 * years)

    series_dict = {}
    for name, ticker in YFINANCE_TICKERS.items():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty or "Close" not in df.columns:
            print(f"  yfinance {name} ({ticker}): FAILED")
            continue
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        series_dict[name] = close.dropna()
        print(f"  yfinance {name} ({ticker}): {len(series_dict[name])} daily obs")

    if not series_dict:
        raise RuntimeError("No market data fetched.")

    daily = pd.concat(series_dict, axis=1)
    daily.columns = list(series_dict.keys())
    monthly = daily.resample("ME").last()

    CACHE_DIR.mkdir(exist_ok=True)
    monthly.to_parquet(MARKET_CACHE)
    print(f"Cached market data -> {MARKET_CACHE.name} ({len(monthly)} months)")
    return monthly


def fetch_macro_data(years: int = YEARS_OF_HISTORY) -> pd.DataFrame:
    if _is_cache_fresh(MACRO_CACHE):
        print(f"Using cached macro data ({MACRO_CACHE.name})")
        return pd.read_parquet(MACRO_CACHE)

    load_dotenv()
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        raise RuntimeError("FRED_API_KEY not found. Add it to backend/.env")

    fred = Fred(api_key=fred_key)
    end = datetime.now()
    # Pull an extra year so that quarterly GDP has enough history to ffill
    # cleanly across the start of the panel.
    start = end - timedelta(days=365 * (years + 1))

    series_dict = {}
    for name, code in FRED_SERIES.items():
        s = fred.get_series(code, observation_start=start, observation_end=end)
        series_dict[name] = pd.to_numeric(s, errors="coerce")
        print(f"  FRED {name} ({code}): {len(s)} obs")

    raw = pd.DataFrame(series_dict)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    # Resample to month-end, then forward-fill so quarterly GDP propagates
    # into the intervening months.
    monthly = raw.resample("ME").last().ffill()

    CACHE_DIR.mkdir(exist_ok=True)
    monthly.to_parquet(MACRO_CACHE)
    print(f"Cached macro data -> {MACRO_CACHE.name} ({len(monthly)} months)")
    return monthly


def load_monthly_panel() -> pd.DataFrame:
    """
    Returns a monthly panel of levels with these columns:
      market: sp500, nasdaq, gold, oil, dxy, vix
      macro:  fed_funds, cpi, gdp, unrate, cfnai, dgs10
    Index is month-end (pd.DatetimeIndex). Rows with any missing value dropped.
    """
    market = fetch_market_data()
    macro = fetch_macro_data()
    panel = market.join(macro, how="inner").dropna()
    print(f"Joined panel: market {len(market)} + macro {len(macro)} -> {len(panel)} aligned months")
    return panel


if __name__ == "__main__":
    panel = load_monthly_panel()
    print(f"\nMonthly panel shape: {panel.shape}")
    print(f"Date range: {panel.index.min().date()} -> {panel.index.max().date()}")
    print(f"Columns: {list(panel.columns)}")
    print("\nLast 5 rows:")
    print(panel.tail())

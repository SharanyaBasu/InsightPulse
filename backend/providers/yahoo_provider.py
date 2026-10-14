"""Yahoo Finance provider for market data fetch and cleaning."""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from threading import Lock

_DOWNLOAD_LOCK = Lock()


def download_history(symbol, **kwargs):
    """Serialize Yahoo downloads because their shared result cache is not request-local."""
    with _DOWNLOAD_LOCK:
        return yf.download(symbol, threads=False, progress=False, timeout=15, **kwargs)


def extract_close(frame, symbol):
    """Select exactly the requested ticker from flat or multi-ticker Close columns."""
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        if symbol not in close.columns:
            raise ValueError(f"Yahoo response is missing Close for {symbol}")
        close = close[symbol]
    if not isinstance(close, pd.Series):
        raise ValueError(f"Yahoo response has ambiguous Close columns for {symbol}")
    return pd.to_numeric(close, errors="coerce").dropna()



def fetch_market_data():
    tickers = {
        # --- Equities ---
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
        "VGK": "VGK",
        "EWJ": "EWJ",
        "EEM": "EEM",
        "MTUM": "MTUM",
        "VTV": "VTV",
        "IWF": "IWF",

        # --- Sector ETFs (NEW) ---
        "XLY": "XLY",
        "XLP": "XLP",
        "XLE": "XLE",
        "XLF": "XLF",
        "XLV": "XLV",
        "XLK": "XLK",
        "XLI": "XLI",
        "XLB": "XLB",
        "XLRE": "XLRE",
        "XLC": "XLC",

        # --- Bonds / Credit ---
        "IRX": "^IRX",
        "FVX": "^FVX",
        "TNX": "^TNX",
        "HYG": "HYG",
        "LQD": "LQD",
        "TLT": "TLT",
        "VIX": "^VIX",

        # --- Commodities ---
        "Oil": "CL=F",
        "NatGas": "NG=F",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",

        # --- FX ---
        "USD_Index": "DX-Y.NYB",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "AUDUSD": "AUDUSD=X",
        "USDJPY": "USDJPY=X",
        "USDCHF": "USDCHF=X",
        "CEW": "CEW",

        # --- Crypto ---
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
    }

    end = datetime.now()
    start = end - timedelta(days=365)
    data = {}

    for name, symbol in tickers.items():
        try:
            df = download_history(symbol, start=start, end=end, interval="1d")

            # Ensure we have a valid DataFrame and a Close column.
            if isinstance(df, pd.DataFrame) and "Close" in df.columns and not df["Close"].empty:
                series = extract_close(df, symbol)
                if len(series) > 5:
                    data[name] = series
                    print(f"Loaded {name} ({len(series)} points)")
                else:
                    print(f"Skipped {name}: not enough data points")
            else:
                print(f"Skipped {name}: invalid or missing Close column")

        except Exception as e:
            print(f"Error loading {name}: {e}")

    if not data:
        raise RuntimeError("No valid data fetched for any ticker.")

    # Combine all valid Series safely.
    df = pd.concat(data, axis=1, sort=False)
    df.dropna(inplace=True)
    if df.empty:
        raise RuntimeError("No overlapping valid market observations were returned.")

    print(f"\nFinal dataframe shape: {df.shape}")
    print(df.tail())

    return df

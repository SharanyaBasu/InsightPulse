"""FRED provider for macroeconomic data fetch and cleaning."""
import time

import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


def fetch_macro_data():
    """
    Fetch key macroeconomic indicators from the Federal Reserve (FRED).
    Returns a pandas DataFrame with monthly/quarterly data.
    """
    load_dotenv()  # loads .env variables
    FRED_API_KEY = os.getenv("FRED_API_KEY")

    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not found. Set it in your .env file.")

    fred = Fred(api_key=FRED_API_KEY)

    # --- Add DGS2 and DGS10 for yields ---
    series = {
        "CPI": "CPIAUCSL",
        "Unemployment": "UNRATE",
        "Fed_Funds_Rate": "FEDFUNDS",
        "GDP": "GDPC1",
        "DGS2": "DGS2",
        "DGS10": "DGS10",
    }

    data = {}
    for name, code in series.items():
        try:
            df = fred.get_series(code)
            data[name] = df
            print(f"Loaded macro: {name}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error loading {name}: {e}")

    missing = [name for name in series if name not in data]
    if missing:
        raise RuntimeError(f"Missing required FRED series: {', '.join(missing)}")

    macro_df = pd.DataFrame(data)
    macro_df.index = pd.to_datetime(macro_df.index)

    # Convert yields to float and forward fill missing business days.
    macro_df["DGS2"] = pd.to_numeric(macro_df["DGS2"], errors="coerce")
    macro_df["DGS10"] = pd.to_numeric(macro_df["DGS10"], errors="coerce")

    macro_df = macro_df.sort_index().ffill().dropna()

    return macro_df

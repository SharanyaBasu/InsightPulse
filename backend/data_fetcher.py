import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv
import os


def get_market_data():
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
            df = yf.download(symbol, start=start, end=end, interval="1d")

            # Ensure we have a valid DataFrame and a Close column
            if isinstance(df, pd.DataFrame) and "Close" in df.columns and not df["Close"].empty:
                series = df["Close"].dropna()
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

    # Combine all valid Series safely
    df = pd.concat(data, axis=1)
    df.columns = list(data.keys())
    df.dropna(inplace=True)

    print(f"\nFinal dataframe shape: {df.shape}")
    print(df.tail())

    save_to_db(df)

    return df


from db import SessionLocal
from models import MarketData

# Helper to force numeric values into native Python floats for DB inserts.
# Why: Postgres/psycopg2 may reject NumPy scalar types that can leak from pandas rows.
def _to_native_scalar(value):
    # Convert pandas missing values (NaN/NaT) to None so SQLAlchemy writes SQL NULL.
    if pd.isna(value):
        return None

    # Cast every non-missing numeric value to a native Python float.
    # Why: this guarantees ORM bind params are float/None (not np.float64).
    return float(value)

def save_to_db(df):
    session = SessionLocal()
    session.query(MarketData).delete()  # optional: clears old data

    for date, row in df.iterrows():
        # Build a plain Python dict of normalized values for this row.
        # Why: converting to dict prevents pandas from re-coercing values back to NumPy dtypes.
        normalized = {col: _to_native_scalar(row[col]) for col in df.columns}

        entry = MarketData(
            date=date.date(),
            # --- Equities ---
            sp500=normalized.get("SP500"),
            nasdaq=normalized.get("NASDAQ"),
            vgk=normalized.get("VGK"),
            ewj=normalized.get("EWJ"),
            eem=normalized.get("EEM"),
            mtum=normalized.get("MTUM"),
            vtv=normalized.get("VTV"),
            iwf=normalized.get("IWF"),

            # --- Sector ETFs (NEW) ---
            xly=normalized.get("XLY"),
            xlp=normalized.get("XLP"),
            xle=normalized.get("XLE"),
            xlf=normalized.get("XLF"),
            xlv=normalized.get("XLV"),
            xlk=normalized.get("XLK"),
            xli=normalized.get("XLI"),
            xlb=normalized.get("XLB"),
            xlre=normalized.get("XLRE"),
            xlc=normalized.get("XLC"),

            # --- Bonds / Credit ---
            irx=normalized.get("IRX"),
            fvx=normalized.get("FVX"),
            tnx=normalized.get("TNX"),
            hyg=normalized.get("HYG"),
            lqd=normalized.get("LQD"),
            tlt=normalized.get("TLT"),
            vix=normalized.get("VIX"),

            # --- Commodities ---
            oil=normalized.get("Oil"),
            natgas=normalized.get("NatGas"),
            gold=normalized.get("Gold"),
            silver=normalized.get("Silver"),
            copper=normalized.get("Copper"),

            # --- FX ---
            usd_index=normalized.get("USD_Index"),
            eurusd=normalized.get("EURUSD"),
            gbpusd=normalized.get("GBPUSD"),
            audusd=normalized.get("AUDUSD"),
            usdjpy=normalized.get("USDJPY"),
            usdchf=normalized.get("USDCHF"),
            cew=normalized.get("CEW"),

            # --- Crypto ---
            bitcoin=normalized.get("Bitcoin"),
            ethereum=normalized.get("Ethereum"),
        )
        session.add(entry)

    session.commit()
    session.close()
    print(f"Saved {len(df)} rows to the database.")


from db import SessionLocal
from models import MacroData

def get_macro_data():
    """
    Fetch key macroeconomic indicators from the Federal Reserve (FRED).
    Returns a pandas DataFrame with monthly/quarterly data and saves to DB.
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
        "DGS10": "DGS10"
    }

    data = {}
    for name, code in series.items():
        try:
            df = fred.get_series(code)
            data[name] = df
            print(f"Loaded macro: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

    macro_df = pd.DataFrame(data)
    macro_df.index = pd.to_datetime(macro_df.index)

    # Convert yields to float and forward fill missing business days
    macro_df["DGS2"] = pd.to_numeric(macro_df["DGS2"], errors="coerce")
    macro_df["DGS10"] = pd.to_numeric(macro_df["DGS10"], errors="coerce")

    macro_df = macro_df.sort_index().ffill().dropna()

    # Save to DB
    session = SessionLocal()
    session.query(MacroData).delete()  # clear old

    for date, row in macro_df.iterrows():
        # Normalize macro row values into plain Python float/None before ORM construction.
        normalized = {col: _to_native_scalar(row[col]) for col in macro_df.columns}

        entry = MacroData(
            date=date.date(),
            cpi=normalized["CPI"],
            unemployment=normalized["Unemployment"],
            fed_funds_rate=normalized["Fed_Funds_Rate"],
            gdp=normalized["GDP"],
            two_year_yield=normalized["DGS2"],
            ten_year_yield=normalized["DGS10"],
        )
        session.add(entry)

    session.commit()
    session.close()

    print(f"Saved {len(macro_df)} macro rows to database.")
    return macro_df



if __name__ == "__main__":
    get_market_data()
    get_macro_data()


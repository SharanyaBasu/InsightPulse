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

def save_to_db(df):
    session = SessionLocal()
    session.query(MarketData).delete()  # optional: clears old data

    for date, row in df.iterrows():
        entry = MarketData(
            date=date.date(),
            # --- Equities ---
            sp500=row.get("SP500"),
            nasdaq=row.get("NASDAQ"),
            vgk=row.get("VGK"),
            ewj=row.get("EWJ"),
            eem=row.get("EEM"),
            mtum=row.get("MTUM"),
            vtv=row.get("VTV"),
            iwf=row.get("IWF"),

            # --- Sector ETFs (NEW) ---
            xly=row.get("XLY"),
            xlp=row.get("XLP"),
            xle=row.get("XLE"),
            xlf=row.get("XLF"),
            xlv=row.get("XLV"),
            xlk=row.get("XLK"),
            xli=row.get("XLI"),
            xlb=row.get("XLB"),
            xlre=row.get("XLRE"),
            xlc=row.get("XLC"),

            # --- Bonds / Credit ---
            irx=row.get("IRX"),
            fvx=row.get("FVX"),
            tnx=row.get("TNX"),
            hyg=row.get("HYG"),
            lqd=row.get("LQD"),
            tlt=row.get("TLT"),
            vix=row.get("VIX"),

            # --- Commodities ---
            oil=row.get("Oil"),
            natgas=row.get("NatGas"),
            gold=row.get("Gold"),
            silver=row.get("Silver"),
            copper=row.get("Copper"),

            # --- FX ---
            usd_index=row.get("USD_Index"),
            eurusd=row.get("EURUSD"),
            gbpusd=row.get("GBPUSD"),
            audusd=row.get("AUDUSD"),
            usdjpy=row.get("USDJPY"),
            usdchf=row.get("USDCHF"),
            cew=row.get("CEW"),

            # --- Crypto ---
            bitcoin=row.get("Bitcoin"),
            ethereum=row.get("Ethereum"),
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
        entry = MacroData(
            date=date.date(),
            cpi=row["CPI"],
            unemployment=row["Unemployment"],
            fed_funds_rate=row["Fed_Funds_Rate"],
            gdp=row["GDP"],
            two_year_yield=row["DGS2"],
            ten_year_yield=row["DGS10"],
        )
        session.add(entry)

    session.commit()
    session.close()

    print(f"Saved {len(macro_df)} macro rows to database.")
    return macro_df



if __name__ == "__main__":
    get_market_data()
    get_macro_data()


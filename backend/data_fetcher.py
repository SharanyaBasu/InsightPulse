import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv
import os


def get_market_data():
    tickers = {
    # Equities
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",

    # Commodities
    "Gold": "GC=F",
    "Oil": "CL=F",
    "Copper": "HG=F",

    # Forex (FX)
    "USD_Index": "DX-Y.NYB",     
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",

    # Fixed Income / Bonds
    "10Y_Yield": "^TNX",        
    "2Y_Yield": "^FVX",         
    "3M_Yield": "^IRX",          

    # Crypto
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
            sp500=row.get("SP500"),
            nasdaq=row.get("NASDAQ"),
            gold=row.get("Gold"),
            oil=row.get("Oil"),
            copper=row.get("Copper"),
            usd_index=row.get("USD_Index"),
            eurusd=row.get("EURUSD"),
            usdjpy=row.get("USDJPY"),
            yield_10y=row.get("10Y_Yield"),
            yield_2y=row.get("2Y_Yield"),
            yield_3m=row.get("3M_Yield"),
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
    load_dotenv()  # ✅ loads .env variables
    FRED_API_KEY = os.getenv("FRED_API_KEY")

    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not found. Set it in your .env file.")

    fred = Fred(api_key=FRED_API_KEY)

    series = {
        "CPI": "CPIAUCSL",           # Inflation
        "Unemployment": "UNRATE",    # %
        "Fed_Funds_Rate": "FEDFUNDS",
        "GDP": "GDPC1"               # Real GDP
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
    macro_df = macro_df.dropna()

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
        )
        session.add(entry)
    session.commit()
    session.close()

    print(f"Saved {len(macro_df)} macro rows to database.")
    return macro_df



if __name__ == "__main__":
    get_market_data()
    get_macro_data()


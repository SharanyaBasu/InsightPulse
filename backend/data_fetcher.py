import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def get_market_data():
    tickers = {
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Gold": "GC=F",
        "Oil": "CL=F",
        "USD_Index": "DX-Y.NYB",
        "10Y_Yield": "^TNX"
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

    return df

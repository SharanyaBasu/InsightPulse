import pandas as pd
from sqlalchemy.orm import Session
from db import SessionLocal
from models import MarketData

# ---------------------------------------------------------------------
# STEP 1: Load market data from DB
# ---------------------------------------------------------------------
def load_market_data():
    """Fetches market data as a pandas DataFrame."""
    db: Session = SessionLocal()
    data = pd.read_sql("SELECT * FROM market_data ORDER BY date", db.bind)
    db.close()
    data["date"] = pd.to_datetime(data["date"])
    data.set_index("date", inplace=True)
    return data


# ---------------------------------------------------------------------
# STEP 2: Define asset groups
# ---------------------------------------------------------------------
GROUPS = {
    # --- Equities ---
    "equities": [
        "sp500", "nasdaq", "vgk", "ewj", "eem", "mtum", "vtv", "iwf"
    ],

    # --- Credit (Corporate Bonds) ---
    "credit": [
        "hyg", "lqd"
    ],

    # --- Bonds & Rates (Safe Havens, inverse) ---
    "bonds": [
        "irx", "fvx", "tnx", "tlt"
    ],

    # --- Commodities ---
    "commodities": [
        "oil", "natgas", "gold", "silver", "copper"
    ],

    # --- FX ---
    "fx": [
        "usd_index", "eurusd", "gbpusd", "audusd", "usdjpy", "usdchf", "cew"
    ],

    # --- Crypto ---
    "crypto": [
        "bitcoin", "ethereum"
    ],

    # --- Volatility ---
    "vol": [
        "vix"
    ],
}


# Fundamental weights (static baseline)
STATIC_WEIGHTS = {
    "equities": 0.30,
    "credit": 0.20,
    "bonds": 0.15,
    "commodities": 0.15,
    "fx": 0.10,
    "crypto": 0.05,
    "vol": 0.05,
}


# ---------------------------------------------------------------------
# STEP 3: Compute normalized returns
# ---------------------------------------------------------------------
def compute_normalized_returns(df: pd.DataFrame):
    """Computes daily % returns and z-score normalization per column."""
    returns = df.pct_change().dropna()
    zscores = (returns - returns.mean()) / returns.std()
    return zscores


# ---------------------------------------------------------------------
# STEP 4: Compute category averages
# ---------------------------------------------------------------------
def compute_group_scores(zscores: pd.DataFrame):
    """Average z-scores by group."""
    group_scores = {}
    for group, cols in GROUPS.items():
        valid_cols = [c for c in cols if c in zscores.columns]
        if valid_cols:
            group_scores[group] = zscores[valid_cols].mean(axis=1)
    return pd.DataFrame(group_scores)


# ---------------------------------------------------------------------
# STEP 5: Weighted risk sentiment index
# ---------------------------------------------------------------------
def compute_sentiment_index(group_scores: pd.DataFrame, weights=None):
    """
    Computes the weighted sentiment index.
    You can plug in:
        - STATIC_WEIGHTS (fundamental)
        - quant weights (e.g. PCA or inverse volatility)
    """
    if weights is None:
        weights = STATIC_WEIGHTS

    # Invert the sign for "vol" group (VIX up → risk-off)
    if "vol" in group_scores.columns:
        group_scores["vol"] *= -1

    sentiment = sum(group_scores[g] * weights.get(g, 0) for g in group_scores.columns)
    sentiment = sentiment / sum(weights.values())  # normalize to 1.0 total
    return sentiment


# ---------------------------------------------------------------------
# STEP 6: Main function
# ---------------------------------------------------------------------
def generate_sentiment_series():
    df = load_market_data()
    zscores = compute_normalized_returns(df)
    group_scores = compute_group_scores(zscores)
    sentiment = compute_sentiment_index(group_scores)
    return sentiment, group_scores

if __name__ == "__main__":
    sentiment, groups = generate_sentiment_series()
    print(sentiment.tail())
    df = load_market_data()
    returns = df["sp500"].pct_change().dropna()
    corr = sentiment.corr(returns)
    print("Correlation with SP500:", corr)

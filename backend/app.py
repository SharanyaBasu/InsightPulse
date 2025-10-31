from fastapi import FastAPI
from data_fetcher import get_market_data
import pandas as pd

from sqlalchemy.orm import Session
from db import SessionLocal
from models import MarketData, MacroData

app = FastAPI() 

# Live Market Snapshot
@app.get("/api/market-data")
def market_data():
    df = get_market_data()
    latest = df.iloc[-1].to_dict()
    prev = df.iloc[-2].to_dict()

    sentiment_score = (
        (latest["SP500"] - prev["SP500"]) +
        (latest["NASDAQ"] - prev["NASDAQ"]) -
        (latest["Gold"] - prev["Gold"]) -
        (latest["USD_Index"] - prev["USD_Index"])
    )

    mood = "bullish" if sentiment_score > 0 else "bearish"

    return {
        "latest": latest,
        "sentiment": mood,
        "score": sentiment_score
    }

# Historical Data (from DB)
@app.get("/api/history")
def get_history():
    db: Session = SessionLocal()
    data = db.query(MarketData).order_by(MarketData.date).all()
    db.close()

    return [
        {
            "date": str(row.date),
            "sp500": row.sp500,
            "nasdaq": row.nasdaq,
            "gold": row.gold,
            "oil": row.oil,
            "copper": row.copper,
            "usd_index": row.usd_index,
            "eurusd": row.eurusd,
            "usdjpy": row.usdjpy,
            "yield_10y": row.yield_10y,
            "yield_2y": row.yield_2y,
            "yield_3m": row.yield_3m,
            "bitcoin": row.bitcoin,
            "ethereum": row.ethereum,
        }
        for row in data
    ]

@app.get("/api/macro")
def get_macro():
    """
    Returns stored macroeconomic indicators from DB.
    """
    db: Session = SessionLocal()
    data = db.query(MacroData).order_by(MacroData.date).all()
    db.close()

    return [
        {
            "date": str(r.date),
            "cpi": r.cpi,
            "unemployment": r.unemployment,
            "fed_funds_rate": r.fed_funds_rate,
            "gdp": r.gdp,
        }
        for r in data
    ]

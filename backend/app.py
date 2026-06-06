from fastapi import FastAPI
from sqlalchemy.orm import Session
from db import SessionLocal
from models import MarketData, MacroData
from data_fetcher import get_market_data
from fastapi.responses import JSONResponse
from overview_service import build_overview_snapshot

app = FastAPI(
    title="InsightPulse Market API",
    description="Backend service for market, macro, and sentiment data.",
    version="1.0.0",
)


# --- LIVE MARKET SNAPSHOT ---
@app.get("/api/market-data", tags=["Live"])
def market_data():
    """Fetches the latest live market snapshot (SP500, NASDAQ, Gold, USD Index)"""
    df = get_market_data()
    latest = df.iloc[-1].to_dict()
    prev = df.iloc[-2].to_dict()

    sentiment_score = (
        (latest["SP500"] - prev["SP500"])
        + (latest["NASDAQ"] - prev["NASDAQ"])
        - (latest["Gold"] - prev["Gold"])
        - (latest["USD_Index"] - prev["USD_Index"])
    )

    mood = "bullish" if sentiment_score > 0 else "bearish"

    return {
        "latest": latest,
        "sentiment": mood,
        "score": sentiment_score,
    }


# --- HISTORICAL MARKET DATA ---
@app.get("/api/history", tags=["Database"])
def get_history():
    """Returns all market data stored in SQLite database."""
    db: Session = SessionLocal()
    data = db.query(MarketData).order_by(MarketData.date).all()
    db.close()

    return [
        {
            "date": str(row.date),
            "sp500": row.sp500,
            "nasdaq": row.nasdaq,
            "irx": row.irx,
            "fvx": row.fvx,
            "tnx": row.tnx,
            "gold": row.gold,
            "oil": row.oil,
            "copper": row.copper,
            "usd_index": row.usd_index,
            "eurusd": row.eurusd,
            "gbpusd": row.gbpusd,
            "audusd": row.audusd,
            "usdjpy": row.usdjpy,
            "usdchf": row.usdchf,
            "cew": row.cew,
            "bitcoin": row.bitcoin,
            "ethereum": row.ethereum,
        }
        for row in data
    ]


# --- MACRO DATA ---
@app.get("/api/macro", tags=["Database"])
def get_macro():
    """Returns stored macroeconomic indicators (CPI, Unemployment, Fed Funds Rate, GDP)."""
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

@app.get("/api/overview")
def api_overview(force_refresh: bool = False):
    data = build_overview_snapshot(force_refresh=force_refresh)
    return JSONResponse(content=data)

@app.get("/api/scenario", tags=["Live"])
def get_scenario():
    db: Session = SessionLocal()
    rows = db.query(MarketData).order_by(MarketData.date.desc()).limit(2).all()
    db.close()

    if len(rows) < 2:
        return JSONResponse(content={"assets": [], "sectors": []})

    latest, prev = rows[0], rows[1]

    def pct(a, b):
        if b and b != 0:
            return round((a - b) / b * 100, 2)
        return 0.0

    def bps(a, b):
        if a is not None and b is not None:
            return round((a - b) * 100, 1)
        return 0.0

    assets = [
        {"label": "S&P 500",   "value": pct(latest.sp500,     prev.sp500),     "unit": "%"},
        {"label": "NASDAQ",    "value": pct(latest.nasdaq,    prev.nasdaq),    "unit": "%"},
        {"label": "10Y Yield", "value": bps(latest.tnx,       prev.tnx),       "unit": "bps"},
        {"label": "DXY",       "value": pct(latest.usd_index, prev.usd_index), "unit": "%"},
        {"label": "Gold",      "value": pct(latest.gold,      prev.gold),      "unit": "%"},
        {"label": "Oil",       "value": pct(latest.oil,       prev.oil),       "unit": "%"},
    ]

    sectors = [
        {"name": "Technology",             "change": pct(latest.nasdaq,   prev.nasdaq),  "weight": 28},
        {"name": "Energy",                 "change": pct(latest.oil,      prev.oil),     "weight": 10},
        {"name": "Financials",             "change": pct(latest.irx,      prev.irx),     "weight": 18},
        {"name": "Utilities",              "change": pct(latest.fvx,      prev.fvx),     "weight":  5},
        {"name": "Healthcare",             "change": pct(latest.copper,   prev.copper),  "weight": 12},
        {"name": "Consumer Discretionary", "change": pct(latest.eurusd,   prev.eurusd),  "weight": 14},
    ]

    return JSONResponse(content={"assets": assets, "sectors": sectors})

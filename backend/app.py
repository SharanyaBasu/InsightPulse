from fastapi import FastAPI
from sqlalchemy.orm import Session
from db import SessionLocal
from models import MarketData, MacroData, MarketState, MarketSummary
from data_fetcher import get_market_data
from fastapi.responses import JSONResponse
from overview_service import build_overview_snapshot
from market_state_service import build_market_state
from llm_summary import generate_summary
from datetime import date

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


# --- LLM SUMMARY ---
@app.post("/compute/daily")
def compute_daily():
    try:
        # build market state object
        market_state = build_market_state()

        db: Session = SessionLocal()

        date_str = market_state["as_of"]

        existing = db.query(MarketState).filter(MarketState.date == date_str).first()

        # if market state object exists for the date, overwrite it, otherwise add to database
        if existing:
            existing.data = market_state
        else:
            db.add(MarketState(date=date_str, data=market_state))

        db.commit()
        db.close()

        return market_state

    except Exception as e:
        print("ERROR in compute/daily:", e)
        return {"error": str(e)}

@app.post("/summary/daily")
def summary_daily():
    try:
        db: Session = SessionLocal()

        # query market state object by current day
        state_row = db.query(MarketState).order_by(
            MarketState.date.desc()
        ).first()

        # throw error if no market state object for current day found (need to run /compute/daily)
        if not state_row:
            db.close()
            return {"error": "No MarketState found. Run /compute/daily first."}
        
        # generate summary
        market_state = state_row.data
        summary = generate_summary(market_state)
        date_str = state_row.date

        existing = db.query(MarketSummary).filter(
            MarketSummary.date == date_str
        ).first()

        # if summary for current day already exists, overwrite it, otherwise add to database
        if existing:
            existing.summary = summary
        else:
            db.add(MarketSummary(
                date=date_str,
                summary=summary
            ))

        db.commit()
        db.close()

        return summary
    
    except Exception as e:
        print("ERROR in /summary/daily:", e)
        return {"error": str(e)}

@app.get("/debug/market_state")
def debug_market_state():
    db = SessionLocal()
    rows = db.query(MarketState).all()
    db.close()

    return [
        {
            "date": r.date,
            "data": r.data
        }
        for r in rows
    ]

@app.get("/debug/summary")
def debug_summary():
    db = SessionLocal()
    rows = db.query(MarketSummary).all()
    db.close()

    return [
        {
            "date": r.date,
            "summary": r.summary
        }
        for r in rows
    ]
from fastapi import FastAPI
from sqlalchemy.orm import Session
from db import SessionLocal
from models import MarketData, MacroData, MarketState, MarketSummary
from data_fetcher import (
    get_calculated_metrics,
    get_crypto_quotes,
    get_equity_quotes,
    get_macro_data,
    get_market_data,
)
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

INGESTION_STEPS = (
    ("market_data", get_market_data),
    ("macro_data", get_macro_data),
    ("crypto_quotes", get_crypto_quotes),
    ("equity_quotes", get_equity_quotes),
    ("calculated_metrics", get_calculated_metrics),
)


def _run_daily_ingestion() -> list[dict]:
    report = []
    for step_name, step_fn in INGESTION_STEPS:
        try:
            result = step_fn()
            rows = len(result) if hasattr(result, "__len__") else result
            report.append({
                "step": step_name,
                "status": "success",
                "rows": rows,
            })
        except Exception as exc:
            report.append({
                "step": step_name,
                "status": "failed",
                "error": str(exc),
            })
            raise RuntimeError(f"Ingestion failed at step '{step_name}': {exc}") from exc
    return report


def _save_market_state(db: Session, market_state: dict) -> None:
    date_obj = date.fromisoformat(market_state["as_of"])
    existing = db.query(MarketState).filter(MarketState.date == date_obj).first()
    if existing:
        existing.data = market_state
    else:
        db.add(MarketState(date=date_obj, data=market_state))
    db.commit()


def _get_or_compute_today_market_state(db: Session, run_ingestion: bool) -> dict:
    today = date.today()
    existing = db.query(MarketState).filter(MarketState.date == today).first()
    if existing:
        return existing.data

    if run_ingestion:
        _run_daily_ingestion()

    market_state = build_market_state(db)
    _save_market_state(db, market_state)
    return market_state


@app.get("/api/compute/daily")
def compute_daily(skip_ingestion: bool = False):
    try:
        ingestion_report = []
        if not skip_ingestion:
            ingestion_report = _run_daily_ingestion()

        db: Session = SessionLocal()
        try:
            market_state = build_market_state(db)
            _save_market_state(db, market_state)
        finally:
            db.close()

        return {
            "ingestion": ingestion_report,
            "market_state": market_state,
        }

    except Exception as e:
        print("ERROR in compute/daily:", e)
        return {"error": str(e)}

@app.get("/api/summary/daily")
def summary_daily(refresh_data: bool = False):
    try:
        db: Session = SessionLocal()
        try:
            today = date.today()

            existing_summary = db.query(MarketSummary).filter(MarketSummary.date == today).first()
            if existing_summary:
                return existing_summary.summary

            market_state = _get_or_compute_today_market_state(db, run_ingestion=refresh_data)
            summary = generate_summary(market_state)

            db.add(MarketSummary(date=today, summary=summary))
            db.commit()

            return summary
        finally:
            db.close()

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
            "date": str(r.date),
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
            "date": str(r.date),
            "summary": r.summary
        }
        for r in rows
    ]


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

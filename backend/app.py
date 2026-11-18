import logging
import os
from contextlib import asynccontextmanager
from threading import Event, Thread

from scenario_modeling.src.scenario_runtime import BundleValidationError
from fastapi import FastAPI, HTTPException
from sqlalchemy.exc import IntegrityError
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
from llm_summary import generate_summary_with_guardrail
from summary_validator import attach_cited_evidence
from datetime import date
import refresh_service
from schemas.scenario import ScenarioRunRequest, ScenarioRunResponse
from scenario_service import run_scenario as run_scenario_engine


@asynccontextmanager
async def _lifespan(_app):
    stop = Event()
    if os.getenv("REFRESH_SCHEDULER", "1") != "0":
        Thread(
            target=refresh_service.scheduler_loop,
            args=(stop, _perform_refresh),
            name="refresh-scheduler",
            daemon=True,
        ).start()
    try:
        yield
    finally:
        stop.set()
        refresh_service.stop_process_refresh()


app = FastAPI(
    title="InsightPulse Market API",
    description="Backend service for market, macro, and sentiment data.",
    version="1.0.0",
    lifespan=_lifespan,
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
    """Load today's market state or rebuild it from stored data or explicit ingestion.

    Args:
        db: Active database session.
        run_ingestion: Whether to refresh external sources rather than stored metrics.

    Returns:
        Today's stored or newly built market state.
    """

    today = date.today()
    existing = db.query(MarketState).filter(MarketState.date == today).first()
    if existing and not run_ingestion:
        return existing.data

    if run_ingestion:
        _run_daily_ingestion()
    else:
        # Summary requests use stored source data; refreshing every provider is a
        # separate operation and must not block reading the dashboard.
        get_calculated_metrics()

    market_state = build_market_state(db)
    if not market_state.get("evidence"):
        raise RuntimeError("No market evidence is available; ingest market data first.")
    _save_market_state(db, market_state)
    return market_state


@app.get("/api/compute/daily")
def compute_daily(skip_ingestion: bool = False):
    """Queue the same refresh job the scheduler uses and return the saved snapshot."""

    perform = _perform_local_refresh if skip_ingestion else _perform_refresh
    refresh_service.start_refresh("manual", perform)
    db: Session = SessionLocal()
    try:
        payload = _public_summary(db)
    finally:
        db.close()
    payload.setdefault("refresh", {})["in_progress"] = True
    return payload

def _stored_market_state(db: Session, day=None) -> dict:
    """Read a saved market state without failing the summary when it is absent."""

    try:
        row = db.query(MarketState).filter(MarketState.date == (day or date.today())).first()
    except Exception:
        db.rollback()
        return {}
    data = getattr(row, "data", None)
    return data if isinstance(data, dict) else {}


def _stored_metric_dependencies(db: Session) -> dict:
    """Read metric source inputs when the calculated-metric table is available."""

    try:
        from services.metric_loader import load_latest_calculated_metrics

        return load_latest_calculated_metrics(db)
    except Exception:
        db.rollback()
        return {}


def _summary_with_evidence(summary, db: Session, market_state: dict | None = None):
    """Return a summary copy whose evidence map matches the cited market records."""

    if not isinstance(summary, dict):
        return summary
    state = market_state if isinstance(market_state, dict) else _stored_market_state(db)
    return attach_cited_evidence(summary, state, _stored_metric_dependencies(db))


def _observation_dates(market_state: dict) -> list[str]:
    return sorted({
        str(row["timestamp"])[:10]
        for row in market_state.get("evidence", {}).values()
        if isinstance(row, dict) and row.get("timestamp")
    })


def _write_snapshot(db: Session, market_state: dict, summary: dict) -> None:
    """Replace today's snapshot only after a complete refresh succeeds."""

    day = date.fromisoformat(market_state["as_of"])
    existing_state = db.query(MarketState).filter(MarketState.date == day).first()
    if existing_state:
        existing_state.data = market_state
    else:
        db.add(MarketState(date=day, data=market_state))
    existing_summary = db.query(MarketSummary).filter(MarketSummary.date == day).first()
    if existing_summary:
        existing_summary.summary = summary
    else:
        db.add(MarketSummary(date=day, summary=summary))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise


def _perform_local_refresh() -> tuple[str | None, str | None]:
    """Rebuild the snapshot from stored data without downloading providers."""

    return _publish_snapshot(ingest=False)


def _perform_refresh() -> tuple[str | None, str | None]:
    """Ingest sources and publish a new snapshot. Failures leave the saved snapshot in place."""

    return _publish_snapshot(ingest=True)


def _publish_snapshot(ingest: bool) -> tuple[str | None, str | None]:
    if ingest:
        _run_daily_ingestion()
    db: Session = SessionLocal()
    try:
        market_state = build_market_state(db)
        if not market_state.get("evidence"):
            raise RuntimeError("No market evidence is available; ingest market data first.")
        summary = generate_summary_with_guardrail(market_state)
        dates = _observation_dates(market_state)
        if dates:
            summary["data_as_of"] = {"oldest": dates[0], "latest": dates[-1]}
        _write_snapshot(db, market_state, summary)
        return market_state.get("as_of"), dates[-1] if dates else None
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _public_summary(db: Session) -> dict:
    """Return the newest saved summary and the refresh status. Never starts a refresh."""

    try:
        row = db.query(MarketSummary).order_by(MarketSummary.date.desc()).first()
    except Exception:
        db.rollback()
        row = None
    status = refresh_service.refresh_status(db)
    if row is None or not isinstance(row.summary, dict):
        return {"refresh": status}
    payload = _summary_with_evidence(dict(row.summary), db, _stored_market_state(db, row.date))
    payload["refresh"] = status
    return payload


@app.get("/api/summary/daily")
def summary_daily(refresh_data: bool = False):
    """Serve the last saved snapshot. A requested refresh runs in the background."""

    if refresh_data:
        refresh_service.start_refresh("manual", _perform_refresh)
    db: Session = SessionLocal()
    try:
        payload = _public_summary(db)
    finally:
        db.close()
    if refresh_data:
        payload.setdefault("refresh", {})["in_progress"] = True
    return payload

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
    """Live day-over-day market deltas for dashboard widgets.

    Not used by Scenario Playground. Playground simulation uses
    POST /api/scenario/run instead.
    """
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


@app.post(
    "/api/scenario/run",
    response_model=ScenarioRunResponse,
    tags=["Scenario Playground"],
    summary="Run Scenario Playground simulation",
)
def run_scenario(payload: ScenarioRunRequest) -> ScenarioRunResponse:
    """Accept macro shocks and return a Scenario Playground result.

    Request/response schemas are defined in `schemas/scenario.py`.
    Projection logic lives in `scenario_service.run_scenario` (Ridge v1 by default).
    """
    try:
        result = run_scenario_engine(payload.model_dump())
    except (BundleValidationError, FileNotFoundError) as exc:
        logging.getLogger(__name__).exception("Scenario model bundle unavailable")
        raise HTTPException(
            status_code=503,
            detail="Scenario model bundle unavailable. Install or restore the verified release bundle, then retry.",
        ) from exc
    return ScenarioRunResponse(**result)

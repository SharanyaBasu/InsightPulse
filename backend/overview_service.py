# backend/overview_service.py
import time
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from db import SessionLocal
from signals_engine import generate_sentiment_series, load_market_data


# --------------------------------------------------------
# Cache
# --------------------------------------------------------
_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
CACHE_TTL_SECONDS = 300  # 5 minutes


# --------------------------------------------------------
# Config (FIXED TO MATCH YOUR DB)
# --------------------------------------------------------
CARD_ASSETS = [
    ("sp500", "S&P 500"),
    ("nasdaq", "NASDAQ"),
    ("vgk", "Europe (VGK)"),
    ("ewj", "Japan (EWJ)"),
    ("eem", "Emerging Markets (EEM)"),
    ("usd_index", "Dollar Index"),
    ("gold", "Gold"),
    ("oil", "Brent Crude"),
]

SECTOR_MAP = {
    "xlk": "Technology",
    "xlf": "Financials",
    "xly": "Consumer Discretionary",
    "xlp": "Consumer Staples",
    "xle": "Energy",
    "xlv": "Health Care",
    "xli": "Industrials",
    "xlb": "Materials",
    "xlre": "Real Estate",
    "xlc": "Communication Services",
}

REGION_TILES = [
    ("US", "sp500"),
    ("Europe", "vgk"),
    ("Japan", "ewj"),
    ("Emerging Markets", "eem"),
]


# --------------------------------------------------------
# Sanitizer for JSON (critical fix)
# --------------------------------------------------------
def sanitize(obj):
    """Recursively clean NaN/inf so JSON won't crash."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def _load_macro_df():
    session = SessionLocal()
    try:
        df = pd.read_sql("SELECT * FROM macro_data ORDER BY date", session.bind)
    finally:
        session.close()

    if df.empty:
        raise RuntimeError("macro_data is empty")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    cols = [
        "cpi", "unemployment", "fed_funds_rate", "gdp",
        "two_year_yield", "ten_year_yield",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].astype(float)


def _pct_change(series: pd.Series, periods: int):
    if series is None or series.dropna().shape[0] <= periods:
        return np.nan
    try:
        latest = series.iloc[-1]
        prev = series.iloc[-(periods + 1)]
        if prev == 0 or pd.isna(prev):
            return np.nan
        return (latest / prev - 1) * 100
    except Exception:
        return np.nan


def _sparkline(series: pd.Series, length: int = 30):
    if series is None or series.dropna().empty:
        return []
    s = series.dropna().tail(length)
    if s.empty:
        return []
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return []
    return (s / base).tolist()


def _classify_risk(score: float):
    if score >= 0.20:
        return "Risk-On"
    if score <= -0.11:
        return "Risk-Off"
    return "Neutral"


def _classify_equity_trend(spx: pd.Series):
    if spx is None or spx.dropna().shape[0] < 60:
        return "Neutral"
    ret5 = _pct_change(spx, 5)
    ma50 = spx.rolling(50).mean()
    above = spx.iloc[-1] > ma50.iloc[-1]
    if ret5 >= 1 and above:
        return "Bullish"
    if ret5 <= -1 and not above:
        return "Bearish"
    return "Neutral"


def _direction(a, b):
    if pd.isna(a) or pd.isna(b):
        return "flat"
    if a > b:
        return "up"
    if a < b:
        return "down"
    return "flat"


def _narrative(spx1d, dxy1d, tenbps, risk):
    parts = []
    if not pd.isna(spx1d):
        parts.append("equities higher" if spx1d > 0 else "equities lower" if spx1d < 0 else "equities flat")
    if not pd.isna(dxy1d):
        parts.append("USD softer" if dxy1d < 0 else "USD firmer" if dxy1d > 0 else "USD steady")
    if not pd.isna(tenbps):
        parts.append("yields rising" if tenbps > 0 else "yields falling" if tenbps < 0 else "yields steady")
    if len(parts) == 0:
        core = "mixed markets"
    elif len(parts) == 1:
        core = parts[0]
    else:
        core = parts[0] + " while " + " and ".join(parts[1:])
    tone = {
        "Risk-On": " → risk-on tone.",
        "Risk-Off": " → risk-off tone.",
        "Neutral": " → balanced tone.",
    }[risk]
    return core.capitalize() + tone


# --------------------------------------------------------
# Main snapshot builder
# --------------------------------------------------------
def build_overview_snapshot(force_refresh=False):
    now = time.time()
    if not force_refresh and _CACHE["data"] and (now - _CACHE["ts"] < CACHE_TTL_SECONDS):
        return _CACHE["data"]

    # load DB data
    market_df = load_market_data()
    macro_df = _load_macro_df()
    macro_daily = macro_df.asfreq("B").ffill()

    # ------------------------
    # Market Cards
    # ------------------------
    cards = []
    for sym, name in CARD_ASSETS:
        if sym not in market_df.columns:
            continue
        s = market_df[sym]
        cards.append({
            "symbol": sym,
            "name": name,
            "price": float(s.iloc[-1]),
            "change_1d": _pct_change(s, 1),
            "change_1w": _pct_change(s, 5),
            "change_1m": _pct_change(s, 21),
            "change_1y": _pct_change(s, 252),
            "sparkline": _sparkline(s, 30),
        })

    # Add US 10Y
    if "ten_year_yield" in macro_daily.columns:
        t = macro_daily["ten_year_yield"]
        cards.append({
            "symbol": "dgs10",
            "name": "US 10Y Yield",
            "price": float(t.iloc[-1]),
            "change_1d": _pct_change(t, 1),
            "change_1w": _pct_change(t, 5),
            "change_1m": _pct_change(t, 21),
            "change_1y": _pct_change(t, 252),
            "sparkline": _sparkline(t, 30),
        })

    # ------------------------
    # Sentiment + Drivers
    # ------------------------
    sentiment_series, group_scores = generate_sentiment_series()
    score = float(sentiment_series.iloc[-1])
    risk_label = _classify_risk(score)
    drivers = []
    if isinstance(group_scores, pd.DataFrame):
        row = group_scores.iloc[-1]
        sorted_groups = row.abs().sort_values(ascending=False)
        for g in sorted_groups.index[:3]:
            arrow = "↑" if row[g] > 0 else "↓" if row[g] < 0 else "→"
            drivers.append(f"{g.capitalize()} {arrow}")

    # ------------------------
    # Macro chips
    # ------------------------
    latest = macro_df.iloc[-1]
    prev = macro_df.iloc[-2] if macro_df.shape[0] >= 2 else latest

    macro_out = {
        "cpi": {
            "value": latest.get("cpi"),
            "prev": prev.get("cpi"),
            "direction": _direction(latest.get("cpi"), prev.get("cpi")),
        },
        "unemployment": {
            "value": latest.get("unemployment"),
            "prev": prev.get("unemployment"),
            "direction": _direction(latest.get("unemployment"), prev.get("unemployment")),
        },
        "policy_rate": {
            "value": latest.get("fed_funds_rate"),
            "prev": prev.get("fed_funds_rate"),
            "direction": _direction(latest.get("fed_funds_rate"), prev.get("fed_funds_rate")),
        },
    }

    # ------------------------
    # Regions
    # ------------------------
    regions = []
    for region_name, sym in REGION_TILES:
        if sym in market_df.columns:
            regions.append({
                "region": region_name,
                "symbol": sym,
                "change_1m": _pct_change(market_df[sym], 21),
            })

    # ------------------------
    # Sectors
    # ------------------------
    sectors = []
    for sym, label in SECTOR_MAP.items():
        if sym in market_df.columns:
            sectors.append({
                "sector": label,
                "symbol": sym,
                "change_1m": _pct_change(market_df[sym], 21),
            })

    # ------------------------
    # Yield panel
    # ------------------------
    ten = macro_daily["ten_year_yield"].iloc[-1]
    two = macro_daily["two_year_yield"].iloc[-1]
    slope_bps = (ten - two) * 100
    slope_label = "Inverted" if slope_bps < 0 else "Normal"

    # ------------------------
    # Narrative
    # ------------------------
    spx1d = _pct_change(market_df.get("sp500"), 1)
    dxy1d = _pct_change(market_df.get("usd_index"), 1)

    ten1d = np.nan
    if macro_daily["ten_year_yield"].dropna().shape[0] > 1:
        ten1d = (macro_daily["ten_year_yield"].iloc[-1] -
                 macro_daily["ten_year_yield"].iloc[-2]) * 100

    narrative = _narrative(spx1d, dxy1d, ten1d, risk_label)

    # ------------------------
    # Payload
    # ------------------------
    payload = {
        "sentiment": {
            "score": score,
            "label": risk_label,
            "drivers": drivers,
            "equity_trend": _classify_equity_trend(market_df.get("sp500")),
        },
        "market_cards": cards,
        "macro": macro_out,
        "regions": regions,
        "sectors": sectors,
        "yield": {
            "ten_year": ten,
            "two_ten_slope_bps": slope_bps,
            "slope_label": slope_label,
        },
        "narrative": narrative,
        "last_updated": pd.Timestamp.utcnow().isoformat(),
    }

    # sanitize to avoid JSON crash
    payload = sanitize(payload)

    _CACHE["ts"] = now
    _CACHE["data"] = payload
    return payload

from datetime import date
from overview_service import (
    build_overview_snapshot,
)

def build_market_state():
    # create formatted date for market state object
    today = date.today()
    formatted_date = today.isoformat()

    # get market overview
    data = build_overview_snapshot()

    # REGIME
    sentiment = data["sentiment"]

    regime = {
        "label": sentiment["label"],
        "confidence": float(sentiment["score"])
    }

    # DRIVERS
    cross = data["cross_asset"]
    drivers = []

    for asset in cross:
        if asset["change_1d"] is None:
            continue
        
        drivers.append({
            "name": f"${asset['name'].lower().replace(' ', '_')}_1d_return",
            "value": float(asset["change_1d"]),
            "unit": "pct"
        })
    
    # top 3 drivers
    drivers = sorted(drivers, key=lambda x: abs(x["value"]), reverse=True)[:3]


    # CORRELATION SHIFTS
    correlation_shifts = []

    for c in data["correlations"]:
        if c["correlation"] is None:
            continue

        correlation_shifts.append({
            "pair": c["pair"],
            "corr_now": float(c["correlation"]),
            "label": c["label"]
        })

    # STRESS FLAGS
    stress_flags = []

    # Volatility check
    vix = next((x for x in cross if x["name"] == "VIX"), None)
    if vix and vix["change_1d"] is not None and vix["change_1d"] > 5:
        stress_flags.append({
            "name": "vol_spike",
            "active": True,
            "severity": "medium"
        })

    # Equity drawdown check (S&P 500)
    spx = next((x for x in cross if x["name"] == "S&P 500"), None)
    if spx and spx["change_1d"] is not None and spx["change_1d"] < -1:
        stress_flags.append({
            "name": "drawdown_alert",
            "active": True,
            "severity": "low"
        })

    # MOOD
    score = float(sentiment["score"])
    mood_label = ""

    if (score > 0.1):
        mood_label = "bullish"
    elif (score < -0.1):
        mood_label = "bearish"
    else:
        mood_label = "neutral"

    mood_5d = {
        "label": mood_label,
        "prob": abs(score),
        "top_features": [
            {
                "name": d["name"],
                "contribution": abs(d["value"])
            }
            for d in drivers[:2]
        ]
    }


    # MARKET STATE OBJECT
    market_state = {
        "as_of": formatted_date,
        "regime": regime,
        "drivers": drivers,
        "correlation_shifts": correlation_shifts,
        "stress_flags": stress_flags,
        "mood_5d": mood_5d
    }

    return market_state
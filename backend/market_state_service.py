"""Assemble daily MarketState from calculated metrics."""

from datetime import date

from sqlalchemy.orm import Session

from db import SessionLocal
from overview_service import sanitize
from services.metric_loader import load_latest_calculated_metrics


MARKET_ASSETS = [
    "sp500", "nasdaq", "vix", "gold", "usd_index", "bitcoin", "ethereum", "tlt",
]

MACRO_METRICS = [
    "yield_curve_slope_10y_2y",
    "real_rate_proxy",
    "dxy_1d_return",
    "btc_eth_ratio",
]

ASSET_METRIC_SUFFIXES = [
    "5d_return",
    "20d_volatility",
    "drawdown_from_peak",
    "price_zscore_252d",
    "ma_distance_50d",
    "ma_distance_200d",
]

REGIME_METRICS = [
    "yield_curve_slope_10y_2y",
    "real_rate_proxy",
    "sp500_5d_return",
    "sp500_drawdown_from_peak",
    "vix_20d_volatility",
]

VIX_VOL_STRESS_THRESHOLD = 0.25
MAX_DRIVERS = 6

ASSET_CLASSES = {
    "equities": ["sp500", "nasdaq"],
    "bonds": ["tlt"],
    "crypto": ["bitcoin", "ethereum"],
    "commodities": ["gold"],
    "fx": ["usd_index"],
}


def _allowed_metric_names() -> set[str]:
    names = set(MACRO_METRICS)
    for asset in MARKET_ASSETS:
        for suffix in ASSET_METRIC_SUFFIXES:
            names.add(f"{asset}_{suffix}")
    return names


def _filter_metrics(metrics: dict) -> dict:
    allowed = _allowed_metric_names()
    return {name: row for name, row in metrics.items() if name in allowed}


def _metric_value(metrics: dict, name: str):
    row = metrics.get(name)
    if not row:
        return None
    return row.get("value")


def _unit_for(metric_name: str) -> str:
    if any(k in metric_name for k in ("return", "drawdown", "ma_distance")):
        return "pct"
    if "volatility" in metric_name:
        return "pct_annualized"
    return "level"


def _display_value(metric_name: str, value: float):
    if any(k in metric_name for k in ("return", "drawdown", "ma_distance")):
        return round(value * 100, 3)
    if "volatility" in metric_name:
        return round(value * 100, 2)
    return round(value, 4)


def _asset_from_metric(metric_name: str):
    for asset in MARKET_ASSETS:
        if metric_name.startswith(f"{asset}_"):
            return asset
    return None


def _z_for_metric(metric_name: str, metrics: dict):
    if "zscore" in metric_name:
        value = _metric_value(metrics, metric_name)
        return round(value, 2) if value is not None else None

    asset = _asset_from_metric(metric_name)
    if not asset:
        return None
    z = _metric_value(metrics, f"{asset}_price_zscore_252d")
    return round(z, 2) if z is not None else None


def _driver_importance(metric_name: str, value: float) -> float:
    magnitude = abs(value)
    if any(k in metric_name for k in ("return", "drawdown", "ma_distance")):
        return magnitude * 100
    if any(k in metric_name for k in ("slope", "proxy", "ratio")):
        return magnitude * 10
    return magnitude


def _classify_regime(metrics: dict) -> tuple[str, float]:
    yield_slope = _metric_value(metrics, "yield_curve_slope_10y_2y")
    real_rate = _metric_value(metrics, "real_rate_proxy")
    spx_drawdown = _metric_value(metrics, "sp500_drawdown_from_peak")
    spx_5d = _metric_value(metrics, "sp500_5d_return")
    vix_vol = _metric_value(metrics, "vix_20d_volatility")

    stress_signals = 0
    if yield_slope is not None and yield_slope < 0:
        stress_signals += 1
    if spx_drawdown is not None and spx_drawdown < -0.05:
        stress_signals += 1
    if vix_vol is not None and vix_vol > VIX_VOL_STRESS_THRESHOLD:
        stress_signals += 1

    if stress_signals >= 2:
        label = "liquidity_stress"
    elif real_rate is not None and real_rate < 0:
        label = "inflationary"
    elif spx_5d is not None and spx_5d > 0.01:
        label = "risk_on"
    elif spx_5d is not None and spx_5d < -0.01:
        label = "risk_off"
    else:
        label = "transitional"

    signal_strength = 0.0
    if spx_5d is not None:
        signal_strength += min(abs(spx_5d) * 10, 0.4)
    if real_rate is not None:
        signal_strength += min(abs(real_rate) / 10, 0.3)
    confidence = round(min(1.0, max(0.3, signal_strength + stress_signals * 0.15)), 2)

    return label, confidence


def _build_regime(metrics: dict) -> dict:
    label, confidence = _classify_regime(metrics)
    return {"label": label, "confidence": confidence}


def _build_drivers(metrics: dict) -> list:
    candidates = []
    for metric_name, row in metrics.items():
        value = row.get("value")
        if value is None:
            continue
        candidates.append({
            "name": metric_name,
            "value": _display_value(metric_name, float(value)),
            "z": _z_for_metric(metric_name, metrics),
            "unit": _unit_for(metric_name),
            "evidence_id": metric_name,
            "importance": _driver_importance(metric_name, float(value)),
        })

    ranked = sorted(candidates, key=lambda x: x["importance"], reverse=True)[:MAX_DRIVERS]
    return [
        {k: v for k, v in item.items() if k != "importance"}
        for item in ranked
    ]


def _build_stress_flags(metrics: dict) -> list:
    flags = []

    vix_vol = _metric_value(metrics, "vix_20d_volatility")
    if vix_vol is not None and vix_vol > VIX_VOL_STRESS_THRESHOLD:
        flags.append({
            "name": "vol_spike",
            "active": True,
            "severity": "high" if vix_vol > 0.35 else "medium",
            "evidence_id": "vix_20d_volatility",
        })

    spx_drawdown = _metric_value(metrics, "sp500_drawdown_from_peak")
    if spx_drawdown is not None and spx_drawdown < -0.05:
        flags.append({
            "name": "drawdown_alert",
            "active": True,
            "severity": "medium",
            "evidence_id": "sp500_drawdown_from_peak",
        })

    yield_slope = _metric_value(metrics, "yield_curve_slope_10y_2y")
    if yield_slope is not None and yield_slope < 0:
        flags.append({
            "name": "yield_curve_inversion",
            "active": True,
            "severity": "medium",
            "evidence_id": "yield_curve_slope_10y_2y",
        })

    dxy_1d = _metric_value(metrics, "dxy_1d_return")
    if dxy_1d is not None and abs(dxy_1d) > 0.01:
        flags.append({
            "name": "fx_shock",
            "active": True,
            "severity": "low",
            "evidence_id": "dxy_1d_return",
        })

    return flags


def _build_asset_class_snapshot(metrics: dict) -> list:
    snapshot = []

    for asset_class, assets in ASSET_CLASSES.items():
        returns = []
        evidence_ids = []
        for asset in assets:
            metric_id = f"{asset}_5d_return"
            value = _metric_value(metrics, metric_id)
            if value is None:
                continue
            returns.append(_display_value(metric_id, float(value)))
            evidence_ids.append(metric_id)

        if not returns:
            continue

        return_5d_pct = round(sum(returns) / len(returns), 3)
        snapshot.append({
            "class": asset_class,
            "return_5d_pct": return_5d_pct,
            "evidence_ids": evidence_ids,
        })

    return snapshot


def _build_mood_5d(metrics: dict) -> dict:
    mood_metrics = [
        ("sp500_5d_return", False),
        ("nasdaq_5d_return", False),
        ("vix_5d_return", True),
        ("gold_5d_return", False),
        ("tlt_5d_return", False),
    ]

    features = []
    for metric_id, bearish_when_positive in mood_metrics:
        value = _metric_value(metrics, metric_id)
        if value is None:
            continue
        features.append({
            "name": metric_id,
            "value": value,
            "bearish_when_positive": bearish_when_positive,
        })

    if not features:
        return {"label": "neutral", "prob": 0.5, "top_features": []}

    bullish = bearish = 0
    scored = []
    for f in features:
        value = f["value"]
        if f["bearish_when_positive"]:
            if value > 0:
                bearish += 1
                scored.append((f["name"], abs(value)))
            elif value < 0:
                bullish += 1
                scored.append((f["name"], abs(value)))
        else:
            if value > 0:
                bullish += 1
                scored.append((f["name"], abs(value)))
            elif value < 0:
                bearish += 1
                scored.append((f["name"], abs(value)))

    total = bullish + bearish
    if bullish > bearish:
        label = "bullish"
        prob = round(bullish / total, 2) if total else 0.5
    elif bearish > bullish:
        label = "bearish"
        prob = round(bearish / total, 2) if total else 0.5
    else:
        label = "neutral"
        prob = 0.5

    top_features = [
        {"name": name, "contribution": round(contrib, 4)}
        for name, contrib in sorted(scored, key=lambda x: x[1], reverse=True)[:3]
    ]

    return {"label": label, "prob": prob, "top_features": top_features}


def _collect_evidence_ids(drivers, stress_flags, mood_5d, asset_class_snapshot) -> set[str]:
    ids = set(REGIME_METRICS)
    for driver in drivers:
        ids.add(driver["evidence_id"])
    for flag in stress_flags:
        ids.add(flag["evidence_id"])
    for feature in mood_5d.get("top_features", []):
        ids.add(feature["name"])
    for row in asset_class_snapshot:
        ids.update(row.get("evidence_ids", []))
    return ids


def _build_evidence(metrics: dict, evidence_ids: set[str]) -> dict:
    evidence = {}
    for eid in sorted(evidence_ids):
        row = metrics.get(eid)
        if not row or row.get("value") is None:
            continue
        raw = float(row["value"])
        evidence[eid] = {
            "value": raw,
            "display_value": _display_value(eid, raw),
            "unit": _unit_for(eid),
            "window": row.get("window"),
            "timestamp": row.get("timestamp"),
            "source": "calculated_metrics",
        }
    return evidence


def build_market_state(db: Session | None = None) -> dict:
    own_session = db is None
    if own_session:
        db = SessionLocal()

    try:
        metrics = _filter_metrics(load_latest_calculated_metrics(db))

        regime = _build_regime(metrics)
        drivers = _build_drivers(metrics)
        stress_flags = _build_stress_flags(metrics)
        mood_5d = _build_mood_5d(metrics)
        asset_class_snapshot = _build_asset_class_snapshot(metrics)
        evidence_ids = _collect_evidence_ids(
            drivers, stress_flags, mood_5d, asset_class_snapshot,
        )

        market_state = {
            "as_of": date.today().isoformat(),
            "regime": regime,
            "drivers": drivers,
            "asset_class_snapshot": asset_class_snapshot,
            "correlation_shifts": [],
            "stress_flags": stress_flags,
            "mood_5d": mood_5d,
            "evidence": _build_evidence(metrics, evidence_ids),
        }

        return sanitize(market_state)
    finally:
        if own_session:
            db.close()

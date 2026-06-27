# backend/regime_service.py
import numpy as np
import pandas as pd

SECTOR_COLS = [
    "xlk", "xlf", "xly", "xlp", "xle", "xlv", "xli", "xlb", "xlre", "xlc",
]

REGIME_DESCRIPTIONS = {
    "Growth Regime": "Risk assets supported — volatility contained and cross-asset signals skew risk-on.",
    "Inflationary Regime": "Rates and inflation-sensitive assets dominate — watch gold/yield and commodity correlations.",
    "Liquidity Stress": "Volatility elevated and defensive correlations strengthening — risk assets under pressure.",
    "Transitional / Unstable": "Mixed macro signals — correlations and dispersion do not point to a single stable regime.",
}


def _corr_label_by_key(correlations: list, key: str) -> str:
    for item in correlations or []:
        if item.get("key") == key:
            return item.get("label") or "—"
    return "—"


def _vix_signals(market_df: pd.DataFrame) -> dict:
    if "vix" not in market_df.columns:
        return {"value": None, "percentile": None, "level": "Unknown"}

    series = market_df["vix"].dropna()
    if series.empty:
        return {"value": None, "percentile": None, "level": "Unknown"}

    value = float(series.iloc[-1])
    window = series.tail(252) if len(series) >= 60 else series
    percentile = float((window <= value).mean() * 100)

    if value >= 30 or percentile >= 85:
        level = "Elevated"
    elif value >= 20 or percentile >= 60:
        level = "Moderate"
    else:
        level = "Subdued"

    return {
        "value": round(value, 2),
        "percentile": round(percentile, 1),
        "level": level,
    }


def _dispersion_signals(market_df: pd.DataFrame) -> dict:
    cols = [c for c in SECTOR_COLS if c in market_df.columns]
    if len(cols) < 3:
        return {"value": None, "level": "Unknown"}

    returns_1m = []
    for col in cols:
        s = market_df[col].dropna()
        if len(s) < 22:
            continue
        ret = (s.iloc[-1] / s.iloc[-22] - 1) * 100
        if np.isfinite(ret):
            returns_1m.append(ret)

    if len(returns_1m) < 3:
        return {"value": None, "level": "Unknown"}

    value = float(np.std(returns_1m))
    if value >= 8:
        level = "High"
    elif value >= 4:
        level = "Moderate"
    else:
        level = "Low"

    return {"value": round(value, 2), "level": level}


def _pick_regime(vix: dict, dispersion: dict, corr_labels: dict) -> tuple[str, str]:
    vix_val = vix.get("value")
    vix_level = vix.get("level")
    disp_level = dispersion.get("level")
    spx_10y = corr_labels.get("spx_vs_10y", "Neutral")
    spx_dxy = corr_labels.get("spx_vs_dxy", "Neutral")
    gold_10y = corr_labels.get("gold_vs_10y", "Neutral")

    stress_score = 0
    growth_score = 0
    inflation_score = 0

    if vix_level == "Elevated" or (vix_val is not None and vix_val >= 28):
        stress_score += 2
    elif vix_level == "Moderate":
        stress_score += 1

    if spx_10y == "Negative":
        stress_score += 1
        growth_score -= 1
    if spx_dxy == "Negative":
        growth_score += 1
    if spx_dxy == "Positive" and spx_10y != "Negative":
        inflation_score += 1

    if gold_10y == "Positive":
        inflation_score += 2
    elif gold_10y == "Negative" and spx_10y == "Positive":
        growth_score += 1

    if disp_level == "High":
        stress_score += 1

    scores = {
        "Liquidity Stress": stress_score,
        "Growth Regime": growth_score,
        "Inflationary Regime": inflation_score,
    }
    best_regime = max(scores, key=scores.get)
    best_score = scores[best_regime]
    second_score = sorted(scores.values(), reverse=True)[1]

    if best_score <= 0 or best_score == second_score:
        regime = "Transitional / Unstable"
        confidence = "Low"
    elif best_score - second_score >= 2:
        regime = best_regime
        confidence = "High"
    else:
        regime = best_regime
        confidence = "Medium"

    return regime, confidence


def classify_regime(
    market_df: pd.DataFrame,
    macro_daily: pd.DataFrame,
    correlations: list,
) -> dict:
    """Classify macro regime from volatility, dispersion, and correlation structure."""
    del macro_daily  # reserved for future macro-conditioned rules

    vix = _vix_signals(market_df)
    dispersion = _dispersion_signals(market_df)
    corr_labels = {
        "spx_vs_10y": _corr_label_by_key(correlations, "spx_vs_10y"),
        "spx_vs_dxy": _corr_label_by_key(correlations, "spx_vs_dxy"),
        "gold_vs_10y": _corr_label_by_key(correlations, "gold_vs_10y"),
    }

    regime, confidence = _pick_regime(vix, dispersion, corr_labels)

    return {
        "regime": regime,
        "confidence": confidence,
        "description": REGIME_DESCRIPTIONS.get(regime, ""),
        "signals": {
            "vix": vix,
            "dispersion": dispersion,
            "correlations": corr_labels,
        },
    }

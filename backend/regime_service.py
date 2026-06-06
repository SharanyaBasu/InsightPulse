# backend/regime_service.py
"""
Regime Label Classification

Uses correlations + volatility + dispersion to assign a structural
regime label to the current macro environment.

Data sources (all from the SQLite DB, originally fetched by data_fetcher.py):
  - VIX              → market_data.vix          (source: yfinance ^VIX)
  - Sector ETFs      → market_data.xl*          (source: yfinance sector ETFs)
  - S&P 500          → market_data.sp500        (source: yfinance ^GSPC)
  - 10Y Yield        → macro_data.ten_year_yield (source: FRED DGS10)
  - Correlations     → computed by correlation_service.py from the above
"""
import numpy as np
import pandas as pd


# Sector columns used for dispersion calculation
SECTOR_COLS = ["xlk", "xlf", "xly", "xlp", "xle", "xlv", "xli", "xlb", "xlre", "xlc"]

# VIX thresholds
VIX_HIGH = 25.0
VIX_ELEVATED = 18.0

# Dispersion thresholds (std dev of 21-day sector returns)
DISPERSION_HIGH = 0.04
DISPERSION_LOW = 0.015


def _vix_level(market_df: pd.DataFrame) -> dict:
    """
    Current VIX level and 60-day percentile.
    Data source: market_data.vix (fetched from yfinance ^VIX by data_fetcher.py)
    """
    if "vix" not in market_df.columns:
        return {"value": None, "percentile": None, "level": "unknown"}

    vix = market_df["vix"].dropna()
    if vix.empty:
        return {"value": None, "percentile": None, "level": "unknown"}

    current = float(vix.iloc[-1])
    window = vix.tail(252)  # 1-year lookback for percentile
    percentile = float((window < current).mean() * 100)

    if current >= VIX_HIGH:
        level = "high"
    elif current >= VIX_ELEVATED:
        level = "elevated"
    else:
        level = "low"

    return {"value": round(current, 2), "percentile": round(percentile, 1), "level": level}


def _sector_dispersion(market_df: pd.DataFrame) -> dict:
    """
    Cross-sector return dispersion (std dev of 21-day sector returns).
    Data source: market_data sector ETF columns (fetched from yfinance by data_fetcher.py)
    """
    available = [c for c in SECTOR_COLS if c in market_df.columns]
    if len(available) < 4:
        return {"value": None, "level": "unknown"}

    sector_df = market_df[available].dropna()
    if len(sector_df) < 22:
        return {"value": None, "level": "unknown"}

    # 21-day returns for each sector
    returns_21d = sector_df.iloc[-1] / sector_df.iloc[-22] - 1
    dispersion = float(returns_21d.std())

    if dispersion >= DISPERSION_HIGH:
        level = "high"
    elif dispersion <= DISPERSION_LOW:
        level = "low"
    else:
        level = "moderate"

    return {"value": round(dispersion * 100, 2), "level": level}


def _extract_corr_signals(correlations: list) -> dict:
    """
    Pull out correlation labels from the already-computed correlation pairs.
    Data source: correlation_service.build_correlations() output
      - spx_vs_10y: market_data.sp500 + macro_data.ten_year_yield
      - spx_vs_dxy: market_data.sp500 + market_data.usd_index
      - gold_vs_10y: market_data.gold + macro_data.ten_year_yield
    """
    signals = {}
    for pair in correlations:
        signals[pair["key"]] = {
            "correlation": pair.get("correlation"),
            "label": pair.get("label"),
        }
    return signals


def classify_regime(
    market_df: pd.DataFrame,
    macro_daily: pd.DataFrame,
    correlations: list,
) -> dict:
    """
    Classify the current macro regime.

    Inputs and their data sources:
      market_df   → loaded from SQLite market_data table
                     (originally fetched from yfinance by data_fetcher.py)
      macro_daily → loaded from SQLite macro_data table, resampled to business days
                     (originally fetched from FRED API by data_fetcher.py)
      correlations→ output of correlation_service.build_correlations()
                     (computed from market_df + macro_daily)

    Returns a dict with regime label, description, and supporting signals.
    """
    vix = _vix_level(market_df)
    dispersion = _sector_dispersion(market_df)
    corr_signals = _extract_corr_signals(correlations)

    # --- Classification logic ---
    spx_10y = corr_signals.get("spx_vs_10y", {})
    spx_dxy = corr_signals.get("spx_vs_dxy", {})
    gold_10y = corr_signals.get("gold_vs_10y", {})

    spx_10y_label = spx_10y.get("label", "Neutral")
    spx_dxy_label = spx_dxy.get("label", "Neutral")
    gold_10y_label = gold_10y.get("label", "Neutral")
    vix_level = vix["level"]
    disp_level = dispersion["level"]

    # Liquidity Stress: VIX spiking + correlations converging (everything sells off)
    if vix_level == "high" and spx_10y_label == "Negative":
        regime = "Liquidity Stress"
        description = (
            "Volatility is elevated and correlations indicate flight to safety. "
            "Assets are moving in lockstep as risk is repriced across the board."
        )

    # Inflationary Regime: yields rising with equity pressure, gold bid
    elif (spx_10y_label == "Negative" and gold_10y_label == "Positive"
          and vix_level != "low"):
        regime = "Inflationary Regime"
        description = (
            "Rising yields are pressuring equity valuations while gold is bid "
            "alongside rates — a classic stagflation signal. Real assets may "
            "outperform duration-sensitive growth."
        )

    # Growth Regime: equities and yields rising together, low vol, narrow dispersion
    elif (spx_10y_label == "Positive" and vix_level == "low"
          and disp_level != "high"):
        regime = "Growth Regime"
        description = (
            "Equities and yields are rising together with contained volatility "
            "and narrow sector dispersion. This is a supportive macro backdrop "
            "for risk assets."
        )

    # Transitional / Unstable: mixed signals across indicators
    else:
        # Check for more specific transitional patterns
        mixed_count = sum([
            spx_10y_label == "Neutral",
            spx_dxy_label == "Neutral",
            gold_10y_label == "Neutral",
        ])

        if mixed_count >= 2 or (vix_level == "elevated" and disp_level == "high"):
            regime = "Transitional / Unstable"
            description = (
                "Cross-asset correlations are sending mixed signals. Elevated "
                "dispersion and unclear directional cues suggest the market is "
                "between regimes — positioning may shift quickly."
            )
        elif spx_10y_label == "Positive" and vix_level == "elevated":
            regime = "Growth Regime"
            description = (
                "Equities and yields are co-moving higher, though volatility "
                "remains somewhat elevated. The growth backdrop is intact but "
                "bears watching."
            )
        elif spx_10y_label == "Negative" and vix_level == "low":
            regime = "Inflationary Regime"
            description = (
                "Yields and equities are diverging despite calm volatility. "
                "Rate sensitivity is rising — watch for duration rotation."
            )
        else:
            regime = "Transitional / Unstable"
            description = (
                "No dominant macro regime is clearly in control. Correlations, "
                "volatility, and dispersion are not aligned — expect choppy "
                "price action."
            )

    # Confidence: how many signals agree
    signal_alignment = _compute_confidence(vix_level, disp_level, spx_10y_label, regime)

    return {
        "regime": regime,
        "description": description,
        "confidence": signal_alignment,
        "signals": {
            "vix": vix,
            "dispersion": dispersion,
            "correlations": {
                "spx_vs_10y": spx_10y_label,
                "spx_vs_dxy": spx_dxy_label,
                "gold_vs_10y": gold_10y_label,
            },
        },
    }


def _compute_confidence(vix_level, disp_level, spx_10y_label, regime):
    """How many signals align with the assigned regime (high / moderate / low)."""
    score = 0

    if regime == "Growth Regime":
        if vix_level == "low":
            score += 1
        if disp_level in ("low", "moderate"):
            score += 1
        if spx_10y_label == "Positive":
            score += 1

    elif regime == "Inflationary Regime":
        if vix_level != "low":
            score += 1
        if spx_10y_label == "Negative":
            score += 1
        score += 1  # gold signal already checked in classification

    elif regime == "Liquidity Stress":
        if vix_level == "high":
            score += 1
        if disp_level == "high":
            score += 1
        if spx_10y_label == "Negative":
            score += 1

    elif regime == "Transitional / Unstable":
        # Low confidence by definition
        return "Low"

    if score >= 3:
        return "High"
    if score >= 2:
        return "Moderate"
    return "Low"

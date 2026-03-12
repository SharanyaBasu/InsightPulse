# backend/correlation_service.py
import numpy as np
import pandas as pd


_INTERPRETATIONS = {
    ("spx_vs_10y", "Positive"): "Reflation trade — equities and yields rising together",
    ("spx_vs_10y", "Negative"): "Flight to safety — equities falling, bonds bid",
    ("spx_vs_10y", "Neutral"):  "Decorrelated — no dominant macro regime",
    ("spx_vs_dxy", "Positive"): "Dollar strength coincides with equity gains (unusual)",
    ("spx_vs_dxy", "Negative"): "Classic risk-on: equities up, dollar soft",
    ("spx_vs_dxy", "Neutral"):  "Mixed dollar and equity signals",
    ("gold_vs_10y", "Positive"): "Unusual — gold and yields rising (stagflation risk)",
    ("gold_vs_10y", "Negative"): "Normal — rising real yields pressuring gold",
    ("gold_vs_10y", "Neutral"):  "Gold decorrelated from rates",
}


def _label(corr: float) -> str:
    if corr >= 0.3:
        return "Positive"
    if corr <= -0.3:
        return "Negative"
    return "Neutral"


def _compute_pair(key: str, pair_name: str, a: pd.Series, b: pd.Series) -> dict:
    combined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(combined) < 62:
        return {
            "pair": pair_name,
            "key": key,
            "correlation": None,
            "label": "Insufficient data",
            "interpretation": "Not enough overlapping data to compute correlation.",
            "trend_sparkline": [],
        }

    a_aligned = combined.iloc[:, 0]
    b_aligned = combined.iloc[:, 1]

    rolling = a_aligned.rolling(60).corr(b_aligned).dropna()
    current_corr = float(rolling.iloc[-1])
    trend_sparkline = [float(v) for v in rolling.tail(30).tolist()]
    label = _label(current_corr)

    return {
        "pair": pair_name,
        "key": key,
        "correlation": current_corr,
        "label": label,
        "interpretation": _INTERPRETATIONS.get((key, label), ""),
        "trend_sparkline": trend_sparkline,
    }


def build_correlations(market_df: pd.DataFrame, macro_daily: pd.DataFrame) -> list:
    results = []

    # S&P vs 10Y
    if "sp500" in market_df.columns and "ten_year_yield" in macro_daily.columns:
        spx_ret = market_df["sp500"].pct_change().dropna()
        ten_bps = macro_daily["ten_year_yield"].diff().dropna() * 100
        results.append(_compute_pair("spx_vs_10y", "S&P 500 vs 10Y Yield", spx_ret, ten_bps))
    else:
        results.append({
            "pair": "S&P 500 vs 10Y Yield", "key": "spx_vs_10y",
            "correlation": None, "label": "Insufficient data",
            "interpretation": "Missing source data.", "trend_sparkline": [],
        })

    # S&P vs DXY
    if "sp500" in market_df.columns and "usd_index" in market_df.columns:
        spx_ret = market_df["sp500"].pct_change().dropna()
        dxy_ret = market_df["usd_index"].pct_change().dropna()
        results.append(_compute_pair("spx_vs_dxy", "S&P 500 vs DXY", spx_ret, dxy_ret))
    else:
        results.append({
            "pair": "S&P 500 vs DXY", "key": "spx_vs_dxy",
            "correlation": None, "label": "Insufficient data",
            "interpretation": "Missing source data.", "trend_sparkline": [],
        })

    # Gold vs 10Y
    if "gold" in market_df.columns and "ten_year_yield" in macro_daily.columns:
        gold_ret = market_df["gold"].pct_change().dropna()
        ten_bps = macro_daily["ten_year_yield"].diff().dropna() * 100
        results.append(_compute_pair("gold_vs_10y", "Gold vs 10Y Yield", gold_ret, ten_bps))
    else:
        results.append({
            "pair": "Gold vs 10Y Yield", "key": "gold_vs_10y",
            "correlation": None, "label": "Insufficient data",
            "interpretation": "Missing source data.", "trend_sparkline": [],
        })

    return results

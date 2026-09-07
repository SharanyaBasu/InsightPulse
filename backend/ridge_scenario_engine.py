"""Ridge v1 Scenario Playground engine adapter.

Loads the trusted bundle from
``scenario_modeling/artifacts/scenario_playground_ridge/v1.0.0`` and maps the
research runtime result into ``ScenarioRunResponse`` for the UI.

Ridge v1 uses the bundled fixed market state (sample date 2026-06-11); it does
not refresh live market conditions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from scenario_modeling.src.scenario_runtime import load_scenario_model_bundle

# Repo root is the parent of backend/
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_PATH = (
    _REPO_ROOT
    / "scenario_modeling"
    / "artifacts"
    / "scenario_playground_ridge"
    / "v1.0.0"
)

# API (ScenarioRunRequest) -> model shock contract (MODEL_BUNDLE_HANDOFF.md)
API_TO_MODEL_SHOCKS = {
    "fed_funds_change_bps": "fed_funds_rate_change_bps",
    "cpi_surprise_pct": "cpi_surprise_pct",
    "oil_change_pct": "oil_price_change_pct",
    "gdp_surprise_pct": "gdp_growth_surprise_pct",
    "unemployment_change_pct": "unemployment_change_pct",
    "pmi_change": "pmi_change_points",
    "dxy_change_pct": "dxy_change_pct",
    # Handoff unit is VIX index points; UI currently labels this as %.
    "vix_change_pct": "vix_change_points",
}

ASSET_TARGETS = (
    "sp500",
    "nasdaq",
    "ten_year_yield_bps",
    "dxy",
    "gold",
    "oil",
)

SECTOR_TARGETS = (
    "technology",
    "energy",
    "financials",
    "utilities",
    "healthcare",
    "consumer_discretionary",
)

# Map research overall_research_confidence -> API ConfidenceLevel
_RESEARCH_CONFIDENCE_TO_API = {
    "OUTSIDE_HISTORICAL_SUPPORT": "Low",
    "VERY_LOW_RESEARCH_CONFIDENCE": "Low",
    "LOW_RESEARCH_CONFIDENCE_WITH_CONTEXT_WARNINGS": "Low",
    "LOW_RESEARCH_CONFIDENCE": "Low",
}

_API_CONFIDENCE_RANK = {"Low": 0, "Medium": 1, "High": 2}

_VIX_REGIME_LABELS = {
    "low_vix_lt_20": "Low VIX",
    "medium_vix_20_to_30": "Medium VIX",
    "high_vix_gte_30": "High VIX",
}


@lru_cache(maxsize=1)
def _load_runtime(bundle_path: str):
    return load_scenario_model_bundle(bundle_path)


def _round(value: float, digits: int = 2) -> float:
    rounded = round(float(value), digits)
    return 0.0 if rounded == 0.0 else rounded


def map_api_inputs_to_model_shocks(inputs: dict) -> dict[str, float]:
    """Translate ScenarioRunRequest fields into Ridge shock names."""

    shocks: dict[str, float] = {}
    for api_key, model_key in API_TO_MODEL_SHOCKS.items():
        shocks[model_key] = float(inputs.get(api_key, 0.0) or 0.0)
    return shocks


def _incremental_map(targets: dict[str, Any], names: tuple[str, ...]) -> dict[str, float]:
    return {
        name: _round(targets[name]["incremental_scenario_effect"])
        for name in names
    }


def _api_confidence(research_result: dict[str, Any]) -> str:
    """Take the most pessimistic per-target research confidence."""

    per_target = research_result.get("confidence", {}).get("targets", {})
    worst = "High"
    worst_rank = _API_CONFIDENCE_RANK[worst]
    for payload in per_target.values():
        status = payload.get("overall_research_confidence", "")
        mapped = _RESEARCH_CONFIDENCE_TO_API.get(status, "Low")
        rank = _API_CONFIDENCE_RANK[mapped]
        if rank < worst_rank:
            worst = mapped
            worst_rank = rank
    # Ridge v1 evidence is research-only; never claim High.
    if worst == "High":
        return "Medium"
    return worst


def _regime_label(research_result: dict[str, Any]) -> str:
    per_target = research_result.get("confidence", {}).get("targets", {})
    raw = None
    for name in ("sp500", *per_target):
        if name in per_target:
            raw = per_target[name].get("current_vix_regime")
            if raw:
                break
    if not raw:
        return "Fixed State (Jun 2026)"
    return _VIX_REGIME_LABELS.get(str(raw), str(raw).replace("_", " ").title())


def _top_incremental_driver_names(research_result: dict[str, Any], limit: int = 3) -> list[str]:
    explanations = research_result.get("explanations", {}).get("targets", {})
    sp500 = explanations.get("sp500", {})
    top = sp500.get("top_drivers", {})
    drivers: list[tuple[float, str]] = []
    for group in ("incremental_positive", "incremental_negative"):
        for item in top.get(group, []) or []:
            name = str(item.get("feature_name", ""))
            contribution = abs(float(item.get("contribution", 0.0) or 0.0))
            if name:
                drivers.append((contribution, name))
    drivers.sort(key=lambda row: row[0], reverse=True)
    seen: set[str] = set()
    ordered: list[str] = []
    for _, name in drivers:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name.replace("_", " "))
        if len(ordered) >= limit:
            break
    return ordered


def adapt_research_result(research_result: dict[str, Any]) -> dict[str, Any]:
    """Map Ridge runtime output to ScenarioRunResponse fields."""

    targets = research_result["targets"]
    assets = _incremental_map(targets, ASSET_TARGETS)
    sectors = _incremental_map(targets, SECTOR_TARGETS)
    confidence = _api_confidence(research_result)
    regime = _regime_label(research_result)

    context = research_result.get("context", {})
    sample_date = context.get("sample_date") or "2026-06-11"
    horizon = research_result.get("horizon_trading_days", 21)
    sp500 = assets["sp500"]
    direction = "higher" if sp500 >= 0 else "lower"

    summary = (
        f"Against the fixed {sample_date} market state, Ridge projects "
        f"equities {direction} ({sp500:+.2f}% S&P 500 incremental "
        f"{horizon}-trading-day effect; {confidence.lower()} research confidence)."
    )

    drivers = _top_incremental_driver_names(research_result)
    if drivers:
        driver_text = ", ".join(drivers[:-1])
        if len(drivers) > 1:
            driver_text = f"{driver_text}, and {drivers[-1]}"
        else:
            driver_text = drivers[0]
        driver_clause = f" Top incremental S&P 500 drivers: {driver_text}."
    else:
        driver_clause = " Macro shocks are near neutral relative to the fixed state."

    global_warning = research_result.get("global_warning") or (
        "Research only. Ridge outputs are predictive associations, not causal effects."
    )
    explanation = (
        f"Evaluated on the bundled {sample_date} state "
        f"(not live market conditions).{driver_clause} {global_warning}"
    )

    return {
        "summary": summary,
        "regime": regime,
        "confidence": confidence,
        "asset_deltas": assets,
        "sector_impacts": sectors,
        "explanation": explanation,
    }


class RidgeScenarioEngine:
    """ScenarioEngine backed by Scenario Playground Ridge v1.0.0."""

    def __init__(self, bundle_path: str | Path | None = None) -> None:
        self.bundle_path = str(Path(bundle_path or DEFAULT_BUNDLE_PATH).resolve())

    def predict(self, inputs: dict) -> dict:
        runtime = _load_runtime(self.bundle_path)
        shocks = map_api_inputs_to_model_shocks(inputs)
        research_result = runtime.predict(shocks)
        return adapt_research_result(research_result)

"""
Scenario Playground simulation service.

v1 uses a deterministic rule engine. The public entrypoint is run_scenario().
To add a model-backed engine later:
1. Implement ScenarioEngine.predict(inputs) -> dict
2. Load weights/artifacts inside that class
3. Set DEFAULT_ENGINE or pass engine= into run_scenario
"""

from __future__ import annotations

from typing import Protocol

# ---------------------------------------------------------------------------
# Contract keys (must match schemas.scenario.ScenarioRunResponse)
# ---------------------------------------------------------------------------

RESPONSE_KEYS = (
    "summary",
    "regime",
    "confidence",
    "asset_deltas",
    "sector_impacts",
    "explanation",
)

INPUT_KEYS = (
    "fed_funds_change_bps",
    "cpi_surprise_pct",
    "oil_change_pct",
    "gdp_surprise_pct",
    "unemployment_change_pct",
    "pmi_change",
    "dxy_change_pct",
    "vix_change_pct",
)

# Max absolute scale used to normalize each input to roughly [-1, 1]
INPUT_SCALES = {
    "fed_funds_change_bps": 100.0,
    "cpi_surprise_pct": 2.0,
    "oil_change_pct": 30.0,
    "gdp_surprise_pct": 5.0,
    "unemployment_change_pct": 2.0,
    "pmi_change": 10.0,
    "dxy_change_pct": 10.0,
    "vix_change_pct": 100.0,
}

# Asset impact coefficients: projected_change ~= sum(coeff * normalized_input)
# Percent assets use % points; ten_year_yield_bps uses basis points.
ASSET_COEFFICIENTS = {
    "sp500": {
        "fed_funds_change_bps": -1.2,
        "cpi_surprise_pct": -0.8,
        "oil_change_pct": -0.3,
        "gdp_surprise_pct": 1.4,
        "unemployment_change_pct": -1.0,
        "pmi_change": 1.1,
        "dxy_change_pct": -0.5,
        "vix_change_pct": -1.6,
    },
    "nasdaq": {
        "fed_funds_change_bps": -1.6,
        "cpi_surprise_pct": -1.0,
        "oil_change_pct": -0.4,
        "gdp_surprise_pct": 1.5,
        "unemployment_change_pct": -1.1,
        "pmi_change": 1.3,
        "dxy_change_pct": -0.6,
        "vix_change_pct": -1.9,
    },
    "ten_year_yield_bps": {
        "fed_funds_change_bps": 35.0,
        "cpi_surprise_pct": 25.0,
        "oil_change_pct": 8.0,
        "gdp_surprise_pct": 12.0,
        "unemployment_change_pct": -15.0,
        "pmi_change": 10.0,
        "dxy_change_pct": 5.0,
        "vix_change_pct": -8.0,
    },
    "dxy": {
        "fed_funds_change_bps": 0.8,
        "cpi_surprise_pct": 0.3,
        "oil_change_pct": -0.2,
        "gdp_surprise_pct": 0.4,
        "unemployment_change_pct": -0.3,
        "pmi_change": 0.2,
        "dxy_change_pct": 1.0,
        "vix_change_pct": 0.5,
    },
    "gold": {
        "fed_funds_change_bps": -0.7,
        "cpi_surprise_pct": 0.9,
        "oil_change_pct": 0.3,
        "gdp_surprise_pct": -0.2,
        "unemployment_change_pct": 0.4,
        "pmi_change": -0.2,
        "dxy_change_pct": -0.8,
        "vix_change_pct": 0.6,
    },
    "oil": {
        "fed_funds_change_bps": -0.3,
        "cpi_surprise_pct": 0.2,
        "oil_change_pct": 1.0,
        "gdp_surprise_pct": 0.6,
        "unemployment_change_pct": -0.4,
        "pmi_change": 0.5,
        "dxy_change_pct": -0.4,
        "vix_change_pct": -0.3,
    },
}

SECTOR_COEFFICIENTS = {
    "technology": {
        "fed_funds_change_bps": -1.8,
        "cpi_surprise_pct": -1.1,
        "oil_change_pct": -0.5,
        "gdp_surprise_pct": 1.6,
        "unemployment_change_pct": -1.0,
        "pmi_change": 1.4,
        "dxy_change_pct": -0.7,
        "vix_change_pct": -2.0,
    },
    "energy": {
        "fed_funds_change_bps": -0.2,
        "cpi_surprise_pct": 0.3,
        "oil_change_pct": 1.5,
        "gdp_surprise_pct": 0.5,
        "unemployment_change_pct": -0.3,
        "pmi_change": 0.4,
        "dxy_change_pct": -0.3,
        "vix_change_pct": -0.4,
    },
    "financials": {
        "fed_funds_change_bps": 0.6,
        "cpi_surprise_pct": -0.4,
        "oil_change_pct": -0.2,
        "gdp_surprise_pct": 1.0,
        "unemployment_change_pct": -0.8,
        "pmi_change": 0.7,
        "dxy_change_pct": 0.2,
        "vix_change_pct": -1.2,
    },
    "utilities": {
        "fed_funds_change_bps": -0.9,
        "cpi_surprise_pct": -0.2,
        "oil_change_pct": 0.1,
        "gdp_surprise_pct": -0.1,
        "unemployment_change_pct": 0.3,
        "pmi_change": -0.2,
        "dxy_change_pct": -0.1,
        "vix_change_pct": 0.4,
    },
    "healthcare": {
        "fed_funds_change_bps": -0.4,
        "cpi_surprise_pct": -0.3,
        "oil_change_pct": -0.1,
        "gdp_surprise_pct": 0.5,
        "unemployment_change_pct": -0.2,
        "pmi_change": 0.3,
        "dxy_change_pct": -0.2,
        "vix_change_pct": -0.5,
    },
    "consumer_discretionary": {
        "fed_funds_change_bps": -1.3,
        "cpi_surprise_pct": -0.9,
        "oil_change_pct": -0.4,
        "gdp_surprise_pct": 1.3,
        "unemployment_change_pct": -1.2,
        "pmi_change": 1.0,
        "dxy_change_pct": -0.4,
        "vix_change_pct": -1.5,
    },
}

ASSET_OUTPUT_BOUNDS = {
    "sp500": (-8.0, 8.0),
    "nasdaq": (-10.0, 10.0),
    "ten_year_yield_bps": (-80.0, 80.0),
    "dxy": (-5.0, 5.0),
    "gold": (-6.0, 6.0),
    "oil": (-12.0, 12.0),
}

SECTOR_OUTPUT_BOUNDS = (-10.0, 10.0)

DRIVER_LABELS = {
    "fed_funds_change_bps": "fed funds",
    "cpi_surprise_pct": "CPI surprise",
    "oil_change_pct": "oil prices",
    "gdp_surprise_pct": "GDP surprise",
    "unemployment_change_pct": "unemployment",
    "pmi_change": "PMI",
    "dxy_change_pct": "the dollar (DXY)",
    "vix_change_pct": "VIX",
}


class ScenarioEngine(Protocol):
    """Interface for scenario projection engines."""

    def predict(self, inputs: dict) -> dict:
        """Return a ScenarioRunResponse-shaped dict."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float, digits: int = 2) -> float:
    return round(value, digits)


def _normalize_inputs(inputs: dict) -> dict[str, float]:
    """Map raw macro shocks to roughly [-1, 1] using configured scales."""

    normalized: dict[str, float] = {}
    for key in INPUT_KEYS:
        raw = float(inputs.get(key, 0.0) or 0.0)
        scale = INPUT_SCALES[key]
        normalized[key] = _clamp(raw / scale, -1.0, 1.0)
    return normalized


def _dot_project(coefficients: dict[str, float], normalized: dict[str, float]) -> float:
    return sum(coefficients.get(key, 0.0) * normalized.get(key, 0.0) for key in INPUT_KEYS)


def _project_assets(normalized: dict[str, float]) -> dict[str, float]:
    assets: dict[str, float] = {}
    for asset, coeffs in ASSET_COEFFICIENTS.items():
        low, high = ASSET_OUTPUT_BOUNDS[asset]
        assets[asset] = _round(_clamp(_dot_project(coeffs, normalized), low, high))
    return assets


def _project_sectors(normalized: dict[str, float]) -> dict[str, float]:
    low, high = SECTOR_OUTPUT_BOUNDS
    sectors: dict[str, float] = {}
    for sector, coeffs in SECTOR_COEFFICIENTS.items():
        sectors[sector] = _round(_clamp(_dot_project(coeffs, normalized), low, high))
    return sectors


def _score_regime(normalized: dict[str, float]) -> tuple[str, str, float]:
    """Return (regime_label, confidence, strength)."""

    risk_off = (
        0.35 * normalized["vix_change_pct"]
        + 0.20 * normalized["fed_funds_change_bps"]
        + 0.15 * normalized["unemployment_change_pct"]
        + 0.10 * normalized["dxy_change_pct"]
        - 0.20 * normalized["gdp_surprise_pct"]
        - 0.20 * normalized["pmi_change"]
    )
    inflation = (
        0.45 * normalized["cpi_surprise_pct"]
        + 0.35 * normalized["oil_change_pct"]
        + 0.20 * normalized["fed_funds_change_bps"]
    )
    liquidity_stress = (
        0.50 * normalized["vix_change_pct"]
        + 0.30 * normalized["dxy_change_pct"]
        + 0.20 * normalized["fed_funds_change_bps"]
    )

    scores = {
        "Risk-Off": risk_off,
        "Risk-On": -risk_off,
        "Inflationary": inflation,
        "Liquidity Stress": liquidity_stress,
    }
    regime = max(scores, key=scores.get)
    strength = abs(scores[regime])

    if strength < 0.18:
        regime = "Transitional"
        confidence = "Low"
    elif strength < 0.40:
        confidence = "Medium"
    else:
        confidence = "High"

    return regime, confidence, strength


def _top_drivers(normalized: dict[str, float], limit: int = 3) -> list[tuple[str, float]]:
    ranked = sorted(
        normalized.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [(key, value) for key, value in ranked[:limit] if abs(value) > 0.05]


def _build_summary(regime: str, confidence: str, assets: dict[str, float]) -> str:
    equity = assets["sp500"]
    direction = "higher" if equity >= 0 else "lower"
    return (
        f"This scenario points to a {regime.lower()} backdrop "
        f"({confidence.lower()} confidence), with equities projected {direction} "
        f"({equity:+.1f}% S&P 500)."
    )


def _build_explanation(
    regime: str,
    drivers: list[tuple[str, float]],
    assets: dict[str, float],
    sectors: dict[str, float],
) -> str:
    if not drivers:
        return (
            "Macro shocks are near zero, so projected asset and sector impacts "
            "remain limited in this deterministic baseline."
        )

    parts = []
    for key, value in drivers:
        label = DRIVER_LABELS.get(key, key)
        direction = "higher" if value > 0 else "lower"
        parts.append(f"{direction} {label}")

    driver_text = ", ".join(parts[:-1])
    if len(parts) > 1:
        driver_text = f"{driver_text}, and {parts[-1]}"
    else:
        driver_text = parts[0]

    top_sector = max(sectors.items(), key=lambda item: abs(item[1]))
    sector_name = top_sector[0].replace("_", " ")
    sector_move = top_sector[1]

    return (
        f"The {regime.lower()} call is driven mainly by {driver_text}. "
        f"Under these shocks, {sector_name} shows the largest sector move "
        f"({sector_move:+.1f}%), while the S&P 500 is projected at "
        f"{assets['sp500']:+.1f}%."
    )


def _ensure_response_shape(result: dict) -> dict:
    missing = [key for key in RESPONSE_KEYS if key not in result]
    if missing:
        raise ValueError(f"Engine result missing keys: {missing}")
    return result


class DeterministicRuleEngine:
    """v1 rule-based scenario engine. Replace via ScenarioEngine later."""

    def predict(self, inputs: dict) -> dict:
        """Project scenario impacts from macro shocks.

        Args:
            inputs: Macro shock dict matching ScenarioRunRequest fields.

        Returns:
            ScenarioRunResponse-shaped result dict.
        """

        normalized = _normalize_inputs(inputs)
        assets = _project_assets(normalized)
        sectors = _project_sectors(normalized)
        regime, confidence, _strength = _score_regime(normalized)
        drivers = _top_drivers(normalized)

        return _ensure_response_shape(
            {
                "summary": _build_summary(regime, confidence, assets),
                "regime": regime,
                "confidence": confidence,
                "asset_deltas": assets,
                "sector_impacts": sectors,
                "explanation": _build_explanation(regime, drivers, assets, sectors),
            }
        )


DEFAULT_ENGINE: ScenarioEngine = DeterministicRuleEngine()


def run_scenario(
    inputs: dict,
    engine: ScenarioEngine | None = None,
) -> dict:
    """Run a scenario simulation.

    Args:
        inputs: Macro shock dict matching ScenarioRunRequest fields.
        engine: Optional engine override. Defaults to DeterministicRuleEngine.

    Returns:
        ScenarioRunResponse-shaped result dict.
    """

    active_engine = engine or DEFAULT_ENGINE
    return active_engine.predict(inputs)

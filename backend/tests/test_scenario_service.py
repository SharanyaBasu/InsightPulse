from ridge_scenario_engine import (
    RidgeScenarioEngine,
    map_api_inputs_to_model_shocks,
)
from scenario_service import DeterministicRuleEngine, run_scenario


ZERO_INPUTS = {
    "fed_funds_change_bps": 0,
    "cpi_surprise_pct": 0,
    "oil_change_pct": 0,
    "gdp_surprise_pct": 0,
    "unemployment_change_pct": 0,
    "pmi_change": 0,
    "dxy_change_pct": 0,
    "vix_change_pct": 0,
}


def test_map_api_inputs_to_model_shocks():
    shocks = map_api_inputs_to_model_shocks(
        {
            **ZERO_INPUTS,
            "fed_funds_change_bps": -25,
            "oil_change_pct": 5,
            "vix_change_pct": 10,
            "pmi_change": -2,
        }
    )

    assert shocks == {
        "fed_funds_rate_change_bps": -25.0,
        "cpi_surprise_pct": 0.0,
        "oil_price_change_pct": 5.0,
        "gdp_growth_surprise_pct": 0.0,
        "unemployment_change_pct": 0.0,
        "pmi_change_points": -2.0,
        "dxy_change_pct": 0.0,
        "vix_change_points": 10.0,
    }


def test_run_scenario_returns_required_keys():
    result = run_scenario(ZERO_INPUTS)

    assert set(result) >= {
        "summary",
        "regime",
        "confidence",
        "asset_deltas",
        "sector_impacts",
        "explanation",
    }
    assert result["confidence"] in {"Low", "Medium", "High"}
    assert "sp500" in result["asset_deltas"]
    assert "technology" in result["sector_impacts"]
    assert "2026-06-11" in result["summary"]


def test_run_scenario_is_deterministic():
    payload = {
        **ZERO_INPUTS,
        "vix_change_pct": 40,
        "oil_change_pct": 10,
    }

    first = run_scenario(payload)
    second = run_scenario(payload)

    assert first == second


def test_risk_off_inputs_pressure_equities():
    result = run_scenario(
        {
            **ZERO_INPUTS,
            "vix_change_pct": 80,
            "fed_funds_change_bps": 50,
            "unemployment_change_pct": 1.0,
            "pmi_change": -5,
            "gdp_surprise_pct": -2,
        }
    )

    assert result["asset_deltas"]["sp500"] < 0
    assert result["asset_deltas"]["nasdaq"] < 0
    assert result["confidence"] in {"Low", "Medium", "High"}
    assert isinstance(result["regime"], str) and result["regime"]


def test_higher_oil_lifts_energy_and_oil_asset():
    result = run_scenario({**ZERO_INPUTS, "oil_change_pct": 25})

    assert result["asset_deltas"]["oil"] > 0
    assert result["sector_impacts"]["energy"] > 0


def test_neutral_shocks_have_zero_incremental_effects():
    result = run_scenario(ZERO_INPUTS)

    assert result["asset_deltas"]["sp500"] == 0.0
    assert result["sector_impacts"]["technology"] == 0.0


def test_engine_override_is_used():
    class FakeEngine:
        def predict(self, inputs: dict) -> dict:
            return {
                "summary": "fake",
                "regime": "Risk-On",
                "confidence": "High",
                "asset_deltas": {
                    "sp500": 1.0,
                    "nasdaq": 1.0,
                    "ten_year_yield_bps": 0.0,
                    "dxy": 0.0,
                    "gold": 0.0,
                    "oil": 0.0,
                },
                "sector_impacts": {
                    "technology": 1.0,
                    "energy": 0.0,
                    "financials": 0.0,
                    "utilities": 0.0,
                    "healthcare": 0.0,
                    "consumer_discretionary": 0.0,
                },
                "explanation": "injected engine",
            }

    result = run_scenario(ZERO_INPUTS, engine=FakeEngine())

    assert result["summary"] == "fake"
    assert result["explanation"] == "injected engine"


def test_default_engine_is_ridge():
    engine = RidgeScenarioEngine()
    result = engine.predict(ZERO_INPUTS)

    assert result["confidence"] in {"Low", "Medium", "High"}
    assert "fixed" in result["summary"].lower() or "2026-06-11" in result["summary"]
    assert result["asset_deltas"]["sp500"] == 0.0


def test_deterministic_rule_engine_still_available():
    engine = DeterministicRuleEngine()
    result = engine.predict(ZERO_INPUTS)

    assert result["regime"] == "Transitional"
    assert result["confidence"] == "Low"

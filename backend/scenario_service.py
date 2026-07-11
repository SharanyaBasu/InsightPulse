"""
Scenario Playground contract stub.

Returns a response that matches `ScenarioRunResponse` so the endpoint can be
tested with curl / FastAPI docs before deterministic models are wired in.
"""

from schemas.scenario import ScenarioRunRequest, ScenarioRunResponse


def build_contract_stub_result(request: ScenarioRunRequest) -> ScenarioRunResponse:
    """Return a contract-shaped stub. Not a real impact model."""
    del request  # accepted for contract validation; ignored until models land
    return ScenarioRunResponse(
        summary="This scenario suggests a mildly risk-off market environment.",
        regime="Transitional / Unstable",
        confidence="Low",
        asset_deltas={
            "sp500": -1.2,
            "nasdaq": -1.8,
            "ten_year_yield_bps": 15,
            "dxy": 0.7,
            "gold": 0.5,
            "oil": 2.1,
        },
        sector_impacts={
            "technology": -2.0,
            "energy": 1.4,
            "financials": -0.6,
            "utilities": 0.3,
            "healthcare": 0.1,
            "consumer_discretionary": -1.1,
        },
        explanation=(
            "Higher VIX and oil prices usually create risk-off pressure. "
            "Technology is negatively affected, while energy benefits from higher oil prices."
        ),
    )

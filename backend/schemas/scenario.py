"""
Scenario Playground API contract.

Endpoint: POST /api/scenario/run

This module is the source of truth for request/response shapes used by the
Scenario Playground UI (summary strip, regime/confidence, delta cards,
sector heatmap, explanation).

Note: GET /api/scenario is a separate live day-over-day market snapshot and
must NOT be used for playground simulation.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ScenarioRunRequest(BaseModel):
    """Macro shocks submitted by the Scenario Playground input panel."""

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_vix_units(cls, values):
        """Reject ambiguous legacy percent inputs rather than silently ignoring them."""
        if isinstance(values, dict) and "vix_change_pct" in values:
            raise ValueError(
                "vix_change_pct is no longer supported. Submit vix_change_points "
                "in index points: VIX 20 to 25 means +5, not +25%."
            )
        return values

    fed_funds_change_bps: float = Field(
        0,
        ge=-100,
        le=100,
        description="Fed funds rate change in basis points.",
        examples=[-50],
    )
    cpi_surprise_pct: float = Field(
        0,
        ge=-2,
        le=2,
        description="CPI / inflation surprise in percentage points.",
        examples=[-0.2],
    )
    oil_change_pct: float = Field(
        0,
        ge=-30,
        le=30,
        description="Oil price percent change.",
        examples=[5],
    )
    gdp_surprise_pct: float = Field(
        0,
        ge=-5,
        le=5,
        description="GDP growth surprise in percentage points.",
        examples=[-0.5],
    )
    unemployment_change_pct: float = Field(
        0,
        ge=-2,
        le=2,
        description="Unemployment rate change in percentage points.",
        examples=[0.2],
    )
    pmi_change: float = Field(
        0,
        ge=-10,
        le=10,
        description="PMI change in index points.",
        examples=[-3],
    )
    dxy_change_pct: float = Field(
        0,
        ge=-10,
        le=10,
        description="DXY / USD percent change.",
        examples=[1],
    )
    vix_change_points: float = Field(
        0,
        ge=-50,
        le=100,
        description="VIX change in index points (20 to 25 means +5 points).",
        examples=[10],
    )

    model_config = {
        "extra": "forbid",
        "strict": True,
        "allow_inf_nan": False,
        "json_schema_extra": {
            "examples": [
                {
                    "fed_funds_change_bps": -50,
                    "cpi_surprise_pct": -0.2,
                    "oil_change_pct": 5,
                    "gdp_surprise_pct": -0.5,
                    "unemployment_change_pct": 0.2,
                    "pmi_change": -3,
                    "dxy_change_pct": 1,
                    "vix_change_points": 10,
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

ConfidenceLevel = Literal["Low", "Medium", "High"]


class AssetProjections(BaseModel):
    """Projected asset impacts for ScenarioDeltaCards.

    Percent fields are percent changes. `ten_year_yield_bps` is basis points.
    """

    sp500: float = Field(..., description="S&P 500 projected % change.")
    nasdaq: float = Field(..., description="NASDAQ projected % change.")
    ten_year_yield_bps: float = Field(..., description="US 10Y yield change in bps.")
    dxy: float = Field(..., description="Dollar index projected % change.")
    gold: float = Field(..., description="Gold projected % change.")
    oil: float = Field(..., description="Oil projected % change.")


class SectorProjections(BaseModel):
    """Projected sector % impacts for SectorHeatmap."""

    technology: float
    energy: float
    financials: float
    utilities: float
    healthcare: float
    consumer_discretionary: float


class ScenarioContext(BaseModel):
    """Baseline and horizon attached to the model result, not the current clock."""

    baseline_date: str
    horizon_trading_days: int = Field(gt=0)
    effect_type: Literal["incremental_vs_baseline"]
    neutral_inputs: bool


class CalculationInput(BaseModel):
    """One submitted scenario input, echoed with its display metadata."""

    label: str
    value: float
    unit: str


class TargetCalculation(BaseModel):
    """Auditable prediction arithmetic for one model target."""

    label: str
    unit: str
    baseline_prediction: float
    scenario_prediction: float
    difference: float = Field(
        ...,
        description="Scenario prediction minus baseline prediction.",
    )


class CalculationDetails(BaseModel):
    """Model identity, inputs, and target-level arithmetic for a scenario run."""

    model_version: str
    bundle_version: str
    baseline_date: str
    horizon_trading_days: int = Field(gt=0)
    submitted_inputs: dict[str, CalculationInput]
    targets: dict[str, TargetCalculation]


class ScenarioWarning(BaseModel):
    """User-facing warning preserved from model diagnostics."""

    code: str
    severity: Literal["low", "medium", "high"]
    message: str


class ScenarioRunResponse(BaseModel):
    """Simulation result consumed by ScenarioPage result panels."""

    context: ScenarioContext | None = None
    calculation_details: CalculationDetails
    warnings: list[ScenarioWarning] = Field(default_factory=list)

    summary: str = Field(
        ...,
        description="One-sentence interpretation of the scenario.",
    )
    regime: str = Field(
        ...,
        description="Named macro regime label for this scenario.",
        examples=["Liquidity Stress"],
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence in the regime / projection call.",
    )
    asset_deltas: AssetProjections = Field(
        ...,
        description="Projected asset impacts (maps to delta cards).",
    )
    sector_impacts: SectorProjections = Field(
        ...,
        description="Projected sector impacts (maps to heatmap).",
    )
    explanation: str = Field(
        ...,
        description="Short paragraph explaining the drivers.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "This scenario suggests a mildly risk-off market environment.",
                    "regime": "Liquidity Stress",
                    "confidence": "Medium",
                    "asset_deltas": {
                        "sp500": -1.2,
                        "nasdaq": -1.8,
                        "ten_year_yield_bps": 15,
                        "dxy": 0.7,
                        "gold": 0.5,
                        "oil": 2.1,
                    },
                    "sector_impacts": {
                        "technology": -2.0,
                        "energy": 1.4,
                        "financials": -0.6,
                        "utilities": 0.3,
                        "healthcare": 0.1,
                        "consumer_discretionary": -1.1,
                    },
                    "explanation": (
                        "Higher VIX and oil prices usually create risk-off pressure. "
                        "Technology is negatively affected, while energy benefits from higher oil prices."
                    ),
                }
            ]
        }
    }

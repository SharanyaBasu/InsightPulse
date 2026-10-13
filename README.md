# InsightPulse

## Scenario Playground v1

The Scenario Playground lets users explore how a set of hypothetical macroeconomic shocks could affect major assets and market sectors. On the `/scenario` page, adjust the macro inputs and select **Run Scenario**. Saving the inputs is optional: the run uses the values currently shown in the form, including unsaved changes.

Results remain hidden until a scenario is run. Each run sends the eight inputs to `POST /api/scenario/run` and displays a scenario summary, regime and confidence label, projected asset moves, sector impacts, and an explanation of the principal drivers.

### How the model works

Version 1 uses a deterministic, rule-based model. Each input is normalized, assigned an explainable sensitivity, combined with the other shocks, and clamped to a reasonable output range. The same inputs therefore always produce the same results. The rules are designed to preserve intuitive directional relationships, such as higher VIX and rate shocks pressuring equities and an oil shock moving the oil projection in the same direction.

The v1 results are scenario projections, not predictions of what will happen. The model:

- is not a machine-learning or AI forecast;
- is not trained on historical outcomes;
- does not call live market-data or external APIs when running a scenario; and
- does not account for every market interaction or changing market conditions.

The output is for exploration and educational use only. It is not financial advice and should not be used as the sole basis for an investment decision.

### Inputs

All request fields are numeric.

| Field | Meaning | Unit |
| --- | --- | --- |
| `fed_funds_change_bps` | Hypothetical change in the federal funds rate | Basis points (bps) |
| `cpi_surprise_pct` | CPI result relative to expectations | Percentage points |
| `oil_change_pct` | Change in the oil price | Percent (%) |
| `gdp_surprise_pct` | GDP growth result relative to expectations | Percentage points |
| `unemployment_change_pct` | Change in the unemployment rate | Percentage points |
| `pmi_change` | Change in the PMI index | Index points |
| `dxy_change_pct` | Change in the U.S. Dollar Index | Percent (%) |
| `vix_change_pct` | Change in the VIX | Percent (%) |

Positive values represent increases or upside surprises; negative values represent decreases or downside surprises.

### Response

| Field | Meaning |
| --- | --- |
| `summary` | Short interpretation of the projected market environment |
| `regime` | Rule-derived scenario regime, such as risk-on, risk-off, or transitional |
| `confidence` | Qualitative confidence level for the scenario classification |
| `asset_deltas` | Projected changes for the S&P 500, NASDAQ, 10-year yield, DXY, gold, and oil |
| `sector_impacts` | Projected percentage impacts for technology, energy, financials, utilities, healthcare, and consumer discretionary |
| `explanation` | Explanation of the input shocks and rules that most influenced the result |

Values under `asset_deltas` are percentages except for `ten_year_yield_bps`, which is expressed in basis points. Values under `sector_impacts` are percentages.

### Manual API test

Start the backend from the `backend` directory:

```bash
python -m uvicorn app:app --reload
```

Then submit a risk-off scenario from another terminal:

```bash
curl -X POST http://127.0.0.1:8000/api/scenario/run \
  -H "Content-Type: application/json" \
  -d '{
    "fed_funds_change_bps": 50,
    "cpi_surprise_pct": 0.5,
    "oil_change_pct": 10,
    "gdp_surprise_pct": -1,
    "unemployment_change_pct": 0.5,
    "pmi_change": -4,
    "dxy_change_pct": 2,
    "vix_change_pct": 30
  }'
```

A successful request returns HTTP 200 and JSON containing all response fields described above. The interactive contract and request tester are also available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Running a scenario requires no database records or external API keys.

### Future direction

A future version may replace or augment the fixed rules with sensitivities estimated from historical regressions. That approach could calibrate asset reactions from observed data while retaining explicit units, bounded outputs, reproducibility, and an explanation of the factors driving each projection.

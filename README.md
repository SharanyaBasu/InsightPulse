# InsightPulse

## Teammate setup

All application changes are on `main`. Use **Python 3.13** (the pinned inference
runtime requires at least 3.12) and **Node 22.12+**. Frontend versions are recorded
in the lockfile; use `npm ci`.

From the repository root, in macOS/Linux or Git Bash:

```bash
git pull origin main
python3.13 -m venv .venv
source .venv/bin/activate
# Windows Git Bash: source .venv/Scripts/activate
cd backend
python -m pip install -r requirements.txt
cp .env.example .env
# Fill in your own API keys in .env before ingestion or summary generation.
python init_db.py
cd ..
python -m scenario_modeling.install_bundle
python -m scenario_modeling.run_bundle_smoke_test --bundle scenario_modeling/artifacts/scenario_playground_ridge/v1.0.0 --forbid-training-data
```

On Windows, create the environment with `py -3.13 -m venv .venv` and activate it
in PowerShell with `.\.venv\Scripts\Activate.ps1`; use `Copy-Item` instead of `cp`.

Start the backend with the environment active:

```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

In another terminal from the repository root:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://127.0.0.1:5173/>. Scenario calculations need the model bundle but no
API keys. Dashboard data comes from the local database; the repository's existing
database snapshot may be old. Refresh it with `python data_fetcher.py` from
`backend/` after configuring the provider keys. Gemini requires `GEMINI_API_KEY`;
summary generation uses stored market evidence and shows its observation dates.

The model downloader checks the pinned release checksum and handles the ZIP's
Windows paths on macOS/Linux. The bundle, virtual environments, `.env`, browser
storage, and locally refreshed database are not included in this source update.
Market data and LLM wording can vary; the fixed model bundle is reproducible.

Validation: run `python -m pytest tests -q` from `backend/`, then `npm test`,
`npm run lint`, and `npm run build` from `frontend/`. For browser tests, run
`npx playwright install chromium` once and then `npm run test:browser`.
See [backend setup](backend/README.md) and [frontend setup](frontend/README.md).

## Scenario Playground v1

The Scenario Playground lets users explore how a set of hypothetical macroeconomic shocks could affect major assets and market sectors. On the `/scenario` page, adjust the macro inputs and select **Run Scenario**. Saving the inputs is optional: the run uses the values currently shown in the form, including unsaved changes.

Results remain hidden until a scenario is run. Each run sends the eight inputs to `POST /api/scenario/run` and displays a scenario summary, regime and confidence label, projected asset moves, sector impacts, and an explanation of the principal drivers.

### How the model works

Version 1 uses 12 trained Ridge pipelines from the separately versioned model
bundle. It evaluates eight hypothetical shocks against a fixed June 11, 2026
market state over a 21-trading-day horizon. Results show the additional impact
relative to that baseline; zero inputs therefore produce zero incremental impact.
The bundled model uses historical predictive associations, not causal estimates,
and does not retrieve live data or retrain during inference. The earlier rule
engine remains available for explicit tests but is not the default API engine.

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
| `vix_change_points` | Change in the VIX | Index points |

Positive values represent increases or upside surprises; negative values represent decreases or downside surprises.

### Response

| Field | Meaning |
| --- | --- |
| `summary` | Short interpretation of the projected market environment |
| `regime` | Regime associated with the bundled historical model state |
| `confidence` | Qualitative confidence level for the scenario classification |
| `asset_deltas` | Projected changes for the S&P 500, NASDAQ, 10-year yield, DXY, gold, and oil |
| `sector_impacts` | Projected percentage impacts for technology, energy, financials, utilities, healthcare, and consumer discretionary |
| `explanation` | Explanation of the model drivers and research limitations |

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
    "vix_change_points": 30
  }'
```

A successful request returns HTTP 200 and JSON containing all response fields described above. The interactive contract and request tester are also available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). Running a scenario requires no database records or external API keys.

### Model versioning

Future models or market-state snapshots should be distributed as new, validated
bundle versions. Do not silently replace the pinned release artifacts.

Scenario input units: CPI, GDP growth, and unemployment shocks use percentage
points; Fed funds uses basis points; PMI and VIX use index points; oil and DXY
use percent changes. For example, unemployment from 4% to 4.5% is +0.5
percentage points, and VIX from 20 to 25 is +5 index points. The legacy
`vix_change_pct` API field is rejected with HTTP 422; clients must explicitly
submit `vix_change_points`. Saved browser inputs from the previous unit contract
retain other shocks but reset VIX to zero with a notice.

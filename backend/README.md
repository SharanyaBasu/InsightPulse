# Backend Setup

Basic setup for running the InsightPulse FastAPI backend.

## Install

Use Python 3.12 or newer: the pinned SciPy 1.18 runtime does not support
Python 3.11.

```bash
cd backend
python3.13 -m venv .venv
source .venv/bin/activate
# Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt
```

`requirements.txt` installs the repo-root `scenario_modeling` package in editable mode
(`-e ..`) so FastAPI can `import scenario_modeling` when launched from `backend/`.
No `PYTHONPATH` or `sys.path` edits are required.

For the Ridge model binary, place the full GitHub release folder at:

```text
scenario_modeling/artifacts/scenario_playground_ridge/v1.0.0/
```

From the repository root, run `python -m scenario_modeling.install_bundle` to
download and verify the pinned release. See `scenario_modeling/MODEL_BUNDLE_HANDOFF.md`
for bundle details and smoke tests.

## Environment

Create a `.env` file in `backend/`:

```bash
FRED_API_KEY=your_fred_api_key
COINGECKO_API_KEY=your_coingecko_api_key
FINNHUB_API_KEY=your_finnhub_api_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash
DATABASE_URL=your_database_url
```

`DATABASE_URL` is optional. If it is missing, the backend uses local SQLite at `backend/insightpulse.db`.

`GEMINI_API_KEY` is required for the LLM daily summary endpoint to work.

`GEMINI_MODEL` is optional. If it is missing, the backend uses `gemini-3.5-flash`.

Note: `.vscode/settings.json` only configures an optional SQLTools connection to the local SQLite database. The backend runtime database is controlled by `DATABASE_URL` in `backend/.env`.

## Run

Create database tables:

```bash
python init_db.py
```

Fetch and load market/macro data:

```bash
python data_fetcher.py
```

Start the API server:

```bash
python -m uvicorn app:app --reload
```

Backend runs at:

```bash
http://127.0.0.1:8000
```

API docs:

```bash
http://127.0.0.1:8000/docs
```

### Local preview environment

A separate Python 3.13 environment at the repository root (`.venv/`) can
run the backend while preserving an existing `backend/.venv` environment:

```bash
# From the repository root
python3.13 -m venv .venv
cd backend
../.venv/bin/python -m pip install -r requirements.txt
../.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Observation dates and scenario context

Overview responses include `data_dates` and per-card/indicator `observed_on`
values. `last_updated` is snapshot generation time, not the observation date.
The next macro ingestion creates a `source_observations` table and stores the
latest original FRED observation dates before forward filling. Existing legacy
records have unknown source dates; the API does not infer dates from repeated
values or database row dates. Dates identify observation periods, not release
timestamps.

Scenario results include `context.baseline_date`, `horizon_trading_days`,
`effect_type`, and `neutral_inputs`. Only exactly zero submitted shocks are
neutral; small effects that round to zero remain identified as non-neutral.

## Daily summary reliability

Normal `/api/summary/daily` requests summarize stored market data and calculate
local metrics if needed; they do not refresh every external provider. Explicit
`refresh_data=true` still requests source ingestion. The response includes the
underlying evidence date range so stored observations are not presented as live.

The summary generator uses `GEMINI_MODEL` (default `gemini-3.5-flash`). On model
unavailability or a deadline error it tries `GEMINI_FALLBACK_MODEL` (default
`gemini-2.5-flash`); set that variable to an empty string to disable fallback.
Both models use the same citation guardrail and semantic verification. Individual
model calls have a 30-second timeout, with a 130-second overall generation budget.
The browser stops waiting after 180 seconds and exposes a retryable error.

A process-local lock prevents duplicate summary generation in the current
single-worker deployment. A concurrent request receives HTTP 503 and Retry-After.
Multiple workers would require a shared job lock. Successful summaries are cached
by day; failed refreshes retain the previous summary in the browser.

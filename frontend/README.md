# Frontend Setup

Basic setup for running the InsightPulse React frontend.

## Install

```bash
cd frontend
npm ci
```

## Run

Start the backend first in a separate terminal:

```bash
cd backend
source ../.venv/bin/activate
# Windows Git Bash: source ../.venv/Scripts/activate
python -m uvicorn app:app --reload
```

Then start the frontend:

```bash
cd frontend
npm run dev
```

Frontend runs at the URL printed by Vite, usually:

```bash
http://localhost:5173
```

The frontend proxies `/api` requests to:

```bash
http://127.0.0.1:8000
```

## Phase 1 checks

Run `npm test` for input persistence and date/state helpers, and `npm run lint`
and `npm run build` for static validation. Run `npx playwright install chromium`
once, then `npm run test:browser` for scenario and freshness browser checks.
Alternatively, use `PLAYWRIGHT_CHANNEL=chrome npm run test:browser` with an
installed Google Chrome. Browser tests mock provider responses and use a local
Vite server; they do not require API keys or write market data.

Scenario results retain the submitted inputs and completion time. Edits flag the
results as outdated; reset cancels the active run. Data observation dates are
separate from the browser's last successful API check.

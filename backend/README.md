# Backend Setup

Basic setup for running the InsightPulse FastAPI backend.

## Install

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

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

# Backend Setup

Basic setup for running the InsightPulse FastAPI backend.

## Install

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Create a `.env` file in `backend/`:

```bash
FRED_API_KEY=your_fred_api_key
DATABASE_URL=your_database_url
```

`DATABASE_URL` is optional. If it is missing, the backend uses local SQLite at `backend/insightpulse.db`.

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

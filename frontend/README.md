# Frontend Setup

Basic setup for running the InsightPulse React frontend.

## Install

```bash
cd frontend
npm install
```

## Run

Start the backend first in a separate terminal:

```bash
cd backend
source .venv/bin/activate
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

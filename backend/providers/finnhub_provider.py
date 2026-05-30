"""Finnhub data provider for current equity and ETF quotes."""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from schemas.source_types import FINNHUB


FINNHUB_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "SPY",
    "QQQ",
    "DIA",
]

FINNHUB_QUOTE_URL = "https://finnhub.io/api/v1/quote"
REQUEST_TIMEOUT_SECONDS = 15


def _utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fetch_equity_quotes():
    load_dotenv()

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is missing from environment")

    timestamp = _utc_now_naive()
    rows = []

    for symbol in FINNHUB_SYMBOLS:
        params = {
            "symbol": symbol,
            "token": api_key,
        }
        response = requests.get(
            FINNHUB_QUOTE_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        quote = response.json()

        rows.append(
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "price": quote.get("c"),
                "change": quote.get("d"),
                "percent_change": quote.get("dp"),
                "high": quote.get("h"),
                "low": quote.get("l"),
                "open": quote.get("o"),
                "previous_close": quote.get("pc"),
                "volume": None,
                "source": FINNHUB,
            }
        )
        print(f"Loaded Finnhub quote: {symbol}")

    return rows

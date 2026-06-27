"""CoinGecko data provider for current crypto market quotes."""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from schemas.source_types import COINGECKO


COINGECKO_COIN_IDS = [
    "bitcoin",
    "ethereum",
    "solana",
    "binancecoin",
    "ripple",
]

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
REQUEST_TIMEOUT_SECONDS = 15


def _utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fetch_crypto_quotes():
    load_dotenv()

    api_key = os.getenv("COINGECKO_API_KEY")
    if not api_key:
        raise RuntimeError("COINGECKO_API_KEY is missing from environment")

    params = {
        "vs_currency": "usd",
        "ids": ",".join(COINGECKO_COIN_IDS),
        "order": "market_cap_desc",
        "per_page": len(COINGECKO_COIN_IDS),
        "page": 1,
        "sparkline": "false",
        "price_change_percentage": "24h",
    }
    headers = {
        "x-cg-demo-api-key": api_key,
    }

    response = requests.get(
        COINGECKO_MARKETS_URL,
        params=params,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    timestamp = _utc_now_naive()
    rows = []

    for coin in response.json():
        rows.append(
            {
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name"),
                "timestamp": timestamp,
                "price": coin.get("current_price"),
                "market_cap": coin.get("market_cap"),
                "volume_24h": coin.get("total_volume"),
                "change_24h": coin.get("price_change_percentage_24h"),
                "source": COINGECKO,
            }
        )

    print(f"Loaded {len(rows)} CoinGecko crypto quotes.")
    return rows

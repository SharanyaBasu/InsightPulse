"""Database write helpers for ingestion workflows."""

from datetime import datetime, timezone

import pandas as pd

from models import CryptoQuote, EquityQuote, IngestionRun, MacroData, MarketData


def _to_native_scalar(value):
    if pd.isna(value):
        return None

    return float(value)


def ingest_market_data(db, df) -> int:
    db.query(MarketData).delete()

    for date, row in df.iterrows():
        normalized = {col: _to_native_scalar(row[col]) for col in df.columns}

        entry = MarketData(
            date=date.date(),
            # --- Equities ---
            sp500=normalized.get("SP500"),
            nasdaq=normalized.get("NASDAQ"),
            vgk=normalized.get("VGK"),
            ewj=normalized.get("EWJ"),
            eem=normalized.get("EEM"),
            mtum=normalized.get("MTUM"),
            vtv=normalized.get("VTV"),
            iwf=normalized.get("IWF"),

            # --- Sector ETFs (NEW) ---
            xly=normalized.get("XLY"),
            xlp=normalized.get("XLP"),
            xle=normalized.get("XLE"),
            xlf=normalized.get("XLF"),
            xlv=normalized.get("XLV"),
            xlk=normalized.get("XLK"),
            xli=normalized.get("XLI"),
            xlb=normalized.get("XLB"),
            xlre=normalized.get("XLRE"),
            xlc=normalized.get("XLC"),

            # --- Bonds / Credit ---
            irx=normalized.get("IRX"),
            fvx=normalized.get("FVX"),
            tnx=normalized.get("TNX"),
            hyg=normalized.get("HYG"),
            lqd=normalized.get("LQD"),
            tlt=normalized.get("TLT"),
            vix=normalized.get("VIX"),

            # --- Commodities ---
            oil=normalized.get("Oil"),
            natgas=normalized.get("NatGas"),
            gold=normalized.get("Gold"),
            silver=normalized.get("Silver"),
            copper=normalized.get("Copper"),

            # --- FX ---
            usd_index=normalized.get("USD_Index"),
            eurusd=normalized.get("EURUSD"),
            gbpusd=normalized.get("GBPUSD"),
            audusd=normalized.get("AUDUSD"),
            usdjpy=normalized.get("USDJPY"),
            usdchf=normalized.get("USDCHF"),
            cew=normalized.get("CEW"),

            # --- Crypto ---
            bitcoin=normalized.get("Bitcoin"),
            ethereum=normalized.get("Ethereum"),
        )
        db.add(entry)

    db.commit()
    rows_written = len(df)
    print(f"Saved {rows_written} rows to the database.")
    return rows_written


def ingest_macro_data(db, df) -> int:
    db.query(MacroData).delete()

    for date, row in df.iterrows():
        normalized = {col: _to_native_scalar(row[col]) for col in df.columns}

        entry = MacroData(
            date=date.date(),
            cpi=normalized["CPI"],
            unemployment=normalized["Unemployment"],
            fed_funds_rate=normalized["Fed_Funds_Rate"],
            gdp=normalized["GDP"],
            two_year_yield=normalized["DGS2"],
            ten_year_yield=normalized["DGS10"],
        )
        db.add(entry)

    db.commit()
    rows_written = len(df)
    print(f"Saved {rows_written} macro rows to database.")
    return rows_written


def ingest_crypto_quotes(db, rows) -> int:
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    for row in rows:
        quote = CryptoQuote(
            symbol=row["symbol"],
            name=row["name"],
            timestamp=row["timestamp"],
            price=_to_native_scalar(row["price"]),
            market_cap=_to_native_scalar(row["market_cap"]),
            volume_24h=_to_native_scalar(row["volume_24h"]),
            change_24h=_to_native_scalar(row["change_24h"]),
            source=row["source"],
            created_at=timestamp,
            updated_at=timestamp,
        )
        db.add(quote)

    db.commit()
    rows_written = len(rows)
    print(f"Saved {rows_written} crypto quote rows to database.")
    return rows_written


def ingest_equity_quotes(db, rows) -> int:
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    for row in rows:
        quote = EquityQuote(
            symbol=row["symbol"],
            timestamp=row["timestamp"],
            price=_to_native_scalar(row["price"]),
            change=_to_native_scalar(row["change"]),
            percent_change=_to_native_scalar(row["percent_change"]),
            high=_to_native_scalar(row["high"]),
            low=_to_native_scalar(row["low"]),
            open=_to_native_scalar(row["open"]),
            previous_close=_to_native_scalar(row["previous_close"]),
            volume=_to_native_scalar(row["volume"]),
            source=row["source"],
            created_at=timestamp,
            updated_at=timestamp,
        )
        db.add(quote)

    db.commit()
    rows_written = len(rows)
    print(f"Saved {rows_written} equity quote rows to database.")
    return rows_written


def record_ingestion_run(
    db,
    source,
    job_name,
    started_at,
    finished_at,
    status,
    rows_fetched,
    rows_written,
    error_message=None,
):
    run = IngestionRun(
        source=source,
        job_name=job_name,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        rows_fetched=rows_fetched,
        rows_written=rows_written,
        error_message=error_message,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    db.add(run)
    db.commit()

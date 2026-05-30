from datetime import datetime, timezone

from db import SessionLocal
from providers.fred_provider import fetch_macro_data
from providers.yahoo_provider import fetch_market_data
from schemas.source_types import FRED, YAHOO
from services.ingestion_service import (
    ingest_macro_data,
    ingest_market_data,
    record_ingestion_run,
)


def get_market_data():
    started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db = SessionLocal()
    df = None

    try:
        df = fetch_market_data()
        rows_written = ingest_market_data(db, df)
        record_ingestion_run(
            db=db,
            source=YAHOO,
            job_name="market_data_refresh",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).replace(tzinfo=None),
            status="success",
            rows_fetched=len(df),
            rows_written=rows_written,
        )
        return df
    except Exception as exc:
        db.rollback()
        record_ingestion_run(
            db=db,
            source=YAHOO,
            job_name="market_data_refresh",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).replace(tzinfo=None),
            status="failed",
            rows_fetched=len(df) if df is not None else 0,
            rows_written=0,
            error_message=str(exc),
        )
        raise
    finally:
        db.close()


def get_macro_data():
    """
    Fetch key macroeconomic indicators from the Federal Reserve (FRED).
    Returns a pandas DataFrame with monthly/quarterly data and saves to DB.
    """
    started_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db = SessionLocal()
    macro_df = None

    try:
        macro_df = fetch_macro_data()
        rows_written = ingest_macro_data(db, macro_df)
        record_ingestion_run(
            db=db,
            source=FRED,
            job_name="macro_data_refresh",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).replace(tzinfo=None),
            status="success",
            rows_fetched=len(macro_df),
            rows_written=rows_written,
        )
        return macro_df
    except Exception as exc:
        db.rollback()
        record_ingestion_run(
            db=db,
            source=FRED,
            job_name="macro_data_refresh",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).replace(tzinfo=None),
            status="failed",
            rows_fetched=len(macro_df) if macro_df is not None else 0,
            rows_written=0,
            error_message=str(exc),
        )
        raise
    finally:
        db.close()



if __name__ == "__main__":
    get_market_data()
    get_macro_data()


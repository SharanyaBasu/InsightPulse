from sqlalchemy import Column, Integer, Float, Date, JSON, String
from sqlalchemy import Column, Date, DateTime, Float, Integer, String, Text
from db import Base

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)

    # --- Equities ---
    sp500 = Column(Float)
    nasdaq = Column(Float)
    vgk = Column(Float)
    ewj = Column(Float)
    eem = Column(Float)
    mtum = Column(Float)
    vtv = Column(Float)
    iwf = Column(Float)

    # --- Sector ETFs (NEW) ---
    xly = Column(Float)   # Consumer Discretionary
    xlp = Column(Float)   # Consumer Staples
    xle = Column(Float)   # Energy
    xlf = Column(Float)   # Financials
    xlv = Column(Float)   # Health Care
    xlk = Column(Float)   # Technology
    xli = Column(Float)   # Industrials
    xlb = Column(Float)   # Materials
    xlre = Column(Float)  # Real Estate
    xlc = Column(Float)   # Communication Services

    # --- Bonds / Credit ---
    irx = Column(Float)
    fvx = Column(Float)
    tnx = Column(Float)
    hyg = Column(Float)
    lqd = Column(Float)
    tlt = Column(Float)
    vix = Column(Float)

    # --- Commodities ---
    oil = Column(Float)
    natgas = Column(Float)
    gold = Column(Float)
    silver = Column(Float)
    copper = Column(Float)

    # --- FX ---
    usd_index = Column(Float)
    eurusd = Column(Float)
    gbpusd = Column(Float)
    audusd = Column(Float)
    usdjpy = Column(Float)
    usdchf = Column(Float)
    cew = Column(Float)

    # --- Crypto ---
    bitcoin = Column(Float)
    ethereum = Column(Float)


class MacroData(Base):
    __tablename__ = "macro_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)

    # --- Core Macro Indicators ---
    cpi = Column(Float)               # Inflation (CPIAUCSL)
    unemployment = Column(Float)      # %
    fed_funds_rate = Column(Float)    # %
    gdp = Column(Float)               # Real GDP

    # --- Added U.S. Treasury Yields ---
    two_year_yield = Column(Float)    # DGS2 (%)
    ten_year_yield = Column(Float)    # DGS10 (%)


class MarketState(Base):
    __tablename__ = "market_state"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True)
    data = Column(JSON)

class MarketSummary(Base):
    __tablename__ = "market_summary"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True)
    summary = Column(JSON) 
class EquityQuote(Base):
    __tablename__ = "equity_quotes"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String)
    timestamp = Column(DateTime, index=True)
    price = Column(Float)
    change = Column(Float)
    percent_change = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open = Column(Float)
    previous_close = Column(Float)
    volume = Column(Float)
    source = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class CryptoQuote(Base):
    __tablename__ = "crypto_quotes"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String)
    name = Column(String)
    timestamp = Column(DateTime, index=True)
    price = Column(Float)
    market_cap = Column(Float)
    volume_24h = Column(Float)
    change_24h = Column(Float)
    source = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String)
    job_name = Column(String)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    status = Column(String)
    rows_fetched = Column(Integer)
    rows_written = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime)


class CalculatedMetric(Base):
    __tablename__ = "calculated_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String)
    category = Column(String)
    timestamp = Column(DateTime, index=True)
    value = Column(Float)
    window = Column(String)
    source_dependencies = Column(Text)
    calculation_version = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

from sqlalchemy import Column, Integer, Float, Date
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

from sqlalchemy import Column, Integer, Float, Date
from db import Base

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)

    # ---- Equities ----
    sp500 = Column(Float)
    nasdaq = Column(Float)

    # ---- Commodities ----
    gold = Column(Float)
    oil = Column(Float)
    copper = Column(Float)

    # ---- Forex ----
    usd_index = Column(Float)
    eurusd = Column(Float)
    usdjpy = Column(Float)

    # ---- Fixed Income / Bonds ----
    yield_10y = Column(Float)
    yield_2y = Column(Float)
    yield_3m = Column(Float)

    # ---- Crypto ----
    bitcoin = Column(Float)
    ethereum = Column(Float)


class MacroData(Base):
    __tablename__ = "macro_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    cpi = Column(Float)              # Inflation (Consumer Price Index)
    unemployment = Column(Float)     # %
    fed_funds_rate = Column(Float)   # %
    gdp = Column(Float)              # Real GDP (billions)

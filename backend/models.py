from sqlalchemy import Column, Integer, Float, String, Date
from db import Base

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    sp500 = Column(Float)
    nasdaq = Column(Float)
    gold = Column(Float)
    oil = Column(Float)
    usd_index = Column(Float)
    yield_10y = Column(Float)

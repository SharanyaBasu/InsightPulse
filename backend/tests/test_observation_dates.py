"""Source dates must survive forward filling and must not be invented for old rows."""
from datetime import date

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import MacroData, SourceObservation
from overview_service import observation_date, macro_observation_dates
from services.ingestion_service import ingest_macro_data


def test_missing_values_do_not_move_observation_date():
    values = pd.Series([1, None, None], index=pd.date_range('2026-06-01', periods=3))
    assert observation_date(values) == '2026-06-01'
    assert observation_date(values * float('nan')) is None


def test_original_macro_date_survives_forward_filled_rows(monkeypatch):
    engine = create_engine('sqlite://')
    MacroData.__table__.create(engine)
    session = sessionmaker(bind=engine)
    monkeypatch.setattr('overview_service.SessionLocal', session)
    assert macro_observation_dates() == {}
    frame = pd.DataFrame({key: [1.0, 1.0] for key in ['CPI', 'Unemployment', 'Fed_Funds_Rate', 'GDP', 'DGS2', 'DGS10']}, index=pd.date_range('2026-06-10', periods=2))
    frame.attrs['observation_dates'] = {'CPI': '2026-05-01', 'DGS10': '2026-06-10'}
    with session() as db:
        ingest_macro_data(db, frame)
        assert db.get(SourceObservation, 'CPI').observed_on == date(2026, 5, 1)
    assert macro_observation_dates() == {'CPI': '2026-05-01', 'DGS10': '2026-06-10'}
    frame.attrs.clear()
    with session() as db:
        ingest_macro_data(db, frame)
    assert macro_observation_dates() == {}


def test_provider_captures_dates_before_forward_fill(monkeypatch):
    from providers import fred_provider

    class FakeFred:
        def __init__(self, **kwargs):
            pass

        def get_series(self, code):
            if code == 'CPIAUCSL':
                return pd.Series([300.0], index=pd.to_datetime(['2026-05-01']))
            return pd.Series([1.0, 2.0], index=pd.to_datetime(['2026-05-01', '2026-06-11']))

    monkeypatch.setenv('FRED_API_KEY', 'test')
    monkeypatch.setattr(fred_provider, 'Fred', FakeFred)
    monkeypatch.setattr(fred_provider.time, 'sleep', lambda _: None)
    frame = fred_provider.fetch_macro_data()
    assert frame.iloc[-1]['CPI'] == 300.0
    assert frame.index[-1].date() == date(2026, 6, 11)
    assert frame.attrs['observation_dates']['CPI'] == '2026-05-01'

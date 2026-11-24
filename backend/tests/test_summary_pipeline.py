"""Regression tests for data-loading failure and duplicate summary jobs."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app
from models import MarketSummary
from providers import yahoo_provider


def test_market_loader_selects_requested_ticker_even_with_extra_columns(monkeypatch):
    def download(symbol, **kwargs):
        return pd.DataFrame({('Close', symbol): [10.0] * 8, ('Close', 'UNRELATED'): [999.0] * 8}, index=pd.date_range('2026-10-01', periods=8))
    monkeypatch.setattr(yahoo_provider, 'download_history', download)
    result = yahoo_provider.fetch_market_data()
    assert result.shape == (8, 39)
    assert (result == 10.0).all().all()
    assert result.columns.is_unique


def test_missing_requested_ticker_is_not_replaced_by_unrelated_price():
    with pytest.raises(ValueError, match='missing Close'):
        yahoo_provider.extract_close(pd.DataFrame({('Close', 'OTHER'): [999]}), '^GSPC')


def test_summary_state_uses_stored_data_without_provider_ingestion(monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    ingest = MagicMock(side_effect=AssertionError('must not download providers'))
    calculate = MagicMock()
    state = {'as_of': '2026-10-13', 'evidence': {'sp500': {'value': 1}}}
    monkeypatch.setattr(app, '_run_daily_ingestion', ingest)
    monkeypatch.setattr(app, 'get_calculated_metrics', calculate)
    monkeypatch.setattr(app, 'build_market_state', lambda _: state)
    monkeypatch.setattr(app, '_save_market_state', lambda *_: None)
    assert app._get_or_compute_today_market_state(db, False) == state
    calculate.assert_called_once()
    ingest.assert_not_called()


@pytest.fixture
def summary_db(monkeypatch):
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    MarketSummary.__table__.create(engine)
    monkeypatch.setattr(app, 'SessionLocal', sessionmaker(bind=engine))
    monkeypatch.setattr(app, '_get_or_compute_today_market_state', lambda *_, **__: {'evidence': {'x': {'timestamp': '2026-10-09T00:00:00'}}})
    yield
    engine.dispose()


def test_concurrent_summary_is_rejected_and_success_is_cached(summary_db, monkeypatch):
    entered, finish = Event(), Event()
    calls = []
    def generate(_):
        calls.append(1)
        entered.set()
        assert finish.wait(5)
        return {'headline': 'Test summary'}
    monkeypatch.setattr(app, 'generate_summary_with_guardrail', generate)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(app.summary_daily)
        try:
            assert entered.wait(5)
            with pytest.raises(HTTPException) as error:
                app.summary_daily()
            assert error.value.status_code == 503
            assert error.value.headers['Retry-After'] == '5'
        finally:
            finish.set()
        assert first.result(timeout=5)['headline'] == 'Test summary'
    cached = app.summary_daily()
    assert len(calls) == 1
    assert cached['data_as_of']['latest'] == '2026-10-09'


def test_generation_failure_releases_lock_and_returns_503(summary_db, monkeypatch):
    def fail(_):
        raise RuntimeError('provider unavailable')
    monkeypatch.setattr(app, 'generate_summary_with_guardrail', fail)
    with pytest.raises(HTTPException) as error:
        app.summary_daily()
    assert error.value.status_code == 503
    assert not app._summary_lock.locked()
    monkeypatch.setattr(app, 'generate_summary_with_guardrail', lambda _: {'headline': 'Recovered'})
    assert app.summary_daily()['headline'] == 'Recovered'


def test_overloaded_gemini_uses_configured_fallback(monkeypatch):
    from types import SimpleNamespace
    from google.api_core.exceptions import ServiceUnavailable
    import llm_summary
    names = []
    def model(name):
        names.append(name)
        if name == 'primary':
            return SimpleNamespace(generate_content=MagicMock(side_effect=ServiceUnavailable('busy')))
        return SimpleNamespace(generate_content=MagicMock(return_value=SimpleNamespace(text='{"headline":"ok"}')))
    monkeypatch.setenv('GEMINI_MODEL', 'primary')
    monkeypatch.setenv('GEMINI_FALLBACK_MODEL', 'secondary')
    monkeypatch.setattr(llm_summary, '_get_model', model)
    response, name = llm_summary._generate_content('test', {}, 0)
    assert name == 'secondary'
    assert names == ['primary', 'secondary']
    assert 'ok' in response.text


def test_expired_generation_budget_makes_no_provider_call(monkeypatch):
    import llm_summary
    model = MagicMock()
    monkeypatch.setattr(llm_summary, '_get_model', model)
    token = llm_summary._GENERATION_DEADLINE.set(llm_summary.time.monotonic() - 1)
    try:
        with pytest.raises(TimeoutError):
            llm_summary._generate_content('test', {}, 0)
        model.assert_not_called()
    finally:
        llm_summary._GENERATION_DEADLINE.reset(token)


def test_cached_summary_can_be_read_even_while_generation_lock_is_held(summary_db, monkeypatch):
    monkeypatch.setattr(app, 'generate_summary_with_guardrail', lambda _: {'headline': 'Cached'})
    app.summary_daily()
    with app._summary_lock:
        assert app.summary_daily()['headline'] == 'Cached'

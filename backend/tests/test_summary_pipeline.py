"""Regression tests for market data loading and summary fallback behavior."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

import app
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



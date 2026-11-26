"""Public reads stay open. Provider calls and stored dumps stay closed."""
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app
from models import MarketState, MarketSummary


@pytest.fixture
def client():
    with TestClient(app.app) as client:
        yield client


def test_health_does_not_call_providers(client, monkeypatch):
    provider = Mock(side_effect=AssertionError('health must not call a provider'))
    monkeypatch.setattr(app, 'get_market_data', provider)
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}
    provider.assert_not_called()


@pytest.mark.parametrize('path', [
    '/debug/market_state',
    '/debug/summary',
    '/api/compute/daily',
    '/api/market-data',
    '/api/summary/daily?refresh_data=true',
    '/api/overview?force_refresh=true',
])
def test_operator_routes_stay_closed_without_a_matching_token(client, monkeypatch, path):
    monkeypatch.setenv('ADMIN_TOKEN', 'secret')
    provider = Mock(side_effect=AssertionError('closed routes must not call a provider'))
    monkeypatch.setattr(app, 'get_market_data', provider)
    monkeypatch.setattr(app, '_run_daily_ingestion', provider)
    monkeypatch.setattr(app, 'build_overview_snapshot', provider)
    response = client.get(path, headers={'X-Admin-Token': 'wrong'})
    assert response.status_code == 404
    provider.assert_not_called()


def test_operator_routes_stay_closed_when_no_token_is_configured(client, monkeypatch):
    monkeypatch.delenv('ADMIN_TOKEN', raising=False)
    assert client.get('/debug/summary', headers={'X-Admin-Token': 'secret'}).status_code == 404


def test_operator_token_opens_debug_and_compute(client, monkeypatch):
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    MarketState.__table__.create(engine)
    MarketSummary.__table__.create(engine)
    monkeypatch.setattr(app, 'SessionLocal', sessionmaker(bind=engine))
    monkeypatch.setenv('ADMIN_TOKEN', 'secret')
    monkeypatch.setattr(app, '_run_daily_ingestion', lambda: [{'step': 'market_data', 'status': 'success', 'rows': 1}])
    monkeypatch.setattr(app, 'build_market_state', lambda db: {'as_of': '2026-11-25', 'evidence': {'sp500': 1}})
    headers = {'X-Admin-Token': 'secret'}
    assert client.get('/debug/market_state', headers=headers).status_code == 200
    assert client.get('/debug/summary', headers=headers).json() == []
    response = client.get('/api/compute/daily', headers=headers)
    assert response.status_code == 200
    assert response.json()['market_state']['as_of'] == '2026-11-25'


def test_operator_token_opens_live_market_download(client, monkeypatch):
    monkeypatch.setenv('ADMIN_TOKEN', 'secret')
    frame = pd.DataFrame([
        {'SP500': 1, 'NASDAQ': 1, 'Gold': 1, 'USD_Index': 1},
        {'SP500': 2, 'NASDAQ': 2, 'Gold': 1, 'USD_Index': 1},
    ])
    monkeypatch.setattr(app, 'get_market_data', lambda: frame)
    response = client.get('/api/market-data', headers={'X-Admin-Token': 'secret'})
    assert response.status_code == 200
    assert response.json()['sentiment'] == 'bullish'


def test_public_overview_still_reads_the_snapshot(client, monkeypatch):
    monkeypatch.setattr(app, 'build_overview_snapshot', lambda force_refresh=False: {'cards': []})
    response = client.get('/api/overview')
    assert response.status_code == 200
    assert response.json() == {'cards': []}

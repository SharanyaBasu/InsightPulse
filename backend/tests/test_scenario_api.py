"""Exercise actual HTTP validation and Ridge response serialization."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import app
from ridge_scenario_engine import RidgeScenarioEngine

BOUNDS = {
    'fed_funds_change_bps': (-100, 100), 'cpi_surprise_pct': (-2, 2),
    'oil_change_pct': (-30, 30), 'gdp_surprise_pct': (-5, 5),
    'unemployment_change_pct': (-2, 2), 'pmi_change': (-10, 10),
    'dxy_change_pct': (-10, 10), 'vix_change_points': (-50, 100),
}


@pytest.fixture
def client():
    with TestClient(app.app) as client:
        yield client


@pytest.mark.parametrize('field,limits', BOUNDS.items())
def test_each_input_accepts_both_boundaries(client, field, limits):
    for value in limits:
        response = client.post('/api/scenario/run', json={field: value})
        assert response.status_code == 200, response.text
        assert response.json()['context']['neutral_inputs'] is False


@pytest.mark.parametrize('field,limits', BOUNDS.items())
def test_each_input_rejects_outside_bounds_without_running_model(client, monkeypatch, field, limits):
    engine = Mock(side_effect=AssertionError('invalid requests must not reach inference'))
    monkeypatch.setattr(app, 'run_scenario_engine', engine)
    for value in [limits[0] - 0.01, limits[1] + 0.01, None, True, '5', 'NaN', 'Infinity']:
        response = client.post('/api/scenario/run', json={field: value})
        assert response.status_code == 422, (field, value, response.text)
    engine.assert_not_called()


@pytest.mark.parametrize('payload', [{'oil_change_typo': 10}, {'vix_change_pct': 25}, [], 'invalid'])
def test_malformed_contract_is_rejected(client, payload):
    assert client.post('/api/scenario/run', json=payload).status_code == 422


def test_invalid_json_is_rejected(client):
    assert client.post('/api/scenario/run', content='{broken', headers={'Content-Type': 'application/json'}).status_code == 422


def test_neutral_and_changed_scenarios_return_real_model_results(client):
    neutral = client.post('/api/scenario/run', json={})
    assert neutral.status_code == 200
    baseline = neutral.json()
    assert all(value == 0 for value in baseline['asset_deltas'].values())
    assert all(value == 0 for value in baseline['sector_impacts'].values())
    changed = client.post('/api/scenario/run', json={'oil_change_pct': 25})
    assert changed.status_code == 200
    result = changed.json()
    assert result['asset_deltas']['oil'] > 0
    assert result['sector_impacts']['energy'] > 0
    assert result['context'] == {**baseline['context'], 'neutral_inputs': False}
    assert result['context']['baseline_date'] == '2026-06-11'
    assert result['context']['horizon_trading_days'] == 21


def test_missing_model_returns_actionable_service_unavailable(client, monkeypatch, tmp_path):
    engine = RidgeScenarioEngine(tmp_path / 'missing-bundle')
    monkeypatch.setattr(app, 'run_scenario_engine', engine.predict)
    response = client.post('/api/scenario/run', json={})
    assert response.status_code == 503
    assert 'model bundle' in response.json()['detail'].lower()
    assert str(tmp_path) not in response.text

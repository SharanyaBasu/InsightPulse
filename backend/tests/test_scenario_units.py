"""Contract tests for explicit macro input units and legacy VIX rejection."""
import pytest
from pydantic import ValidationError

from schemas.scenario import ScenarioRunRequest
from ridge_scenario_engine import map_api_inputs_to_model_shocks


def test_units_survive_request_to_model_mapping():
    """VIX 20 to 25 is five points; percentage-point surprises are not rescaled."""
    request = ScenarioRunRequest(
        vix_change_points=5, cpi_surprise_pct=0.5,
        unemployment_change_pct=0.5, gdp_surprise_pct=0.5,
        fed_funds_change_bps=-25, oil_change_pct=5, dxy_change_pct=1, pmi_change=2,
    )
    assert map_api_inputs_to_model_shocks(request.model_dump()) == {
        'vix_change_points': 5, 'cpi_surprise_pct': 0.5,
        'unemployment_change_pct': 0.5, 'gdp_growth_surprise_pct': 0.5,
        'fed_funds_rate_change_bps': -25, 'oil_price_change_pct': 5,
        'dxy_change_pct': 1, 'pmi_change_points': 2,
    }


@pytest.mark.parametrize('payload', [
    {'vix_change_pct': 25}, {'vix_change_pct': 0},
    {'vix_change_pct': 25, 'vix_change_points': 5},
])
def test_legacy_vix_field_is_explicitly_rejected(payload):
    """Old clients receive an actionable error instead of a neutral VIX shock."""
    with pytest.raises(ValidationError, match='Submit vix_change_points'):
        ScenarioRunRequest.model_validate(payload)


@pytest.mark.parametrize('value', [-51, 101, None])
def test_vix_points_bounds_are_validated(value):
    """The renamed field retains validation at the API boundary."""
    with pytest.raises(ValidationError):
        ScenarioRunRequest(vix_change_points=value)


def test_openapi_contract_uses_points():
    """The documented API advertises the same unit as the model."""
    properties = ScenarioRunRequest.model_json_schema()['properties']
    assert 'vix_change_pct' not in properties
    assert 'index points' in properties['vix_change_points']['description']

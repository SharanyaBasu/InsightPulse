"""An unavailable upstream must not silently become fabricated market evidence."""
from unittest.mock import Mock

import pandas as pd
import pytest

from providers import fred_provider, yahoo_provider


@pytest.mark.parametrize('response', [pd.DataFrame(), TimeoutError('provider timed out')])
def test_yahoo_total_failure_is_explicit(monkeypatch, response):
    download = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
    monkeypatch.setattr(yahoo_provider, 'download_history', download)
    with pytest.raises(RuntimeError, match='No valid data'):
        yahoo_provider.fetch_market_data()


def test_missing_fred_key_does_not_start_provider(monkeypatch):
    monkeypatch.delenv('FRED_API_KEY', raising=False)
    provider = Mock()
    monkeypatch.setattr(fred_provider, 'Fred', provider)
    with pytest.raises(RuntimeError, match='FRED_API_KEY'):
        fred_provider.fetch_macro_data()
    provider.assert_not_called()


def test_partial_fred_failure_is_not_presented_as_complete_snapshot(monkeypatch):
    def series(code):
        if code == 'CPIAUCSL':
            raise TimeoutError('CPI upstream unavailable')
        return pd.Series([1.0], index=pd.to_datetime(['2026-01-01']))
    monkeypatch.setenv('FRED_API_KEY', 'test-only')
    monkeypatch.setattr(fred_provider, 'Fred', lambda **_: Mock(get_series=series))
    monkeypatch.setattr(fred_provider.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='Missing required FRED series: CPI'):
        fred_provider.fetch_macro_data()

"""Release integrity, safe installation and historical-state diagnostics."""
import hashlib
import io
import json
import zipfile
from unittest.mock import Mock

import pytest

from ridge_scenario_engine import DEFAULT_BUNDLE_PATH
from scenario_modeling import install_bundle as installer
from scenario_modeling.src import scenario_runtime as runtime


def test_release_checksum_failure_does_not_modify_existing_bundle(tmp_path):
    destination = tmp_path / 'installed'
    destination.mkdir()
    existing = destination / 'model_manifest.json'
    existing.write_text('preserve me')
    archive = tmp_path / 'corrupt.zip'
    archive.write_bytes(b'not the verified release')
    with pytest.raises(ValueError, match='checksum mismatch'):
        installer.install_bundle(destination, archive)
    assert existing.read_text() == 'preserve me'
    assert list(destination.iterdir()) == [existing]


def archive_fixture(tmp_path, monkeypatch, filename):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr(filename, b'test payload')
    data = buffer.getvalue()
    # Synthetic fixtures exercise extraction; the production release hash stays pinned.
    monkeypatch.setattr(installer, 'SHA256', hashlib.sha256(data).hexdigest())
    path = tmp_path / 'fixture.zip'
    path.write_bytes(data)
    return path


def test_windows_release_paths_install_idempotently(tmp_path, monkeypatch):
    archive = archive_fixture(tmp_path, monkeypatch, 'v1.0.0\\model_manifest.json')
    destination = tmp_path / 'installed'
    installer.install_bundle(destination, archive)
    assert (destination / 'model_manifest.json').read_bytes() == b'test payload'
    assert installer.install_bundle(destination, archive) == destination
    (destination / 'model_manifest.json').write_text('different')
    with pytest.raises(ValueError, match='Existing bundle differs'):
        installer.install_bundle(destination, archive)
    assert (destination / 'model_manifest.json').read_text() == 'different'


@pytest.mark.parametrize('filename', ['../escaped', '/v1.0.0/absolute', 'v1.0.0/../escaped'])
def test_unsafe_archive_paths_leave_no_installation(tmp_path, monkeypatch, filename):
    archive = archive_fixture(tmp_path, monkeypatch, filename)
    with pytest.raises(ValueError, match='Unexpected release path'):
        installer.install_bundle(tmp_path / 'installed', archive)
    assert not (tmp_path / 'installed').exists()
    assert not (tmp_path / 'escaped').exists()


@pytest.mark.parametrize('contents', [None, b'changed model bytes'])
def test_missing_or_corrupt_payload_is_rejected_before_deserialization(tmp_path, monkeypatch, contents):
    name = 'scenario_model_bundle.joblib'
    (tmp_path / 'artifact_checksums.json').write_text(json.dumps({
        'algorithm': 'sha256', 'files': {name: hashlib.sha256(b'expected bytes').hexdigest()},
    }))
    if contents is not None:
        (tmp_path / name).write_bytes(contents)
    deserialize = Mock(side_effect=AssertionError('must verify before loading'))
    monkeypatch.setattr(runtime.joblib, 'load', deserialize)
    with pytest.raises(runtime.BundleValidationError, match='missing|SHA-256 mismatch'):
        runtime.load_scenario_model_bundle(tmp_path)
    deserialize.assert_not_called()


@pytest.mark.parametrize('diagnostic_date,status,age', [
    ('2026-06-18', 'FRESH', 7), ('2026-06-19', 'AGING', 8),
    ('2026-07-11', 'AGING', 30), ('2026-07-12', 'STALE', 31),
])
def test_old_model_state_is_not_reported_as_current(diagnostic_date, status, age):
    model = runtime.load_scenario_model_bundle(DEFAULT_BUNDLE_PATH)
    result = model.predict({name: 0.0 for name in model.scenario_input_names}, diagnostic_as_of_date=diagnostic_date)
    freshness = result['confidence']['state_freshness']
    assert freshness['state_sample_date'] == '2026-06-11'
    assert freshness['calendar_age_days'] == age
    assert freshness['freshness_status'] == status
    assert freshness['included_in_confidence'] is True
    if status == 'STALE':
        assert all(target['warnings']['state_freshness'] for target in result['confidence']['targets'].values())

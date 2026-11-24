"""Tests use isolated SQLite and mocked providers, never developer credentials."""
import os
import socket

import pytest

# Set before test modules import db/app or load .env.
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
os.environ['DATABASE_URL'] = 'sqlite://'


@pytest.fixture(autouse=True)
def forbid_external_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Network calls are forbidden in backend tests; mock the provider.')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)

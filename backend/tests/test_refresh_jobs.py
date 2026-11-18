"""Scheduled refresh keeps the last snapshot and runs one job at a time."""

from datetime import date
from threading import Event, Thread

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app
import refresh_service
from models import MarketState, MarketSummary
from refresh_service import refresh_is_due


def _bind(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    MarketSummary.__table__.create(engine)
    MarketState.__table__.create(engine)
    sessions = sessionmaker(bind=engine)
    monkeypatch.setattr(app, "SessionLocal", sessions)
    monkeypatch.setattr(refresh_service, "SessionLocal", sessions)
    return sessions


def _save_summary(sessions, headline, day=date(2026, 11, 3)):
    db = sessions()
    db.add(MarketSummary(date=day, summary={"headline": headline, "regime_summary": {"text": "Saved.", "evidence_ids": []}}))
    db.commit()
    db.close()


def test_opening_the_summary_does_not_call_providers_or_the_model(monkeypatch):
    sessions = _bind(monkeypatch)
    _save_summary(sessions, "Saved snapshot")
    monkeypatch.setattr(app, "_run_daily_ingestion", lambda: (_ for _ in ()).throw(AssertionError("must not ingest")))
    monkeypatch.setattr(app, "generate_summary_with_guardrail", lambda _: (_ for _ in ()).throw(AssertionError("must not generate")))

    payload = app.summary_daily()

    assert payload["headline"] == "Saved snapshot"
    assert payload["refresh"]["in_progress"] is False
    assert payload["refresh"]["last_success_at"] is None


def test_refresh_request_returns_the_saved_snapshot_without_running_it(monkeypatch):
    sessions = _bind(monkeypatch)
    _save_summary(sessions, "Kept")
    called = []
    monkeypatch.setattr(refresh_service, "start_refresh", lambda trigger, perform: called.append(trigger) or True)

    payload = app.summary_daily(refresh_data=True)

    assert payload["headline"] == "Kept"
    assert payload["refresh"]["in_progress"] is True
    assert called == ["manual"]


def test_failed_refresh_keeps_the_snapshot_and_records_the_error(monkeypatch):
    sessions = _bind(monkeypatch)
    _save_summary(sessions, "Kept")
    monkeypatch.setattr(app, "_run_daily_ingestion", lambda: (_ for _ in ()).throw(RuntimeError("yahoo down")))

    refresh_service.execute_refresh("manual", app._perform_refresh)
    payload = app.summary_daily()

    assert payload["headline"] == "Kept"
    assert payload["refresh"]["last_error"] == "yahoo down"
    assert payload["refresh"]["last_success_at"] is None
    saved = sessions().query(MarketSummary).one()
    assert saved.summary["headline"] == "Kept"


def test_a_second_worker_does_not_run_while_the_lock_is_held(monkeypatch):
    _bind(monkeypatch)
    started, finish = Event(), Event()
    calls = []

    def perform():
        calls.append(1)
        started.set()
        assert finish.wait(5)
        return "2026-11-11", "2026-11-11"

    first = Thread(target=refresh_service.execute_refresh, args=("manual", perform))
    first.start()
    assert started.wait(5)
    refresh_service.execute_refresh("manual", perform)
    assert calls == [1]
    finish.set()
    first.join(5)
    assert not first.is_alive()


def test_successful_refresh_publishes_a_new_snapshot(monkeypatch):
    sessions = _bind(monkeypatch)
    _save_summary(sessions, "Old")
    monkeypatch.setattr(app, "_run_daily_ingestion", lambda: None)
    monkeypatch.setattr(app, "build_market_state", lambda db: {
        "as_of": "2026-11-11",
        "evidence": {"sp500_5d_return": {"timestamp": "2026-11-10T00:00:00", "display_value": 1.2}},
    })
    monkeypatch.setattr(app, "generate_summary_with_guardrail", lambda state: {"headline": "Fresh"})

    refresh_service.execute_refresh("schedule", app._perform_refresh)
    payload = app.summary_daily()

    assert payload["headline"] == "Fresh"
    assert payload["data_as_of"]["latest"] == "2026-11-10"
    assert payload["refresh"]["last_success_at"]
    assert payload["refresh"]["observations_latest"] == "2026-11-10"
    assert payload["refresh"]["last_error"] is None


def test_refresh_is_due_only_after_the_interval():
    assert refresh_is_due(None, "2026-11-11T18:00:00Z", 60) is True
    assert refresh_is_due("2026-11-11T18:00:00Z", "2026-11-11T18:00:30Z", 60) is False
    assert refresh_is_due("2026-11-11T18:00:00Z", "2026-11-11T18:01:00Z", 60) is True

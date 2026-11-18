"""Background refresh jobs shared by every server process."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from db import SessionLocal
from models import RefreshJob, RefreshLock

logger = logging.getLogger(__name__)

LOCK_NAME = "daily_refresh"
LOCK_TTL_SECONDS = 20 * 60
_workers: list[Thread] = []


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _plus(value: str, seconds: int) -> str:
    return (_parse(value) + timedelta(seconds=seconds)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_refresh_tables() -> None:
    """Create the refresh tables on the active database if they are missing."""

    session = SessionLocal()
    try:
        bind = session.get_bind()
    finally:
        session.close()
    RefreshLock.__table__.create(bind=bind, checkfirst=True)
    RefreshJob.__table__.create(bind=bind, checkfirst=True)


def refresh_is_due(last_success_at: str | None, now: str, interval_seconds: int) -> bool:
    """Return whether a scheduled refresh should run."""

    if not last_success_at:
        return True
    elapsed = (_parse(now) - _parse(last_success_at)).total_seconds()
    return elapsed >= interval_seconds


def _mark_stale_jobs(db, now: str) -> None:
    cutoff = _plus(now, -LOCK_TTL_SECONDS)
    db.query(RefreshJob).filter(
        RefreshJob.status == "running",
        RefreshJob.started_at < cutoff,
    ).update(
        {
            "status": "failed",
            "finished_at": now,
            "error_message": "Refresh stopped before it finished.",
        },
        synchronize_session=False,
    )


def acquire_refresh_lock(owner: str) -> bool:
    """Claim the shared refresh lock, or return False when another process holds it."""

    ensure_refresh_tables()
    now = _now()
    db = SessionLocal()
    try:
        row = db.query(RefreshLock).filter(RefreshLock.name == LOCK_NAME).one_or_none()
        if row is not None and row.expires_at > now:
            return False
        if row is not None:
            db.delete(row)
            db.commit()
        db.add(RefreshLock(
            name=LOCK_NAME,
            owner=owner,
            acquired_at=now,
            expires_at=_plus(now, LOCK_TTL_SECONDS),
        ))
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return False
        _mark_stale_jobs(db, now)
        db.commit()
        return True
    finally:
        db.close()


def release_refresh_lock(owner: str) -> None:
    """Release the lock only when this worker still owns it."""

    db = SessionLocal()
    try:
        db.query(RefreshLock).filter(
            RefreshLock.name == LOCK_NAME,
            RefreshLock.owner == owner,
        ).delete()
        db.commit()
    finally:
        db.close()


def refresh_status(db) -> dict:
    """Describe the latest refresh without starting one."""

    try:
        ensure_refresh_tables()
        now = _now()
        lock = db.query(RefreshLock).filter(RefreshLock.name == LOCK_NAME).one_or_none()
        success = (
            db.query(RefreshJob)
            .filter(RefreshJob.status == "succeeded")
            .order_by(RefreshJob.finished_at.desc())
            .first()
        )
        failure = (
            db.query(RefreshJob)
            .filter(RefreshJob.status == "failed")
            .order_by(RefreshJob.finished_at.desc())
            .first()
        )
    except Exception:
        db.rollback()
        return {
            "in_progress": False,
            "last_success_at": None,
            "last_failure_at": None,
            "last_error": None,
            "observations_latest": None,
        }

    last_error = None
    last_failure_at = None
    if failure and (success is None or (failure.finished_at or "") > (success.finished_at or "")):
        last_error = failure.error_message
        last_failure_at = failure.finished_at
    return {
        "in_progress": bool(lock and lock.expires_at > now),
        "last_success_at": success.finished_at if success else None,
        "last_failure_at": last_failure_at,
        "last_error": last_error,
        "observations_latest": success.observations_latest if success else None,
    }


def _start_job(trigger: str, owner: str) -> int:
    db = SessionLocal()
    try:
        job = RefreshJob(status="running", trigger=trigger, owner=owner, started_at=_now())
        db.add(job)
        db.commit()
        return job.id
    finally:
        db.close()


def _finish_job(job_id: int, status: str, **fields) -> None:
    db = SessionLocal()
    try:
        job = db.query(RefreshJob).filter(RefreshJob.id == job_id).one()
        job.status = status
        job.finished_at = _now()
        for name, value in fields.items():
            setattr(job, name, value)
        db.commit()
    finally:
        db.close()


def execute_refresh(trigger: str, perform) -> None:
    """Run one refresh if this process can claim the shared lock."""

    owner = f"{os.getpid()}-{uuid4().hex}"
    if not acquire_refresh_lock(owner):
        return
    job_id = None
    try:
        job_id = _start_job(trigger, owner)
        summary_date, observations_latest = perform()
        _finish_job(
            job_id,
            "succeeded",
            summary_date=summary_date,
            observations_latest=observations_latest,
            error_message=None,
        )
    except Exception as exc:
        logger.exception("Refresh failed")
        if job_id is not None:
            _finish_job(job_id, "failed", error_message=str(exc)[:500])
    finally:
        release_refresh_lock(owner)


def start_refresh(trigger: str, perform) -> bool:
    """Start one background refresh. If one is already running, leave it and do nothing."""

    if refresh_in_progress():
        return True
    worker = Thread(
        target=execute_refresh,
        args=(trigger, perform),
        name="refresh-job",
        daemon=True,
    )
    worker.start()
    _workers.append(worker)
    return True


def refresh_in_progress() -> bool:
    """Return whether a refresh lock is currently held."""

    db = SessionLocal()
    try:
        return refresh_status(db)["in_progress"]
    finally:
        db.close()


def stop_process_refresh() -> None:
    """Release this process's lock when the server shuts down so the next process can run."""

    prefix = f"{os.getpid()}-"
    db = SessionLocal()
    try:
        now = _now()
        db.query(RefreshJob).filter(
            RefreshJob.status == "running",
            RefreshJob.owner.startswith(prefix),
        ).update(
            {
                "status": "failed",
                "finished_at": now,
                "error_message": "Refresh stopped before it finished.",
            },
            synchronize_session=False,
        )
        db.query(RefreshLock).filter(RefreshLock.owner.startswith(prefix)).delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def scheduler_loop(stop: Event, perform) -> None:
    """Refresh on an interval without blocking requests."""

    delay = float(os.getenv("REFRESH_STARTUP_DELAY_SECONDS", "30"))
    interval = int(os.getenv("REFRESH_INTERVAL_SECONDS", str(6 * 60 * 60)))
    if stop.wait(delay):
        return
    while not stop.is_set():
        try:
            db = SessionLocal()
            try:
                status = refresh_status(db)
            finally:
                db.close()
            if refresh_is_due(status.get("last_success_at"), _now(), interval):
                start_refresh("schedule", perform)
        except Exception:
            logger.exception("Scheduled refresh could not start")
        if stop.wait(interval):
            return

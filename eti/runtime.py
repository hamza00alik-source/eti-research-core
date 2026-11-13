"""Optional single-instance Railway runtime: colocated worker and off-site backups."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import threading

from .worker import AgentWorker
from .backup import backup_once, validate_backup_configuration
from .storage import MemoryStore


logger = logging.getLogger("eti.runtime")


def _worker_loop(stop: threading.Event, poll_seconds: float) -> None:
    worker = AgentWorker()
    while not stop.is_set():
        try:
            result = worker.run_once()
            if result is None:
                stop.wait(poll_seconds)
            else:
                logger.info("agent task processed: id=%s status=%s", result["id"], result["status"])
        except Exception:
            logger.exception("agent worker iteration failed; polling will resume")
            stop.wait(poll_seconds)


def _backup_loop(stop: threading.Event, interval: int) -> None:
    while not stop.is_set():
        try:
            object_key = backup_once()
            logger.info("encrypted database backup uploaded: %s", object_key)
        except Exception:
            logger.exception("encrypted database backup failed")
        stop.wait(interval)


@asynccontextmanager
async def runtime_lifespan(_app):
    threads: list[threading.Thread] = []
    stops: list[threading.Event] = []
    worker_enabled = os.environ.get("ETI_EMBEDDED_WORKER", "0").lower() in {"1", "true", "yes"}
    backup_enabled = os.environ.get("ETI_BACKUP_ENABLED", "0").lower() in {"1", "true", "yes"}
    if worker_enabled or backup_enabled:
        MemoryStore()  # initialize the shared database before either background loop starts
    if backup_enabled:
        validate_backup_configuration()

    if worker_enabled:
        poll_seconds = float(os.environ.get("ETI_WORKER_POLL_SECONDS", "2"))
        if not 0.1 <= poll_seconds <= 60:
            raise ValueError("ETI_WORKER_POLL_SECONDS must be between 0.1 and 60")
        stop = threading.Event()
        thread = threading.Thread(
            target=_worker_loop, args=(stop, poll_seconds), name="eti-worker", daemon=True
        )
        stops.append(stop)
        threads.append(thread)
        thread.start()
        logger.info("embedded ETI worker started")

    if backup_enabled:
        interval = int(os.environ.get("ETI_BACKUP_INTERVAL_SECONDS", "86400"))
        if interval < 3600:
            raise ValueError("ETI_BACKUP_INTERVAL_SECONDS must be at least 3600")
        stop = threading.Event()
        thread = threading.Thread(
            target=_backup_loop, args=(stop, interval), name="eti-backup", daemon=True
        )
        stops.append(stop)
        threads.append(thread)
        thread.start()
        logger.info("ETI off-site backup scheduler started")

    try:
        yield
    finally:
        for stop in stops:
            stop.set()
        for thread in threads:
            thread.join(timeout=10)

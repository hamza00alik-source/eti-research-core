"""Optional single-instance Railway runtime: colocated worker and off-site backups."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import threading
import time

from .worker import AgentWorker
from .backup import backup_once
from .storage import MemoryStore


logger = logging.getLogger("eti.runtime")


def _worker_loop(stop: threading.Event) -> None:
    worker = AgentWorker()
    poll_seconds = float(os.environ.get("ETI_WORKER_POLL_SECONDS", "2"))
    if not 0.1 <= poll_seconds <= 60:
        raise ValueError("ETI_WORKER_POLL_SECONDS must be between 0.1 and 60")
    while not stop.is_set():
        result = worker.run_once()
        if result is None:
            stop.wait(poll_seconds)
        else:
            logger.info("agent task processed: id=%s status=%s", result["id"], result["status"])


def _backup_loop(stop: threading.Event) -> None:
    interval = int(os.environ.get("ETI_BACKUP_INTERVAL_SECONDS", "86400"))
    if interval < 3600:
        raise ValueError("ETI_BACKUP_INTERVAL_SECONDS must be at least 3600")
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

    if os.environ.get("ETI_EMBEDDED_WORKER", "0").lower() in {"1", "true", "yes"}:
        stop = threading.Event()
        thread = threading.Thread(target=_worker_loop, args=(stop,), name="eti-worker", daemon=True)
        stops.append(stop)
        threads.append(thread)
        thread.start()
        logger.info("embedded ETI worker started")

    if os.environ.get("ETI_BACKUP_ENABLED", "0").lower() in {"1", "true", "yes"}:
        stop = threading.Event()
        thread = threading.Thread(target=_backup_loop, args=(stop,), name="eti-backup", daemon=True)
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

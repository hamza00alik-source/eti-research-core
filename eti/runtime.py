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


def _worker_loop(stop: threading.Event, poll_seconds: float, ready: threading.Event | None = None) -> None:
    worker = AgentWorker()
    while not stop.is_set():
        try:
            result = worker.run_once()
            if ready is not None and not ready.is_set():
                ready.set()
                print("ETI_WORKER_HEARTBEAT_READY", flush=True)
            if result is None:
                stop.wait(poll_seconds)
            else:
                logger.info("agent task processed: id=%s status=%s", result["id"], result["status"])
        except Exception:
            logger.exception("agent worker iteration failed; polling will resume")
            stop.wait(poll_seconds)


def _backup_loop(stop: threading.Event, interval: int, worker_ready: threading.Event | None = None) -> None:
    if worker_ready is not None and not worker_ready.wait(timeout=15):
        print("ETI_BACKUP_STARTING_WITHOUT_WORKER_HEARTBEAT", flush=True)
    while not stop.is_set():
        try:
            object_key = backup_once()
            print(f"ETI_BACKUP_UPLOADED key={object_key}", flush=True)
        except Exception:
            logger.exception("encrypted database backup failed")
            print("ETI_BACKUP_FAILED; see preceding traceback", flush=True)
        stop.wait(interval)


@asynccontextmanager
async def runtime_lifespan(_app):
    threads: list[threading.Thread] = []
    stops: list[threading.Event] = []
    worker_ready = threading.Event()
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
            target=_worker_loop, args=(stop, poll_seconds, worker_ready), name="eti-worker", daemon=True
        )
        stops.append(stop)
        threads.append(thread)
        thread.start()
        print("ETI_WORKER_STARTED", flush=True)

    if backup_enabled:
        interval = int(os.environ.get("ETI_BACKUP_INTERVAL_SECONDS", "86400"))
        if interval < 3600:
            raise ValueError("ETI_BACKUP_INTERVAL_SECONDS must be at least 3600")
        stop = threading.Event()
        thread = threading.Thread(
            target=_backup_loop, args=(stop, interval, worker_ready if worker_enabled else None), name="eti-backup", daemon=True
        )
        stops.append(stop)
        threads.append(thread)
        thread.start()
        print("ETI_BACKUP_SCHEDULER_STARTED", flush=True)

    try:
        yield
    finally:
        for stop in stops:
            stop.set()
        for thread in threads:
            thread.join(timeout=10)

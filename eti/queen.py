"""ETI Queen: bounded, demand-driven supervisor for the durable worker colony.

The Queen scales execution slots, not model permissions. Every slot runs the
existing allowlisted AgentWorker against the durable lease-based task queue.
It never creates capabilities, changes financial budgets, approves models, or
enables order execution.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .colony_intelligence import build_colony_intelligence
from .intelligence import build_intelligence_manifest, department_for_capability
from .storage import MemoryStore
from .worker import AgentWorker


@dataclass(frozen=True)
class ColonyConfig:
    """Hard operational limits for one Queen-managed colony."""

    max_workers: int = 2
    backlog_per_worker: int = 2
    poll_seconds: float = 1.0
    lease_seconds: int = 120

    def __post_init__(self) -> None:
        if isinstance(self.max_workers, bool) or not isinstance(self.max_workers, int) or not 1 <= self.max_workers <= 8:
            raise ValueError("max_workers must be an integer from 1 to 8")
        if isinstance(self.backlog_per_worker, bool) or not isinstance(self.backlog_per_worker, int) or not 1 <= self.backlog_per_worker <= 10:
            raise ValueError("backlog_per_worker must be an integer from 1 to 10")
        if isinstance(self.poll_seconds, bool) or not isinstance(self.poll_seconds, (int, float)) or not 0.1 <= self.poll_seconds <= 60:
            raise ValueError("poll_seconds must be between 0.1 and 60")
        if isinstance(self.lease_seconds, bool) or not isinstance(self.lease_seconds, int) or not 5 <= self.lease_seconds <= 3600:
            raise ValueError("lease_seconds must be an integer from 5 to 3600")


@dataclass
class _WorkerSlot:
    slot_id: int
    stop: threading.Event
    thread: threading.Thread


def calculate_worker_target(
    queued_tasks: int, *, max_workers: int, backlog_per_worker: int
) -> int:
    """Return bounded concurrency from queue demand; zero means no idle workers."""
    for value, name, minimum in (
        (queued_tasks, "queued_tasks", 0),
        (max_workers, "max_workers", 1),
        (backlog_per_worker, "backlog_per_worker", 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if max_workers > 8 or backlog_per_worker > 10:
        raise ValueError("worker scaling exceeds hard colony limits")
    if queued_tasks == 0:
        return 0
    return min(max_workers, math.ceil(queued_tasks / backlog_per_worker))


class QueenOrchestrator:
    """Scale durable worker slots for one authenticated owner/workspace.

    A slot is a concurrent AgentWorker instance. Task capabilities remain
    server-implemented and allowlisted by AgentCapabilityRegistry. Workers
    claim tasks using the existing SQLite lease mechanism, so multiple Queen
    processes can safely compete for work in a single-instance deployment.
    """

    def __init__(
        self,
        owner_id: str,
        workspace_id: str,
        *,
        store: MemoryStore | None = None,
        config: ColonyConfig | None = None,
        worker_factory: Callable[[], AgentWorker] | None = None,
    ) -> None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id is required")
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id is required")
        self.owner_id = owner_id.strip()
        self.workspace_id = workspace_id.strip()
        self.store = store or MemoryStore()
        self.config = config or ColonyConfig()
        self.worker_factory = worker_factory or (
            lambda: AgentWorker(
                self.store,
                lease_seconds=self.config.lease_seconds,
                owner_id=self.owner_id,
                workspace_id=self.workspace_id,
            )
        )
        self._slots: dict[int, _WorkerSlot] = {}
        self._next_slot_id = 1
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._last_intelligence_emit = 0.0

    @staticmethod
    def _emit(event: str, **fields: object) -> None:
        print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)

    @staticmethod
    def _task_department(capability: object) -> str:
        if not isinstance(capability, str):
            return "unmapped"
        try:
            return department_for_capability(capability)
        except LookupError:
            return "unmapped"

    def intelligence_manifest(self) -> dict:
        """Return a read-only inventory of department and specialist readiness."""
        return build_intelligence_manifest(self.store.list_model_versions())

    def _schedule_due_work(self) -> int:
        """Use existing owner-authorized schedules; never invent new permissions."""
        return (
            self.store.enqueue_due_autonomy_tasks()
            + self.store.enqueue_due_paper_tasks()
            + self.store.enqueue_due_live_execution_tasks()
        )

    def _queued_count(self) -> int:
        return len(self.store.list_agent_tasks(
            self.owner_id, self.workspace_id, status="queued", limit=500
        ))

    def _worker_loop(self, slot_id: int, stop: threading.Event) -> None:
        worker = self.worker_factory()
        self._emit("queen.worker.started", slot_id=slot_id)
        while not stop.is_set() and not self._shutdown.is_set():
            try:
                result = worker.run_once()
                if result is None:
                    stop.wait(self.config.poll_seconds)
                else:
                    self._emit(
                        "queen.worker.task_finished",
                        slot_id=slot_id,
                        task_id=result.get("id"),
                        capability=result.get("target_capability"),
                        department=self._task_department(result.get("target_capability")),
                        status=result.get("status"),
                    )
            except Exception as exc:
                # Do not log task payloads, credentials, or exception messages.
                self._emit(
                    "queen.worker.loop_error",
                    slot_id=slot_id,
                    error_type=type(exc).__name__,
                )
                stop.wait(self.config.poll_seconds)
        self._emit("queen.worker.stopped", slot_id=slot_id)

    def reconcile(self, queued_tasks: int | None = None) -> dict[str, int]:
        """Grow or shrink worker slots to match current queue demand."""
        queued = self._queued_count() if queued_tasks is None else queued_tasks
        target = calculate_worker_target(
            queued,
            max_workers=self.config.max_workers,
            backlog_per_worker=self.config.backlog_per_worker,
        )
        with self._lock:
            # Drop completed threads before calculating the current colony size.
            for slot_id, slot in list(self._slots.items()):
                if not slot.thread.is_alive():
                    self._slots.pop(slot_id, None)
            active = len(self._slots)
            if active < target:
                for _ in range(target - active):
                    slot_id = self._next_slot_id
                    self._next_slot_id += 1
                    stop = threading.Event()
                    thread = threading.Thread(
                        target=self._worker_loop,
                        args=(slot_id, stop),
                        name=f"eti-queen-worker-{slot_id}",
                        daemon=True,
                    )
                    self._slots[slot_id] = _WorkerSlot(slot_id, stop, thread)
                    thread.start()
            elif active > target:
                # Retiring workers finish their current leased task before exiting.
                for slot_id in sorted(self._slots, reverse=True)[: active - target]:
                    self._slots[slot_id].stop.set()
            active_now = sum(1 for slot in self._slots.values() if slot.thread.is_alive())
        self._emit(
            "queen.colony.reconciled",
            queued_tasks=queued,
            target_workers=target,
            active_workers=active_now,
            max_workers=self.config.max_workers,
        )
        # Operational history informs observability only. It cannot alter
        # permissions, budgets, model promotion, or execution authority.
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_intelligence_emit >= 60.0:
            task_history = self.store.list_agent_tasks(
                self.owner_id, self.workspace_id, limit=500
            )
            intelligence = build_colony_intelligence(task_history)
            self._emit(
                "queen.colony.intelligence",
                observed_tasks=intelligence["observed_tasks"],
                history_may_be_truncated=intelligence["history_may_be_truncated"],
                capabilities=intelligence["capabilities"],
                authority="observational_only",
            )
            self._last_intelligence_emit = now_monotonic
        return {
            "queued_tasks": queued,
            "target_workers": target,
            "active_workers": active_now,
            "max_workers": self.config.max_workers,
        }

    def run_forever(self) -> None:
        """Schedule approved work and elastically scale the colony until stopped."""
        manifest = self.intelligence_manifest()
        self._emit(
            "queen.intelligence.architecture",
            schema=manifest["schema"],
            departments={key: value["status"] for key, value in manifest["departments"].items()},
            registered_model_versions=manifest["registered_model_versions"],
            unassigned_legacy_model_versions=manifest["unassigned_legacy_model_versions"],
            authority="observational_only",
        )
        self._emit(
            "queen.started",
            owner_id=self.owner_id,
            workspace_id=self.workspace_id,
            max_workers=self.config.max_workers,
            backlog_per_worker=self.config.backlog_per_worker,
            execution_authority="existing_task_permissions_only",
        )
        try:
            while not self._shutdown.is_set():
                try:
                    scheduled = self._schedule_due_work()
                    queued = self._queued_count()
                    status = self.reconcile(queued)
                    if scheduled:
                        self._emit("queen.work_scheduled", scheduled_tasks=scheduled)
                    if queued == 0 and status["active_workers"] == 0:
                        self._shutdown.wait(self.config.poll_seconds)
                    else:
                        self._shutdown.wait(min(self.config.poll_seconds, 1.0))
                except Exception as exc:
                    self._emit("queen.supervisor_error", error_type=type(exc).__name__)
                    self._shutdown.wait(self.config.poll_seconds)
        finally:
            self.stop()

    def stop(self, join_timeout: float = 5.0) -> None:
        """Stop accepting work and let in-flight workers finish naturally."""
        self._shutdown.set()
        with self._lock:
            slots = list(self._slots.values())
            for slot in slots:
                slot.stop.set()
        for slot in slots:
            slot.thread.join(timeout=join_timeout)
        self._emit("queen.stopped", worker_slots=len(slots))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ETI's bounded Queen worker supervisor")
    parser.add_argument("--owner-id", default=os.environ.get("ETI_QUEEN_OWNER_ID"), required=False)
    parser.add_argument("--workspace-id", default=os.environ.get("ETI_QUEEN_WORKSPACE_ID"), required=False)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--backlog-per-worker", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--lease-seconds", type=int, default=120)
    args = parser.parse_args()
    if not args.owner_id or not args.workspace_id:
        parser.error("provide --owner-id and --workspace-id, or set ETI_QUEEN_OWNER_ID and ETI_QUEEN_WORKSPACE_ID")
    queen = QueenOrchestrator(
        args.owner_id,
        args.workspace_id,
        config=ColonyConfig(
            max_workers=args.max_workers,
            backlog_per_worker=args.backlog_per_worker,
            poll_seconds=args.poll_seconds,
            lease_seconds=args.lease_seconds,
        ),
    )

    def request_stop(_signum, _frame):
        queen.stop()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    queen.run_forever()


if __name__ == "__main__":
    main()

"""Separate ETI durable agent worker process.

Run explicitly with python -m eti.worker --once or
python -m eti.worker --poll-seconds 2. The FastAPI app never starts workers
implicitly; operators control worker processes separately.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any
from uuid import uuid4

from .agents import AgentCapabilityRegistry, default_capabilities
from .storage import MemoryStore


class AgentWorker:
    def __init__(
        self,
        store: MemoryStore | None = None,
        registry: AgentCapabilityRegistry | None = None,
        lease_seconds: int = 120,
    ):
        self.store = store or MemoryStore()
        self.registry = registry or default_capabilities()
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be an integer from 5 to 3600")
        self.lease_seconds = lease_seconds
        self.worker_id = str(uuid4())

    def run_once(self) -> dict[str, Any] | None:
        # The supervisor autonomously seeds owner-enabled research schedules.
        # This only creates public-data research tasks; order execution is not registered.
        scheduled = self.store.enqueue_due_autonomy_tasks()
        if scheduled:
            print(json.dumps({"event": "autonomy.research_scheduled", "tasks": scheduled},
                             separators=(",", ":")), flush=True)
        self.store.record_agent_worker_heartbeat(self.worker_id, self.registry.capabilities())
        task = self.store.claim_next_agent_task(lease_seconds=self.lease_seconds)
        if task is None:
            return None
        lease_token = task["lease_token"]
        try:
            result = self.registry.execute(task["target_capability"], task["input"])
            status, retryable = "completed", False
        except LookupError as exc:
            result = {"error": {"type": type(exc).__name__, "message": "unsupported agent capability"}}
            status, retryable = "failed", False
        except ValueError as exc:
            result = {"error": {"type": type(exc).__name__, "message": str(exc)[:300]}}
            status, retryable = "failed", False
        except Exception as exc:
            result = {"error": {"type": type(exc).__name__, "message": "Temporary capability failure"}}
            status, retryable = "failed", True

        try:
            if status == "completed":
                updated = self.store.complete_claimed_agent_task(
                    task["id"], lease_token, result
                )
                message_type = "task.completed"
            else:
                updated = self.store.fail_claimed_agent_task(
                    task["id"], lease_token, result, retryable=retryable
                )
                message_type = "task.requeued" if updated["status"] == "queued" else "task.failed"
        except ValueError:
            # A user may cancel a running task, or the lease may expire while a
            # capability is running. A stale worker must never overwrite it.
            return self.store.get_agent_task(
                task["owner_id"], task["workspace_id"], task["id"]
            )

        self.store.send_agent_message(
            task["owner_id"], task["workspace_id"], "eti.worker",
            task["requester"], message_type,
            {"task_id": task["id"], "status": updated["status"], "result": result},
            task["id"],
        )
        return updated

    def run_forever(self, poll_seconds: float = 2.0) -> None:
        if not 0.1 <= poll_seconds <= 60:
            raise ValueError("poll_seconds must be between 0.1 and 60")
        while True:
            result = self.run_once()
            if result is None:
                time.sleep(poll_seconds)
            else:
                print(json.dumps({
                    "task_id": result["id"],
                    "status": result["status"],
                    "capability": result["target_capability"],
                }, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ETI durable agent worker")
    parser.add_argument("--once", action="store_true", help="claim and process at most one task")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--lease-seconds", type=int, default=120)
    args = parser.parse_args()
    worker = AgentWorker(lease_seconds=args.lease_seconds)
    if args.once:
        result = worker.run_once()
        print(json.dumps(result, default=str, separators=(",", ":")))
    else:
        worker.run_forever(args.poll_seconds)


if __name__ == "__main__":
    main()

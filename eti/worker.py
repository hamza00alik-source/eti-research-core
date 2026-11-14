"""Separate ETI durable agent worker process.

Run explicitly with python -m eti.worker --once or
python -m eti.worker --poll-seconds 2. The FastAPI app never starts workers
implicitly; operators control worker processes separately.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from typing import Any
from uuid import uuid4

from .agents import AgentCapabilityRegistry, default_capabilities
from .artifacts import ModelArtifactStore
from .paper import run_paper_tick
from .execution import run_live_tick
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
        if "paper.tick" not in self.registry.capabilities():
            self.registry.register("paper.tick", lambda payload: run_paper_tick(payload, self.store))
        if "live.tick" not in self.registry.capabilities():
            self.registry.register("live.tick", lambda payload: run_live_tick(payload, self.store))
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be an integer from 5 to 3600")
        self.lease_seconds = lease_seconds
        self.worker_id = str(uuid4())

    @staticmethod
    def _log_event(event: str, **fields: Any) -> None:
        """Emit structured, payload-free worker lifecycle telemetry."""
        print(json.dumps({"event": event, **fields}, separators=(",", ":")), flush=True)

    def _start_lease_heartbeat(
        self, task_id: str, lease_token: str
    ) -> tuple[threading.Event, threading.Event, threading.Thread]:
        """Keep a claimed task leased while a long capability is executing.

        A failed renewal is fail-closed: the worker must not persist a result
        after it can no longer prove that it owns the task lease.
        """
        stop = threading.Event()
        lease_lost = threading.Event()
        interval = max(1.0, self.lease_seconds / 3.0)

        def renew_until_stopped() -> None:
            while not stop.wait(interval):
                try:
                    self.store.renew_agent_task_lease(
                        task_id, lease_token, lease_seconds=self.lease_seconds
                    )
                except Exception as exc:
                    lease_lost.set()
                    print(json.dumps({
                        "event": "agent.task.lease_renewal_failed",
                        "task_id": task_id,
                        "error_type": type(exc).__name__,
                    }, separators=(",", ":")), flush=True)
                    return

        thread = threading.Thread(
            target=renew_until_stopped,
            name=f"eti-lease-{task_id[:8]}",
            daemon=True,
        )
        thread.start()
        return stop, lease_lost, thread

    def _enqueue_strategy_followup(self, task: dict[str, Any]) -> None:
        task_input = task.get("input")
        if (task.get("target_capability") != "market.snapshot"
                or task.get("requester") != "eti.autonomy.supervisor"
                or not isinstance(task_input, dict)
                or task_input.get("scheduled_by") != "eti.autonomy.supervisor"):
            return
        payload = {
            "venue": task_input.get("venue", "mexc"),
            "asset_class": task_input.get("asset_class", "crypto"),
            "market_type": task_input.get("market_type", "spot"),
            "symbol": task_input.get("symbol"),
            "timeframe": task_input.get("timeframe", "1h"), "limit": 500,
            "min_history": 120, "validation_size": 80, "test_size": 80,
            "step_size": 80, "purge_bars": 2, "fee_bps": 6.0, "slippage_bps": 2.0,
            "scheduled_by": "eti.autonomy.supervisor", "source_snapshot_task_id": task["id"],
        }
        followup = self.store.create_agent_task(
            task["owner_id"], task["workspace_id"], "eti.autonomy.supervisor",
            "strategy.walk_forward", payload, max_attempts=2,
        )
        print(json.dumps({"event": "autonomy.strategy_evaluation_queued",
                          "task_id": followup["id"], "source_snapshot_task_id": task["id"],
                          "symbol": payload["symbol"], "timeframe": payload["timeframe"]},
                         separators=(",", ":")), flush=True)

    def _enqueue_discovered_model_training(
        self, task: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Queue one top-ranked instrument for training after scheduled discovery."""
        task_input = task.get("input")
        if (task.get("target_capability") != "market.discover"
                or task.get("requester") != "eti.autonomy.model_training"
                or not isinstance(task_input, dict)
                or task_input.get("scheduled_by") != "eti.autonomy.model_training"):
            return
        instruments = result.get("instruments") if isinstance(result, dict) else None
        if not isinstance(instruments, list) or not instruments:
            self._log_event("autonomy.market_discovery.no_eligible_instruments",
                            task_id=task["id"])
            return
        # Provider returns highest public quote-volume first where supported.
        selected = instruments[0]
        symbol = selected.get("symbol") if isinstance(selected, dict) else None
        venue = selected.get("venue") if isinstance(selected, dict) else None
        asset_class = selected.get("asset_class") if isinstance(selected, dict) else None
        market_type = selected.get("market_type") if isinstance(selected, dict) else None
        if not all(isinstance(value, str) and value for value in (symbol, venue, asset_class, market_type)):
            raise ValueError("market discovery returned an incomplete instrument record")
        existing = self.store.list_agent_tasks(
            task["owner_id"], task["workspace_id"], limit=500
        )
        if any(
            item.get("input", {}).get("source_discovery_task_id") == task["id"]
            for item in existing
        ):
            return
        payload = {
            "venue": venue, "asset_class": asset_class, "market_type": market_type,
            "symbol": symbol, "timeframe": task_input.get("timeframe", "1h"),
            "limit": 3000, "horizon": 6,
            "scheduled_by": "eti.autonomy.model_training",
            "source_discovery_task_id": task["id"],
        }
        queued = self.store.create_agent_task(
            task["owner_id"], task["workspace_id"], "eti.autonomy.model_training",
            "model.train_research", payload, max_attempts=2,
        )
        self._log_event(
            "autonomy.discovered_model_training_queued",
            discovery_task_id=task["id"], training_task_id=queued["id"],
            venue=venue, asset_class=asset_class, market_type=market_type, symbol=symbol,
        )

    def _register_model_candidate(self, result: dict[str, Any]) -> dict[str, Any]:
        """Register candidates and promote only after independent quality and artifact verification."""
        artifact = result.get("candidate_artifact") or {}
        schema = result.get("input_schema") or {}
        digest = artifact.get("artifact_sha256")
        uri = artifact.get("artifact_uri")
        required = ("venue", "asset_class", "market_type", "symbol", "timeframe", "training_data_ref")
        if not digest or not uri or not isinstance(schema, dict) or any(not result.get(k) for k in required):
            raise ValueError("trained candidate is missing registry metadata")
        safe_symbol = result["symbol"].replace("/", "-").replace(":", "-").replace("=", "-").replace("^", "")
        asset_class = result["asset_class"]
        market_type = result["market_type"]
        model_id = f"eti-{result['venue']}-{asset_class}-{market_type}-{safe_symbol}-{result['timeframe']}".lower()
        version = "research-" + digest[:16]
        evaluation = dict(result.get("evaluation") or {})
        evaluation["passed"] = bool(result.get("evaluation_passed", False))
        evaluation["quality_checks"] = result.get("quality_checks", {})
        evaluation["candidate_only"] = True
        calibration_status = (
            "calibrated" if result.get("calibration_status") == "calibrated" else "failed"
        )
        try:
            registered = self.store.register_model_version(
                model_id, version, uri, digest, schema, [f"{asset_class}_{market_type}"],
                result["training_data_ref"], evaluation, calibration_status, "eti.worker",
            )
        except sqlite3.IntegrityError:
            registered = self.store.get_model_version(model_id, version)
            if not registered or registered.get("artifact_sha256") != digest:
                raise
        auto_approved = False
        if evaluation["passed"] and calibration_status == "calibrated" and registered["lifecycle"] == "candidate":
            # The promotion authority is a deterministic server quality gate, not the model itself.
            artifact_model = ModelArtifactStore().load_model(uri, digest)
            trained_features = getattr(getattr(artifact_model, "estimator", None), "feature_names_in_", None)
            expected_features = schema.get("features")
            if (not isinstance(expected_features, list) or trained_features is None
                    or [str(name) for name in trained_features] != expected_features
                    or not artifact_model.has_valid_calibration()):
                raise ValueError("candidate artifact failed independent feature-schema or calibration verification")
            registered = self.store.approve_model_version(model_id, version, "eti.quality_gate")
            auto_approved = True
        result["registry"] = {
            "model_id": registered["model_id"],
            "version": registered["version"],
            "lifecycle": registered["lifecycle"],
            "evaluation_passed": evaluation["passed"],
            "approval_required": registered["lifecycle"] != "approved",
            "promotion_authority": "eti.quality_gate" if auto_approved else None,
        }
        result["lifecycle"] = registered["lifecycle"]
        return result

    def run_once(self) -> dict[str, Any] | None:
        # Seed public research, simulated paper and owner-authorized live spot tasks.
        # Live orders are reachable only through the supervisor-gated live.tick capability.
        scheduled = self.store.enqueue_due_autonomy_tasks()
        paper_scheduled = self.store.enqueue_due_paper_tasks()
        live_scheduled = self.store.enqueue_due_live_execution_tasks()
        solana_refresh_scheduled = self.store.enqueue_due_solana_research_refreshes()
        if scheduled or paper_scheduled or live_scheduled or solana_refresh_scheduled:
            print(json.dumps({"event": "autonomy.tasks_scheduled",
                              "research_tasks": scheduled, "paper_tasks": paper_scheduled,
                              "live_tasks": live_scheduled,
                              "solana_refresh_tasks": solana_refresh_scheduled},
                             separators=(",", ":")), flush=True)
        self.store.record_agent_worker_heartbeat(self.worker_id, self.registry.capabilities())
        task = self.store.claim_next_agent_task(lease_seconds=self.lease_seconds)
        if task is None:
            return None
        lease_token = task["lease_token"]
        capability = task["target_capability"]
        attempt = task.get("attempt_count")
        self._log_event(
            "agent.task.claimed", task_id=task["id"], capability=capability,
            attempt_count=attempt,
        )
        started_at = time.monotonic()
        self._log_event(
            "agent.task.execution_started", task_id=task["id"], capability=capability,
            attempt_count=attempt,
        )
        stop_heartbeat, lease_lost, heartbeat_thread = self._start_lease_heartbeat(
            task["id"], lease_token
        )
        error_type = None
        try:
            task_input = task["input"]
            if capability == "live.tick":
                if (task.get("requester") != "eti.live.supervisor"
                        or not isinstance(task_input, dict)
                        or task_input.get("scheduled_by") != "eti.live.supervisor"):
                    raise ValueError("live execution tasks must be created by the owner-authorized supervisor")
                task_input = {**task_input, "owner_id": task["owner_id"], "task_id": task["id"]}
            result = self.registry.execute(capability, task_input)
            if capability == "model.train_research":
                result = self._register_model_candidate(result)
            status, retryable = "completed", False
        except LookupError as exc:
            error_type = type(exc).__name__
            result = {"error": {"type": error_type, "message": "unsupported agent capability"}}
            status, retryable = "failed", False
        except ValueError as exc:
            error_type = type(exc).__name__
            result = {"error": {"type": error_type, "message": str(exc)[:300]}}
            status, retryable = "failed", False
        except Exception as exc:
            error_type = type(exc).__name__
            result = {"error": {"type": error_type, "message": "Temporary capability failure"}}
            status, retryable = "failed", True
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=max(2.0, self.lease_seconds / 3.0 + 1.0))

        duration_ms = round((time.monotonic() - started_at) * 1000, 2)
        if lease_lost.is_set():
            # Another worker may now own or be processing this task. Never let
            # this worker overwrite that newer state with a stale result.
            self._log_event(
                "agent.task.result_discarded_after_lease_loss",
                task_id=task["id"], capability=capability, duration_ms=duration_ms,
            )
            return self.store.get_agent_task(
                task["owner_id"], task["workspace_id"], task["id"]
            )

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
        except ValueError as exc:
            # A user may cancel a running task, or the lease may expire while a
            # capability is running. A stale worker must never overwrite it.
            self._log_event(
                "agent.task.completion_rejected", task_id=task["id"],
                capability=capability, duration_ms=duration_ms,
                error_type=type(exc).__name__,
            )
            return self.store.get_agent_task(
                task["owner_id"], task["workspace_id"], task["id"]
            )

        self._log_event(
            "agent.task.execution_finished", task_id=task["id"], capability=capability,
            status=updated["status"], duration_ms=duration_ms,
            error_type=error_type,
        )
        if status == "completed" and capability in {"solana.market.scan", "solana.token.audit"}:
            try:
                label_result = self.store.generate_solana_research_labels(
                    task["owner_id"], task["workspace_id"]
                )
                self._log_event(
                    "solana.labels.generated", task_id=task["id"],
                    generated=label_result["generated"], available=label_result["available"],
                )
            except Exception as exc:
                self._log_event(
                    "solana.labels.generation_failed", task_id=task["id"],
                    error_type=type(exc).__name__,
                )
        self.store.send_agent_message(
            task["owner_id"], task["workspace_id"], "eti.worker",
            task["requester"], message_type,
            {"task_id": task["id"], "status": updated["status"], "result": result},
            task["id"],
        )
        if status == "completed":
            try:
                self._enqueue_strategy_followup(task)
                self._enqueue_discovered_model_training(task, result)
            except Exception:
                print("ETI_STRATEGY_FOLLOWUP_ENQUEUE_FAILED", flush=True)
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
                }, separators=(",", ":")), flush=True)


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

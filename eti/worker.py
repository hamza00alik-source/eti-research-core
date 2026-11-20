"""Separate ETI durable agent worker process.

Run explicitly with python -m eti.worker --once or
python -m eti.worker --poll-seconds 2. The FastAPI app never starts workers
implicitly; operators control worker processes separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import sqlite3
import threading
import time
from typing import Any
from uuid import uuid4

from .agents import AgentCapabilityRegistry, default_capabilities
from .paper import run_paper_tick
from .live_market import MarketDataUnavailable
from .execution import run_live_tick
from .storage import MemoryStore
from .model_identity import model_id_for_instrument


class AgentWorker:
    def __init__(
        self,
        store: MemoryStore | None = None,
        registry: AgentCapabilityRegistry | None = None,
        lease_seconds: int = 120,
        owner_id: str | None = None,
        workspace_id: str | None = None,
    ):
        self.store = store or MemoryStore()
        self.registry = registry or default_capabilities()
        if "paper.tick" not in self.registry.capabilities():
            self.registry.register("paper.tick", self._run_paper_tick_supervised)
        if "live.tick" not in self.registry.capabilities():
            self.registry.register("live.tick", lambda payload: run_live_tick(payload, self.store))
        if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int) or not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be an integer from 5 to 3600")
        if (owner_id is None) != (workspace_id is None):
            raise ValueError("owner_id and workspace_id must be supplied together")
        self.owner_id = owner_id
        self.workspace_id = workspace_id
        self.lease_seconds = lease_seconds
        self.worker_id = str(uuid4())

    def _run_paper_tick_supervised(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one paper decision and persist public-feed health without masking failures."""
        owner_id = payload.get("owner_id")
        session_id = payload.get("session_id")
        try:
            result = run_paper_tick(payload, self.store)
        except MarketDataUnavailable as exc:
            self.store.record_paper_feed_health(
                owner_id, session_id, error=exc
            )
            self._log_event(
                "paper.market_feed.unavailable",
                session_id=session_id,
                error_type=type(exc).__name__,
            )
            raise
        if result.get("processed") is True:
            quote = result.get("quote") or {}
            self.store.record_paper_feed_health(
                owner_id, session_id,
                snapshot={
                    "snapshot_sha256": result.get("market_snapshot_sha256"),
                    "quote": {"timestamp": quote.get("timestamp")},
                },
            )
            self._log_event(
                "paper.market_feed.healthy",
                session_id=session_id,
                snapshot_sha256=result.get("market_snapshot_sha256"),
            )
        return result

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
        asset_class = result["asset_class"]
        market_type = result["market_type"]
        model_id = model_id_for_instrument(
            venue=result["venue"],
            asset_class=asset_class,
            market_type=market_type,
            symbol=result["symbol"],
            timeframe=result["timeframe"],
        )
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
                experiment_id=result.get("experiment_id"),
                lineage=result.get("lineage"),
            )
        except sqlite3.IntegrityError:
            registered = self.store.get_model_version(model_id, version)
            if not registered or registered.get("artifact_sha256") != digest:
                raise
        # Training workers may register research candidates but never approve them.
        # Approval requires the separate, documented human-review promotion process.
        auto_approved = False
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

    def _bind_paper_portfolio_snapshot(
        self, task: dict[str, Any], payload: dict[str, Any],
        *, now: datetime | None = None,
    ) -> dict[str, Any]:
        """Replace caller-supplied exposure with a fresh owner-scoped paper snapshot."""
        if not isinstance(payload, dict):
            raise ValueError("portfolio allocation input must be an object")
        account_id = payload.get("portfolio_account_id")
        if not isinstance(account_id, str) or not account_id.strip():
            raise ValueError("portfolio.allocate requires portfolio_account_id")
        owner_id, workspace_id = task.get("owner_id"), task.get("workspace_id")
        if not isinstance(owner_id, str) or not isinstance(workspace_id, str):
            raise ValueError("portfolio allocation task is missing owner/workspace scope")
        account = self.store.get_paper_portfolio_account(owner_id, workspace_id, account_id)
        if not isinstance(account, dict):
            raise ValueError("paper portfolio account is not owned by this task workspace")
        if account.get("status") != "active":
            raise ValueError("paper portfolio account is not active")
        if account.get("base_currency") != "USD":
            raise ValueError("portfolio allocation currently requires a USD base-currency account; FX conversion is not implemented")
        snapshot = self.store.get_latest_paper_portfolio_snapshot(
            owner_id, workspace_id, account_id
        )
        if not isinstance(snapshot, dict) or snapshot.get("source") != "owner_attested":
            raise ValueError("a saved owner-attested paper portfolio snapshot is required")
        try:
            as_of = datetime.fromisoformat(str(snapshot["as_of"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("paper portfolio snapshot timestamp is invalid") from exc
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("paper portfolio snapshot timestamp must include a timezone")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("worker time must include a timezone")
        age_seconds = (current.astimezone(timezone.utc) - as_of.astimezone(timezone.utc)).total_seconds()
        if age_seconds < -30:
            raise ValueError("paper portfolio snapshot is future-dated")
        if age_seconds > 86400:
            raise ValueError("paper portfolio snapshot is stale; refresh it within 24 hours")

        positions = snapshot.get("positions")
        if not isinstance(positions, list):
            raise ValueError("paper portfolio snapshot positions are invalid")
        current_positions: dict[str, float] = {}
        group_exposure: dict[str, float] = {}
        for position in positions:
            if not isinstance(position, dict):
                raise ValueError("paper portfolio snapshot contains an invalid position")
            symbol, group, notional = (
                position.get("symbol"), position.get("correlation_group"), position.get("notional")
            )
            if (not isinstance(symbol, str) or not symbol
                    or not isinstance(group, str) or not group
                    or isinstance(notional, bool) or not isinstance(notional, (int, float))
                    or not math.isfinite(float(notional)) or float(notional) <= 0):
                raise ValueError("paper portfolio snapshot contains invalid position evidence")
            if symbol in current_positions:
                raise ValueError("paper portfolio snapshot contains duplicate symbols")
            current_positions[symbol] = float(notional)
            group_exposure[group] = group_exposure.get(group, 0.0) + float(notional)

        bound = dict(payload)
        # Never trust caller-provided equity or exposure when an account snapshot exists.
        bound["equity"] = snapshot["equity"]
        bound["available_cash"] = snapshot["cash"]
        bound["current_positions"] = current_positions
        bound["correlated_group_exposure"] = group_exposure
        bound["_eti_validated_portfolio_snapshot"] = {
            "account_id": account_id, "base_currency": account["base_currency"],
            "snapshot_id": snapshot["id"], "as_of": snapshot["as_of"],
            "source": "owner_attested",
            "freshness_seconds": max(0, int(age_seconds)),
        }
        return bound

    def run_once(self) -> dict[str, Any] | None:
        # Seed public research, simulated paper and owner-authorized live spot tasks.
        # Live orders are reachable only through the supervisor-gated live.tick capability.
        scheduled = self.store.enqueue_due_autonomy_tasks()
        research_scheduled = int(scheduled.get("research_tasks", 0))
        training_scheduled = int(scheduled.get("model_training_tasks", 0))
        paper_scheduled = self.store.enqueue_due_paper_tasks()
        live_scheduled = self.store.enqueue_due_live_execution_tasks()
        if research_scheduled or training_scheduled or paper_scheduled or live_scheduled:
            print(json.dumps({"event": "autonomy.tasks_scheduled",
                              "research_tasks": research_scheduled,
                              "model_training_tasks": training_scheduled,
                              "paper_tasks": paper_scheduled,
                              "live_tasks": live_scheduled},
                             separators=(",", ":")), flush=True)
        self.store.record_agent_worker_heartbeat(self.worker_id, self.registry.capabilities())
        task = self.store.claim_next_agent_task(
            lease_seconds=self.lease_seconds,
            owner_id=self.owner_id,
            workspace_id=self.workspace_id,
        )
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
            if capability == "portfolio.allocate":
                task_input = self._bind_paper_portfolio_snapshot(task, task_input)
            if capability == "live.tick":
                if (task.get("requester") != "eti.live.supervisor"
                        or not isinstance(task_input, dict)
                        or task_input.get("scheduled_by") != "eti.live.supervisor"):
                    raise ValueError("live execution tasks must be created by the owner-authorized supervisor")
                task_input = {**task_input, "owner_id": task["owner_id"], "task_id": task["id"]}
            result = self.registry.execute(capability, task_input)
            if capability == "portfolio.allocate" and isinstance(result, dict):
                result["portfolio_snapshot"] = task_input["_eti_validated_portfolio_snapshot"]
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

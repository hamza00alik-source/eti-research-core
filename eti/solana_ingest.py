"""PumpPortal -> ETI durable Solana discovery-event ingestion process.

Run as a separate process from the API and trading workers. This process only
records public discovery events; it cannot sign transactions or place orders.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
from typing import Any

from .solana_research import PumpPortalStream
from .storage import MemoryStore


def normalize_pumpportal_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize only documented event essentials; reject ambiguous payloads."""
    mint = event.get("mint")
    signature = event.get("signature")
    tx_type = str(event.get("txType", "")).lower()
    if not isinstance(mint, str) or not 32 <= len(mint) <= 64:
        return None
    if not isinstance(signature, str) or not signature.strip():
        return None
    if tx_type in {"create", "new_token"}:
        event_type = "new_token"
    elif tx_type in {"migrate", "migration"}:
        event_type = "migration"
    else:
        return None
    source_time = event.get("timestamp")
    source_timestamp = str(source_time) if isinstance(source_time, (str, int, float)) else None
    return {
        "event_key": f"pumpportal:{signature}:{event_type}",
        "source": "pumpportal",
        "event_type": event_type,
        "mint": mint,
        "source_timestamp": source_timestamp,
        "payload": event,
    }


class PumpPortalIngestor:
    def __init__(self, owner_id: str, workspace_id: str, store: MemoryStore | None = None):
        self.owner_id = owner_id
        self.workspace_id = workspace_id
        self.store = store or MemoryStore()

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stream = PumpPortalStream()
        async for event in stream.events(stop=stop):
            normalized = normalize_pumpportal_event(event)
            if normalized is None:
                continue
            try:
                result = self.store.record_solana_research_event(
                    self.owner_id, self.workspace_id, normalized["event_key"],
                    normalized["source"], normalized["event_type"], normalized["mint"],
                    normalized["payload"], normalized["source_timestamp"],
                )
                print(json.dumps({
                    "event": "solana.discovery.ingested",
                    "event_id": result["id"],
                    "event_key": result["event_key"],
                    "event_type": result["event_type"],
                    "mint": result["mint"],
                    "duplicate": result["duplicate"],
                    "payload_conflict": result["payload_conflict"],
                }, separators=(",", ":")), flush=True)
                if result["payload_conflict"]:
                    print(json.dumps({
                        "event": "solana.discovery.payload_conflict",
                        "event_key": result["event_key"],
                    }, separators=(",", ":")), flush=True)
            except (LookupError, ValueError) as exc:
                print(json.dumps({
                    "event": "solana.discovery.persist_failed",
                    "error_type": type(exc).__name__,
                }, separators=(",", ":")), flush=True)


async def _main() -> None:
    owner_id = os.environ.get("ETI_SOLANA_OWNER_ID", "").strip()
    workspace_id = os.environ.get("ETI_SOLANA_WORKSPACE_ID", "").strip()
    if not owner_id or not workspace_id:
        raise SystemExit("ETI_SOLANA_OWNER_ID and ETI_SOLANA_WORKSPACE_ID are required")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    await PumpPortalIngestor(owner_id, workspace_id).run(stop)


if __name__ == "__main__":
    asyncio.run(_main())

import asyncio
import sys
import types

import pytest

from eti.solana_ingest import PumpPortalIngestor
from eti.solana_research import PumpPortalStream


class _FakeSocket:
    def __init__(self, stop):
        self.stop = stop
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        async def items():
            yield '{"mint":"Mint111","signature":"Signature111","txType":"create"}'
            self.stop.set()
        return items()


class _FakeConnection:
    def __init__(self, socket):
        self.socket = socket

    async def __aenter__(self):
        return self.socket

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_pumpportal_stream_records_successful_connection_without_payload_or_secret(monkeypatch):
    stop = asyncio.Event()
    socket = _FakeSocket(stop)
    observations = []
    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(
        connect=lambda *_args, **_kwargs: _FakeConnection(socket)
    ))
    stream = PumpPortalStream(api_key="do-not-record-this", observation_callback=observations.append)

    events = [event async for event in stream.events(stop=stop)]

    assert len(events) == 1
    assert socket.sent == [
        '{"method": "subscribeNewToken"}',
        '{"method": "subscribeMigration"}',
    ]
    assert len(observations) == 1
    sample = observations[0]
    assert sample["provider"] == "pumpportal"
    assert sample["endpoint_id"] == "stream_connect"
    assert sample["status"] == "success"
    assert sample["latency_ms"] >= 0
    assert sample["observed_at"].endswith("Z")
    assert sample["error_code"] is None
    assert "do-not-record-this" not in repr(sample)
    assert "Mint111" not in repr(sample)
    assert "Signature111" not in repr(sample)


@pytest.mark.asyncio
async def test_pumpportal_stream_records_sanitized_connection_error(monkeypatch):
    stop = asyncio.Event()
    observations = []

    def fail_connect(*_args, **_kwargs):
        raise OSError("wss://host/?api-key=must-not-leak")

    async def stop_after_retry(_delay):
        stop.set()

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=fail_connect))
    monkeypatch.setattr("eti.solana_research.asyncio.sleep", stop_after_retry)
    stream = PumpPortalStream(api_key="private-key", observation_callback=observations.append)

    assert [event async for event in stream.events(stop=stop)] == []
    assert len(observations) == 1
    assert observations[0]["status"] == "error"
    assert observations[0]["error_code"] == "OSError"
    assert "must-not-leak" not in repr(observations)
    assert "private-key" not in repr(observations)


@pytest.mark.asyncio
async def test_pumpportal_telemetry_callback_failure_does_not_stop_ingestion(monkeypatch):
    stop = asyncio.Event()
    socket = _FakeSocket(stop)
    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(
        connect=lambda *_args, **_kwargs: _FakeConnection(socket)
    ))

    def broken_callback(_sample):
        raise RuntimeError("telemetry unavailable")

    stream = PumpPortalStream(observation_callback=broken_callback)
    events = [event async for event in stream.events(stop=stop)]
    assert len(events) == 1


def test_ingestor_binds_provider_telemetry_to_configured_owner_and_workspace():
    class Store:
        def __init__(self):
            self.args = None

        def record_solana_provider_observation(self, owner_id, workspace_id, **sample):
            self.args = (owner_id, workspace_id, sample)

    store = Store()
    ingestor = PumpPortalIngestor("trusted-owner", "trusted-workspace", store)
    ingestor._record_provider_observation({
        "provider": "pumpportal",
        "endpoint_id": "stream_connect",
        "observed_at": "2026-11-15T00:00:00Z",
        "status": "success",
        "latency_ms": 12.5,
        "error_code": None,
    })

    assert store.args[0:2] == ("trusted-owner", "trusted-workspace")
    assert store.args[2]["provider"] == "pumpportal"

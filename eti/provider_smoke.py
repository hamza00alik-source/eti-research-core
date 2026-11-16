"""Low-cost runtime smoke check for public Coinbase Exchange and Kraken spot OHLCV."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import validate_ohlcv
from .providers import CcxtPublicProvider


TIMEFRAME_MS = {"1h": 3_600_000}


def probe_provider(
    venue: str,
    provider: Any,
    *,
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    limit: int = 100,
    min_rows: int = 50,
    now_ms: int | None = None,
    max_age_bars: int = 3,
) -> dict[str, Any]:
    """Probe one venue; return diagnostics without merging feeds or hiding failures."""
    if timeframe not in TIMEFRAME_MS:
        raise ValueError("provider smoke check currently supports 1h only")
    frame = validate_ohlcv(provider.fetch_ohlcv(symbol, timeframe, limit))
    if len(frame) < min_rows:
        raise ValueError(f"provider returned {len(frame)} candles; at least {min_rows} required")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("provider timestamps are not unique and chronological")
    current = int(now_ms if now_ms is not None else pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    last_open_ms = int(timestamps[-1].timestamp() * 1000)
    if last_open_ms > current:
        raise ValueError("provider returned a future-dated candle")
    last_close_ms = last_open_ms + TIMEFRAME_MS[timeframe]
    if last_close_ms > current:
        raise ValueError("provider latest candle is incomplete")
    age_seconds = (current - last_close_ms) / 1000.0
    if age_seconds > max_age_bars * TIMEFRAME_MS[timeframe] / 1000:
        raise ValueError(f"provider latest completed candle is stale ({age_seconds:.0f}s)")
    return {
        "venue": venue, "symbol": symbol, "timeframe": timeframe,
        "status": "healthy", "rows": int(len(frame)),
        "first_candle": timestamps[0].isoformat(),
        "last_candle": timestamps[-1].isoformat(),
        "last_completed_candle_age_seconds": age_seconds,
        "execution_capability": False,
    }


def run_provider_smoke_check(
    providers: dict[str, Any] | None = None,
    *,
    now_ms: int | None = None,
    min_rows: int = 50,
    require_all: bool = False,
) -> dict[str, Any]:
    """Check each public spot provider independently; never merge or substitute candles."""
    configured = providers or {
        "coinbaseexchange": CcxtPublicProvider("coinbaseexchange", "spot"),
        "kraken": CcxtPublicProvider("kraken", "spot"),
    }
    results = {}
    for venue, provider in sorted(configured.items()):
        try:
            results[venue] = probe_provider(
                venue, provider, min_rows=min_rows, now_ms=now_ms,
            )
        except Exception as exc:
            results[venue] = {
                "venue": venue, "symbol": "BTC/USD", "timeframe": "1h",
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
                "execution_capability": False,
            }
    healthy = sum(item["status"] == "healthy" for item in results.values())
    status = "healthy" if healthy == len(results) else "degraded" if healthy else "unavailable"
    return {
        "schema_version": "eti.public_provider_smoke.v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": status, "healthy_provider_count": healthy,
        "provider_count": len(results), "providers": results,
        "training_performed": False, "execution_enabled": False,
        "warning": "Runtime reachability and candle quality smoke check only; not a guarantee of future availability or execution-grade market data.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe public spot market-data providers without training")
    parser.add_argument("--output", type=Path, default=Path("artifacts/provider-smoke.json"))
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.min_rows <= 100:
        parser.error("--min-rows must be between 1 and 100")
    report = run_provider_smoke_check(min_rows=args.min_rows, require_all=args.require_all)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["healthy_provider_count"] == 0:
        return 1
    if args.require_all and report["healthy_provider_count"] != report["provider_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

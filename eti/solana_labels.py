"""Causal, observation-based labels for Solana research snapshots.

Labels are delayed until a future observation exists beyond the requested horizon.
DexScreener sampled marks are not OHLCV candles or executable quotes. Exitability
and transaction-level slippage remain UNKNOWN unless separately measured.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

DEFAULT_HORIZONS_SECONDS = (300, 900, 3600, 21600, 86400)


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and number < float("inf") else None


def _pct_change(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0:
        return None
    return round((end / start - 1.0) * 100.0, 8)


def build_forward_labels(
    snapshots: Iterable[dict[str, Any]],
    *,
    now: datetime | None = None,
    horizons_seconds: tuple[int, ...] = DEFAULT_HORIZONS_SECONDS,
) -> list[dict[str, Any]]:
    """Build deterministic labels only from later observations of the same pool.

    The baseline is a scan snapshot. A target must be timestamped at/after the
    horizon and no later than now. A label is omitted if that target is absent.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if any(isinstance(h, bool) or not isinstance(h, int) or h <= 0 for h in horizons_seconds):
        raise ValueError("horizons_seconds must contain positive integers")
    grouped: dict[tuple[str, str], list[tuple[datetime, dict[str, Any]]]] = {}
    for item in snapshots:
        if not isinstance(item, dict) or item.get("capability") != "solana.market.scan":
            continue
        mint, pool = item.get("mint"), item.get("pool_address")
        observed = _timestamp(item.get("fetched_at"))
        payload = item.get("payload")
        if not isinstance(mint, str) or not isinstance(pool, str) or not pool or observed is None or not isinstance(payload, dict):
            continue
        grouped.setdefault((mint, pool), []).append((observed, item))
    labels: list[dict[str, Any]] = []
    for (mint, pool), observations in grouped.items():
        observations.sort(key=lambda pair: (pair[0], str(pair[1].get("id", ""))))
        for index, (base_time, base) in enumerate(observations):
            base_payload = base.get("payload") or {}
            base_evidence = base_payload.get("evidence") if isinstance(base_payload.get("evidence"), dict) else {}
            base_price = _finite_positive(base_evidence.get("price_usd"))
            base_liquidity = _finite_positive(base_evidence.get("liquidity_usd"))
            base_volume = _finite_positive(base_evidence.get("volume_h24_usd"))
            for horizon in horizons_seconds:
                due = base_time.timestamp() + horizon
                if current.timestamp() < due:
                    continue
                target = next(
                    ((stamp, item) for stamp, item in observations[index + 1:]
                     if stamp.timestamp() >= due and stamp <= current),
                    None,
                )
                if target is None:
                    continue
                target_time, target_item = target
                target_payload = target_item.get("payload") or {}
                target_evidence = target_payload.get("evidence") if isinstance(target_payload.get("evidence"), dict) else {}
                target_price = _finite_positive(target_evidence.get("price_usd"))
                target_liquidity = _finite_positive(target_evidence.get("liquidity_usd"))
                target_volume = _finite_positive(target_evidence.get("volume_h24_usd"))
                labels.append({
                    "mint": mint,
                    "pool_address": pool,
                    "baseline_snapshot_id": base.get("id"),
                    "target_snapshot_id": target_item.get("id"),
                    "horizon_seconds": horizon,
                    "baseline_at": base_time.isoformat().replace("+00:00", "Z"),
                    "target_at": target_time.isoformat().replace("+00:00", "Z"),
                    "actual_elapsed_seconds": round((target_time - base_time).total_seconds(), 3),
                    "sampled_price_return_pct": _pct_change(base_price, target_price),
                    "liquidity_change_pct": _pct_change(base_liquidity, target_liquidity),
                    "reported_h24_volume_change_pct": _pct_change(base_volume, target_volume),
                    "exitability_status": "UNKNOWN",
                    "realized_slippage_bps": None,
                    "failed_exit": None,
                    "label_method": "same_pool_future_sampled_observation_v1",
                    "limitations": [
                        "Sampled DexScreener marks are not OHLCV candles or executable quotes.",
                        "Reported rolling 24h volume change is not volume traded during this label horizon.",
                        "Exitability, realized slippage and failed exits are UNKNOWN without transaction-level or validated pool-curve simulation evidence.",
                    ],
                })
    return labels

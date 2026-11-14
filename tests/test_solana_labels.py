from datetime import datetime, timezone

from eti.solana_labels import build_forward_labels


def _snapshot(snapshot_id, timestamp, price, liquidity=100_000, pool="Pool111"):
    return {
        "id": snapshot_id,
        "capability": "solana.market.scan",
        "mint": "A" * 40,
        "pool_address": pool,
        "fetched_at": timestamp,
        "payload": {"evidence": {
            "price_usd": price,
            "liquidity_usd": liquidity,
            "volume_h24_usd": 50_000,
        }},
    }


def test_forward_label_uses_only_a_later_observation_after_horizon():
    rows = [
        _snapshot("base", "2026-01-01T00:00:00Z", 1.0, 100_000),
        _snapshot("early", "2026-01-01T00:04:59Z", 50.0, 1),
        _snapshot("target", "2026-01-01T00:05:01Z", 1.2, 80_000),
    ]
    labels = build_forward_labels(
        rows, now=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
        horizons_seconds=(300,),
    )
    label = next(item for item in labels if item["baseline_snapshot_id"] == "base")
    assert label["target_snapshot_id"] == "target"
    assert label["sampled_price_return_pct"] == 20.0
    assert label["liquidity_change_pct"] == -20.0
    assert label["exitability_status"] == "UNKNOWN"
    assert label["realized_slippage_bps"] is None
    assert label["failed_exit"] is None


def test_forward_label_is_not_created_before_horizon_or_without_future_sample():
    rows = [_snapshot("base", "2026-01-01T00:00:00Z", 1.0)]
    assert build_forward_labels(
        rows, now=datetime(2026, 1, 1, 0, 4, 59, tzinfo=timezone.utc),
        horizons_seconds=(300,),
    ) == []
    assert build_forward_labels(
        rows, now=datetime(2026, 1, 1, 0, 6, tzinfo=timezone.utc),
        horizons_seconds=(300,),
    ) == []


def test_forward_label_never_crosses_pool_identity():
    rows = [
        _snapshot("base", "2026-01-01T00:00:00Z", 1.0, pool="Pool111"),
        _snapshot("other-pool", "2026-01-01T00:10:00Z", 9.0, pool="Pool222"),
    ]
    assert build_forward_labels(
        rows, now=datetime(2026, 1, 1, 0, 20, tzinfo=timezone.utc),
        horizons_seconds=(300,),
    ) == []


def test_invalid_or_missing_market_marks_remain_null_not_zero():
    rows = [
        _snapshot("base", "2026-01-01T00:00:00Z", None, None),
        _snapshot("target", "2026-01-01T00:06:00Z", 1.5, 90_000),
    ]
    labels = build_forward_labels(
        rows, now=datetime(2026, 1, 1, 0, 10, tzinfo=timezone.utc),
        horizons_seconds=(300,),
    )
    label = next(item for item in labels if item["baseline_snapshot_id"] == "base")
    assert label["sampled_price_return_pct"] is None
    assert label["liquidity_change_pct"] is None

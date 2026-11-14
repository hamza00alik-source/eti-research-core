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


def test_chronological_group_split_purges_overlap_and_never_splits_pool():
    from eti.solana_labels import chronological_group_split

    rows = []
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(6):
        start = origin.replace(day=1 + day)
        end = start.replace(minute=5)
        rows.append({
            "mint": f"Mint{day}", "pool_address": f"Pool{day}",
            "baseline_at": start.isoformat().replace("+00:00", "Z"),
            "target_at": end.isoformat().replace("+00:00", "Z"),
        })
    split = chronological_group_split(rows, train_fraction=0.5, embargo_seconds=3600)
    train_groups = {(x["mint"], x["pool_address"]) for x in split["train"]}
    validation_groups = {(x["mint"], x["pool_address"]) for x in split["validation"]}
    assert train_groups.isdisjoint(validation_groups)
    assert split["train_group_count"] == len(train_groups)
    assert split["validation_group_count"] == len(validation_groups)
    assert split["purged_group_count"] >= 1
    assert all(x["target_at"] < split["boundary_at"] for x in split["train"])
    assert all(x["baseline_at"] >= split["validation_start_at"] for x in split["validation"])


def test_chronological_group_split_requires_enough_distinct_pools():
    from eti.solana_labels import chronological_group_split

    with pytest.raises(ValueError, match="three distinct"):
        chronological_group_split([{
            "mint": "A", "pool_address": "P",
            "baseline_at": "2026-01-01T00:00:00Z",
            "target_at": "2026-01-01T00:05:00Z",
        }])


def test_future_target_after_generation_cutoff_is_not_used():
    rows = [
        _snapshot("base", "2026-01-01T00:00:00Z", 1.0),
        _snapshot("future", "2026-01-01T00:10:00Z", 2.0),
    ]
    assert build_forward_labels(
        rows, now=datetime(2026, 1, 1, 0, 9, tzinfo=timezone.utc),
        horizons_seconds=(300,),
    ) == []


def test_label_generation_rejects_naive_cutoff_datetime():
    import pytest

    with pytest.raises(ValueError, match="timezone-aware"):
        build_forward_labels(
            [_snapshot("base", "2026-01-01T00:00:00Z", 1.0)],
            now=datetime(2026, 1, 1, 0, 10),
            horizons_seconds=(300,),
        )


def test_expanding_walk_forward_splits_are_chronological_grouped_and_purged():
    from eti.solana_labels import chronological_walk_forward_splits

    rows = []
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for day in range(10):
        start = origin.replace(day=1 + day)
        end = start.replace(minute=5)
        rows.append({
            "mint": f"Mint{day}", "pool_address": f"Pool{day}",
            "baseline_at": start.isoformat().replace("+00:00", "Z"),
            "target_at": end.isoformat().replace("+00:00", "Z"),
        })
    folds = chronological_walk_forward_splits(
        rows, n_splits=3, min_train_groups=3, embargo_seconds=3600
    )
    assert len(folds) == 3
    previous_validation_start = None
    for fold in folds:
        train_groups = {(x["mint"], x["pool_address"]) for x in fold["train"]}
        validation_groups = {(x["mint"], x["pool_address"]) for x in fold["validation"]}
        assert train_groups.isdisjoint(validation_groups)
        assert all(x["target_at"] < fold["boundary_at"] for x in fold["train"])
        assert all(x["baseline_at"] >= fold["validation_start_at"] for x in fold["validation"])
        assert fold["purged_group_count"] >= 1
        if previous_validation_start is not None:
            assert fold["validation_start_at"] > previous_validation_start
        previous_validation_start = fold["validation_start_at"]
    assert folds[0]["train_group_count"] < folds[-1]["train_group_count"]


def test_walk_forward_split_rejects_insufficient_groups():
    from eti.solana_labels import chronological_walk_forward_splits

    with pytest.raises(ValueError, match="not enough distinct"):
        chronological_walk_forward_splits([{
            "mint": "Mint1", "pool_address": "Pool1",
            "baseline_at": "2026-01-01T00:00:00Z",
            "target_at": "2026-01-01T00:05:00Z",
        }])

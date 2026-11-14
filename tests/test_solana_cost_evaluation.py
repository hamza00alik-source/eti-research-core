import pytest

from eti.solana_cost_evaluation import evaluate_stored_solana_walk_forward


def _fixture():
    exported, prepared = [], []
    for index in range(8):
        day = 1 + index * 2
        stamp = f"2026-01-{day:02d}T00:00:00Z"
        label_id = f"ledger-label-{index}"
        mint = f"ledger-mint-{index}"
        pool = f"ledger-pool-{index}"
        exported.append({
            "id": label_id,
            "mint": mint,
            "pool_address": pool,
            "horizon_seconds": 3600,
            "label": {
                "baseline_at": stamp,
                "target_at": f"2026-01-{day:02d}T01:00:00Z",
            },
        })
        prepared.append({
            "label_id": label_id,
            "mint": mint,
            "pool_address": pool,
            "baseline_at": stamp,
            "horizon_seconds": 3600,
            "features": {"txns_h1_buys": 8, "txns_h1_sells": 2},
            "target_sampled_price_return_pct": 5.0,
        })
    return exported, {
        "status": "PREPARATION_ONLY",
        "training_authorized": False,
        "rows": prepared,
    }


def _evidence(label):
    components = {}
    for name in (
        "spread", "pool_price_impact", "protocol_fee",
        "priority_fee", "latency_slippage",
    ):
        components[name] = {
            "cost_bps": 10,
            "unit": "bps_of_entry_notional",
            "measurement": "measured",
            "source_id": f"{label}-{name}",
            "observed_at": "2026-11-14T12:00:00Z",
        }
    return {
        "label_id": label["id"],
        "mint": label["mint"],
        "pool_address": label["pool_address"],
        "cost_components": components,
        "exit_evidence": {
            "status": "exit_observed",
            "source_id": f"{label['id']}-exit",
            "transaction_id": f"{label['id']}-signature",
            "observed_at": "2026-11-14T12:01:00Z",
        },
    }


class FakeLedger:
    def __init__(self, rows_by_label):
        self.rows_by_label = rows_by_label
        self.calls = []

    def list_latest_solana_execution_cost_evidence(self, owner, workspace, label_ids):
        self.calls.append((owner, workspace, tuple(label_ids)))
        selected = []
        for label_id in label_ids:
            records = sorted(
                self.rows_by_label.get(label_id, []),
                key=lambda item: (item["created_at"], item["id"]),
                reverse=True,
            )
            if records:
                selected.append({**records[0], "label_id": label_id})
        return selected


def test_stored_evaluation_uses_newest_intact_scoped_record():
    exported, prepared = _fixture()
    ledger_rows = {}
    for label in exported:
        ledger_rows[label["id"]] = [{
            "id": f"record-{label['id']}",
            "created_at": "2026-11-14T12:00:00Z",
            "integrity_valid": True,
            "evidence_sha256": "a" * 64,
            "evidence": _evidence(label),
        }]
    store = FakeLedger(ledger_rows)

    report = evaluate_stored_solana_walk_forward(
        store, "owner-1", "workspace-1", exported, prepared,
        n_splits=2, min_train_groups=3, embargo_seconds=3600,
    )

    costs = report["execution_cost_evidence"]
    assert costs["validation_rows_checked"] > 0
    assert costs["ledger_integrity_valid_count"] == costs["validation_rows_checked"]
    assert costs["ledger_integrity_invalid_count"] == 0
    assert costs["complete_evidence_count"] == costs["validation_rows_checked"]
    assert costs["net_return_pct"] is None
    assert costs["realized_pnl_available"] is False
    assert all(call[:2] == ("owner-1", "workspace-1") for call in store.calls)
    assert all(row["ledger_evidence_status"] == "NEWEST_RECORD_INTEGRITY_VALID" for row in costs["rows"])


def test_tampered_newest_record_blocks_fallback_to_older_valid_record():
    exported, prepared = _fixture()
    ledger_rows = {}
    for label in exported:
        ledger_rows[label["id"]] = [
            {
                "id": f"newest-tampered-{label['id']}",
                "created_at": "2026-11-14T13:00:00Z",
                "integrity_valid": False,
                "evidence_sha256": "b" * 64,
                "evidence": _evidence(label),
            },
            {
                "id": f"older-valid-{label['id']}",
                "created_at": "2026-11-14T12:00:00Z",
                "integrity_valid": True,
                "evidence_sha256": "a" * 64,
                "evidence": _evidence(label),
            },
        ]
    report = evaluate_stored_solana_walk_forward(
        FakeLedger(ledger_rows), "owner-1", "workspace-1", exported, prepared,
        n_splits=2, min_train_groups=3, embargo_seconds=3600,
    )

    costs = report["execution_cost_evidence"]
    assert costs["validation_rows_checked"] > 0
    assert costs["ledger_integrity_invalid_count"] == costs["validation_rows_checked"]
    assert costs["complete_evidence_count"] == 0
    assert costs["incomplete_evidence_count"] == costs["validation_rows_checked"]
    assert all(row["ledger_record_id"].startswith("newest-tampered-") for row in costs["rows"])
    assert all("ledger_integrity_invalid" in row["invalid_components"] for row in costs["rows"])
    assert all(row["ledger_evidence_status"] == "NEWEST_RECORD_INTEGRITY_INVALID" for row in costs["rows"])


def test_stored_evaluation_requires_string_label_ids():
    with pytest.raises(ValueError, match="string label IDs"):
        evaluate_stored_solana_walk_forward(
            FakeLedger({}), "owner-1", "workspace-1",
            [{"id": 123}], {"status": "PREPARATION_ONLY", "training_authorized": False, "rows": []},
        )

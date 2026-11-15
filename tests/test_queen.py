import pytest

from eti.queen import ColonyConfig, QueenOrchestrator, calculate_worker_target


@pytest.mark.parametrize(
    ("queued", "maximum", "backlog", "expected"),
    [
        (0, 4, 2, 0),
        (1, 4, 2, 1),
        (2, 4, 2, 1),
        (3, 4, 2, 2),
        (8, 4, 2, 4),
        (100, 3, 1, 3),
    ],
)
def test_worker_target_scales_with_queue_and_respects_cap(queued, maximum, backlog, expected):
    assert calculate_worker_target(
        queued, max_workers=maximum, backlog_per_worker=backlog
    ) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_workers": 0},
        {"max_workers": 9},
        {"max_workers": True},
        {"backlog_per_worker": 0},
        {"backlog_per_worker": 11},
        {"poll_seconds": 0},
        {"lease_seconds": 4},
    ],
)
def test_colony_config_rejects_unsafe_limits(kwargs):
    with pytest.raises(ValueError):
        ColonyConfig(**kwargs)


@pytest.mark.parametrize("queued", [-1, True, 1.5])
def test_worker_target_rejects_invalid_queue_counts(queued):
    with pytest.raises(ValueError):
        calculate_worker_target(queued, max_workers=2, backlog_per_worker=2)


def test_queen_reconcile_starts_bounded_worker_slots(tmp_path):
    from eti.storage import MemoryStore

    store = MemoryStore(tmp_path / "queen.sqlite3")
    workspace = store.create_workspace("owner", "queen tests")
    created = []

    class FakeWorker:
        def run_once(self):
            return None

    def factory():
        created.append(True)
        return FakeWorker()

    queen = QueenOrchestrator(
        "owner", workspace["id"], store=store,
        config=ColonyConfig(max_workers=3, backlog_per_worker=1, poll_seconds=0.1),
        worker_factory=factory,
    )
    try:
        status = queen.reconcile(queued_tasks=2)
        assert status["target_workers"] == 2
        assert status["active_workers"] == 2
        assert len(created) == 2
        status = queen.reconcile(queued_tasks=100)
        assert status["target_workers"] == 3
        assert status["active_workers"] <= 3
    finally:
        queen.stop()

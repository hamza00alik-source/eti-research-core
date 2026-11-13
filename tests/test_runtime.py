from eti.runtime import _worker_loop


def test_embedded_worker_survives_one_transient_iteration_error(monkeypatch):
    calls = {"count": 0}

    class BrokenWorker:
        def run_once(self):
            calls["count"] += 1
            raise RuntimeError("temporary sqlite lock")

    class StopAfterWait:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _seconds):
            self.stopped = True

    monkeypatch.setattr("eti.runtime.AgentWorker", BrokenWorker)
    stop = StopAfterWait()
    _worker_loop(stop, 0.1)

    assert calls["count"] == 1
    assert stop.stopped is True

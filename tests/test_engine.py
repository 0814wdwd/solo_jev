import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from solo_decision import DecisionEngine, DecisionResponse, DecisionSpec, LAYOUTS


class FakeBackend:
    supports_cache_salt = True

    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.calls = []
        self.closed = False

    def decide(self, state, spec, *, cache_salt=None):
        row = json.loads(state)
        with self.lock:
            self.active += 1
            self.peak = max(self.active, self.peak)
            self.calls.append((row, cache_salt))
        try:
            time.sleep(0.002 if row["amount"] == "1" else 0.0001)
            if row["amount"] == "ERROR":
                raise ValueError("intentional failure")
            p = (0.05, 0.95) if int(row["amount"]) > 1 else (0.9, 0.1)
            return DecisionResponse(p, 100, 50)
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        self.closed = True


@pytest.mark.parametrize("method", LAYOUTS)
def test_complete_cells_original_positions_duplicate_index_and_bounded_requests(method):
    frame = pd.DataFrame({"amount": [3, 1, 2, 1, 1], "notes": ["x\n\"", "北京", "", "北京", "北京"]},
                         index=pd.Index(["z", "z", "a", "b", "b"], name="record"))
    backend = FakeBackend()
    with DecisionEngine(backend=backend, concurrency=2) as engine:
        result = engine.scan(frame, "amount exceeds one?", method=method)
    assert result.decisions.tolist() == [True, False, True, False, False]
    pd.testing.assert_index_equal(result.to_pandas().index, frame.index)
    sent = sorted(json.dumps(row, sort_keys=True) for row, _ in backend.calls)
    expected = sorted(json.dumps({k: str(v) for k, v in row.items()}, sort_keys=True) for row in frame.to_dict("records"))
    assert sent == expected
    assert backend.peak <= 2 and backend.closed
    assert result.prompt_tokens == 500 and result.cached_tokens == 250


def test_compare_uses_fresh_namespaces_and_reversed_order():
    backend = FakeBackend()
    data = np.array([[1], [2], [3]])
    with DecisionEngine(backend=backend) as engine:
        result = engine.compare(data, "question", columns=["amount"], repeats=2,
                                methods=["original", "solo"], truth=[False, True, True])
    assert [r["method"] for r in result.runs] == ["original", "solo", "solo", "original"]
    assert len({r["cache_salt"] for r in result.runs}) == 4
    assert all(s["accuracy"] == 1 and s["agreement_with_original"] == 1 for s in result.summary)
    assert result.to_pandas().index.tolist() == ["original", "solo"]


def test_failure_never_becomes_false_and_inflight_work_is_drained():
    backend = FakeBackend()
    with DecisionEngine(backend=backend, concurrency=2) as engine:
        with pytest.raises(RuntimeError, match="input row position 1"):
            engine.scan([{"amount": "1"}, {"amount": "ERROR"}, {"amount": "3"}], "question", method="original")
        assert backend.active == 0


def test_empty_input_no_requests_and_closed_engine_fails():
    backend = FakeBackend()
    engine = DecisionEngine(backend=backend)
    result = engine.scan(np.empty((0, 2)), "question")
    assert result.probabilities.shape == (0, 2) and not backend.calls
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.scan(np.empty((0, 2)), "question")


def test_missing_usage_is_unknown():
    class Backend:
        def decide(self, state, spec, *, cache_salt=None):
            return DecisionResponse((0.2, 0.8))
    with DecisionEngine(backend=Backend()) as engine:
        result = engine.scan([[1]], "question")
    assert result.prompt_tokens is None and result.cached_tokens is None
    assert result.metrics()["cached_fraction"] is None


def test_large_input_does_not_enqueue_one_future_per_row():
    queued = []
    class Backend(FakeBackend):
        def decide(self, state, spec, *, cache_salt=None):
            queued.append(engine._pool._work_queue.qsize())
            return super().decide(state, spec, cache_salt=cache_salt)
    with DecisionEngine(backend=Backend(), concurrency=2) as engine:
        engine.scan(np.ones((200, 1), dtype=int), "question", columns=["amount"], method="original")
    assert max(queued) <= 2


@pytest.mark.parametrize("kind,options", [("bad", None), ("choice", ["a", "a"]),
                                        ("choice", "ab"), ("score", [1, 2]), ("noul", [False, True])])
def test_invalid_decision_specs(kind, options):
    with pytest.raises(ValueError):
        DecisionSpec.create("question", kind, options)

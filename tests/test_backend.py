import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import numpy as np
import pytest

from solo_layout import DecisionEngine, DecisionSpec, JevBackend, VllmJevBackend


@pytest.fixture
def metadata(tmp_path):
    (tmp_path / "adapter_vllm").mkdir()
    (tmp_path / "adapter_vllm/decision_head.json").write_text(json.dumps({
        "bias": [0.0] * 24, "verbalizer_ids": list(range(24)),
        "slots": {"template_version": "bare-v1", "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}}
    }))
    (tmp_path / "calibration.json").write_text(json.dumps({"per_kind": {"noul": 1, "score": 1, "choice": 1}}))
    return tmp_path


def test_real_http_protocol_all_heads_and_payload_isolation(metadata):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload))
            ids = payload["allowed_token_ids"]
            scores = {f"token_id:{i}": (-0.01 if i == ids[-1] else -10.0) for i in ids}
            body = json.dumps({
                "id": "cmpl-test",
                "choices": [{"logprobs": {"top_logprobs": [scores]}}],
                "usage": {
                    "prompt_tokens": 800,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 528, "created_cache_tokens": 272},
                },
                "metrics": {
                    "queue_time_ms": 1.5,
                    "time_to_first_token_ms": 12.25,
                    "generation_time_ms": 0.0,
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with DecisionEngine(f"http://127.0.0.1:{server.server_port}/v1", model_dir=metadata) as engine:
            for kind, options, expected in [("noul", None, True), ("score", None, 5), ("choice", ["a", "b"], "b")]:
                result = engine.scan([["complete value"]], "question", kind=kind, options=options, cache_salt="trial")
                assert result.decisions.tolist() == [expected]
                assert result.cached_tokens == 528
                assert result.prompt_tokens == 800 and result.completion_tokens == 1
                assert result.created_cache_tokens == 272
                assert result.request_traces[0].request_id == "cmpl-test"
                assert result.request_traces[0].queue_time_ms == 1.5
                assert result.request_traces[0].engine_prefill_interval_ms == 12.25
                assert result.request_traces[0].generation_time_ms == 0.0
        for path, payload in calls:
            assert path == "/v1/completions"
            assert payload["cache_salt"] == "trial"
            assert payload["max_tokens"] == 1 and payload["add_special_tokens"] is False
            assert "complete value" in payload["prompt"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_invalid_server_metric_is_rejected(metadata):
    backend = JevBackend(model_dir=metadata)
    backend.prepare(DecisionSpec.create("question"))
    assert backend._head is not None
    from solo_layout.backend import _optional_count, _optional_milliseconds
    with pytest.raises(RuntimeError, match="prompt_tokens"):
        _optional_count({"prompt_tokens": -1}, "prompt_tokens")
    with pytest.raises(RuntimeError, match="queue_time_ms"):
        _optional_milliseconds({"queue_time_ms": float("nan")}, "queue_time_ms")


def test_vllm_jev_choice_protocol_all_heads_and_prefix_cache():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, payload, self.headers.get("Authorization")))
            probabilities = {option: 0.01 for option in payload["options"]}
            probabilities[payload["options"][-1]] = 0.97
            body = json.dumps({
                "choice": payload["options"][-1],
                "probabilities": probabilities,
                "prompt_tokens": [400] * len(payload["options"]),
                "cached_tokens": [264] * len(payload["options"]),
                "prefix_cache_requested": payload["use_prefix_cache"],
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("x-request-id", "jev-test")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        backend = VllmJevBackend(
            f"http://127.0.0.1:{server.server_port}/v1",
            api_key="secret", temperature=0.8)
        with DecisionEngine(backend=backend) as engine:
            cases = [
                ("noul", None, True, ["no", "yes"]),
                ("score", None, 5, ["0", "1", "2", "3", "4", "5"]),
                ("choice", ["a", "b"], "b", ["a", "b"]),
            ]
            for kind, options, expected, labels in cases:
                result = engine.scan(
                    [["complete value"]], "question", kind=kind,
                    options=options, cache_salt="same-batch")
                assert result.decisions.tolist() == [expected]
                assert result.prompt_tokens == 400 * len(labels)
                assert result.cached_tokens == 264 * len(labels)
                assert result.completion_tokens == 0
                assert result.request_traces[0].request_id == "jev-test"

        for (path, payload, authorization), (_, _, _, labels) in zip(calls, cases):
            assert path == "/plugins/vllm-jev/choice"
            assert payload["cache_salt"] == "same-batch"
            assert payload["use_prefix_cache"] is True
            assert payload["temperature"] == 0.8
            assert payload["options"] == labels
            assert isinstance(payload["state"], str)
            assert "complete value" in payload["state"]
            assert authorization == "Bearer secret"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("body, message", [
    ({"probabilities": {"no": 1.0}}, "every candidate"),
    ({"probabilities": {"no": 0.5, "yes": float("nan")}}, "invalid candidate"),
    ({"probabilities": {"no": 0.5, "yes": 0.5}, "prompt_tokens": [-1]},
     "invalid prompt_tokens"),
])
def test_vllm_jev_rejects_malformed_responses(body, message, monkeypatch):
    class Response:
        headers = {}

        def raise_for_status(self):
            pass

        def json(self):
            return body

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    backend = VllmJevBackend()
    monkeypatch.setattr(backend, "_session", lambda: Session())
    with pytest.raises(RuntimeError, match=message):
        backend.decide("complete state", DecisionSpec.create("question"))

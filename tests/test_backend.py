import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import numpy as np
import pytest

from solo_decision import DecisionEngine, DecisionSpec, JevBackend


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
            body = json.dumps({"choices": [{"logprobs": {"top_logprobs": [scores]}}],
                               "usage": {"prompt_tokens": 800, "prompt_tokens_details": {"cached_tokens": 528}}}).encode()
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
        for path, payload in calls:
            assert path == "/v1/completions"
            assert payload["cache_salt"] == "trial"
            assert payload["max_tokens"] == 1 and payload["add_special_tokens"] is False
            assert "complete value" in payload["prompt"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()

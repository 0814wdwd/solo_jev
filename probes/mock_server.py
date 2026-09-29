#!/usr/bin/env python3
"""A stdlib mock of POST /v1/systemone, for testing the probes without a key.

It validates the request against the schema pinned from jeff
(src/jeff/core/schemas.py) and returns a well-formed response including
usage.input_tokens. It does no inference -- answers are deterministic filler.

Its only job is to prove the probes are wire-correct and their analysis code
runs, so that a real run against the hosted API is not the first time any of
this executes. Optionally simulates a prefix cache so probe_cache's verdict
logic can be checked against a known ground truth:

  python3 probes/mock_server.py --port 8099 --cache prefix --delay-per-ktok 0.02
  python3 probes/probe_cache.py --base-url http://localhost:8099 --reps 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

CFG = {"cache": "none", "delay_per_ktok": 0.02, "base_delay": 0.05, "seen": {}}


def _tokens(obj) -> int:
    """Crude but stable token estimate, mirroring len/4."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return max(1, len(text) // 4)


def _validate(payload: dict) -> tuple[bool, str]:
    if "state" not in payload:
        return False, "state is required"
    if not payload.get("model") and not payload.get("selectedModels"):
        return False, "model is required"
    qs = payload.get("questions")
    if not isinstance(qs, dict) or not qs:
        return False, "questions must be a non-empty object keyed by name"
    for name, q in qs.items():
        if not isinstance(q, dict) or q.get("type") not in ("noul", "choice", "score"):
            return False, f"question {name!r}: type must be noul|choice|score"
        if q["type"] == "choice":
            if not isinstance(q.get("criteria"), dict) or not q["criteria"]:
                return False, f"question {name!r}: choice needs a non-empty criteria object"
        if q["type"] == "score":
            if not isinstance(q.get("criteria"), list) or not q["criteria"]:
                return False, f"question {name!r}: score needs a non-empty criteria list"
    return True, ""


def _answer(name: str, q: dict) -> dict:
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    if q["type"] == "noul":
        return {"type": "noul", "noul": (h % 1000) / 1000.0}
    if q["type"] == "choice":
        opts = list(q["criteria"])
        pick = opts[h % len(opts)]
        p = {o: (1.0 if o == pick else 0.0) for o in opts}
        return {"type": "choice", "choice": pick, "confidence": 0.9, "probabilities": p}
    levels = q["criteria"]
    return {"type": "score", "score": float(h % len(levels)), "confidence": 0.8,
            "legend": {str(i): lv for i, lv in enumerate(levels)},
            "probabilities": {str(i): 1.0 / len(levels) for i in range(len(levels))}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: dict) -> None:
        blob = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"models": [{"name": "jev-1.13.0", "description": "mock",
                                         "release_date": "2026-09-15"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            self._send(400, {"error": f"bad json: {exc}"})
            return
        ok, msg = _validate(payload)
        if not ok:
            self._send(422, {"error": msg})
            return

        state = payload["state"]
        qs = payload["questions"]
        state_tok = _tokens(state)
        q_tok = sum(_tokens(q) for q in qs.values())

        # Simulated cache, so probe_cache's verdict logic can be checked.
        billed = state_tok + q_tok
        text = state if isinstance(state, str) else json.dumps(state)
        hit = False
        if CFG["cache"] == "exact":
            key = hashlib.sha256(text.encode()).hexdigest()
            hit = key in CFG["seen"]
            CFG["seen"][key] = True
        elif CFG["cache"] == "prefix":
            key = hashlib.sha256(text[: max(1, len(text) // 2)].encode()).hexdigest()
            hit = key in CFG["seen"]
            CFG["seen"][key] = True
        if hit:
            billed = q_tok + state_tok // 10

        time.sleep(CFG["base_delay"] + (billed / 1000.0) * CFG["delay_per_ktok"])
        self._send(200, {
            "model": payload.get("model") or payload["selectedModels"][0],
            "answers": {name: _answer(name, q) for name, q in qs.items()},
            "usage": {"input_tokens": billed, "output_tokens": 0},
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--cache", choices=("none", "exact", "prefix"), default="none",
                    help="simulate a cross-request cache, to test the probe's verdict logic")
    ap.add_argument("--delay-per-ktok", type=float, default=0.02)
    ap.add_argument("--base-delay", type=float, default=0.05)
    args = ap.parse_args()
    CFG.update(cache=args.cache, delay_per_ktok=args.delay_per_ktok, base_delay=args.base_delay)
    print(f"# mock systemone on http://localhost:{args.port}  cache={args.cache}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

"""OpenRouter Decisions API transport for Jev.

Confirmed empirically on 2026-09-28:
  POST https://openrouter.ai/api/alpha/decisions
  model: "typesafe/jev-1.13"      ("typesafe/jev-latest" does NOT exist here;
                                   the catalogue also lists "typesafe/jev-router")
  body:  {"model", "state", "questions": {"<name>": {...}}}

`questions` must be a RECORD keyed by name. A list is rejected with
400 "expected record, received array", which settles the schema question: the
shape pinned from jeff's schemas.py is correct.

Response adds OpenRouter fields on top of the native shape:
  {"model", "answers", "usage": {"input_tokens", "output_tokens", "cost"},
   "id", "provider"}

Measured: 305 input tokens cost $1.281e-05, i.e. exactly 305/1e6 * $0.042, so
OpenRouter passes through TypeSafe's posted input price and bills no output.

CAVEAT for the research: this is a proxy in front of TypeSafe. Encoding rankings
and parallelism shape survive the indirection; a cache result does NOT -- any hit
observed here could be OpenRouter's own layer, not Jev's. Settling the cache
question needs a TypeSafe-direct key.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import requests

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://openrouter.ai/api"
ENDPOINT = "/alpha/decisions"
MODEL = "typesafe/jev-1.13"
POSTED_INPUT_PRICE_PER_MTOK = 0.042


def load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    p = ROOT / ".env.local"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "TYPESAFE_API_KEY"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def noul(name: str, instructions: str) -> Dict[str, Any]:
    return {name: {"type": "noul", "instructions": instructions}}


def choice(name: str, instructions: str, options: Sequence[str] | Dict[str, Any]) -> Dict[str, Any]:
    criteria = {o: None for o in options} if not isinstance(options, dict) else dict(options)
    return {name: {"type": "choice", "instructions": instructions, "criteria": criteria}}


def score(name: str, instructions: str, levels: Sequence[Any]) -> Dict[str, Any]:
    return {name: {"type": "score", "instructions": instructions, "criteria": list(levels)}}


def merge(*qs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for q in qs:
        out.update(q)
    return out


@dataclass
class Result:
    ok: bool
    status: int
    latency_s: float
    body: Any = None
    error: Optional[str] = None

    @property
    def usage(self) -> Dict[str, Any]:
        if isinstance(self.body, dict) and isinstance(self.body.get("usage"), dict):
            return self.body["usage"]
        return {}

    @property
    def input_tokens(self) -> Optional[int]:
        v = self.usage.get("input_tokens")
        return v if isinstance(v, int) else None

    @property
    def output_tokens(self) -> Optional[int]:
        v = self.usage.get("output_tokens")
        return v if isinstance(v, int) else None

    @property
    def cost(self) -> Optional[float]:
        v = self.usage.get("cost")
        return float(v) if isinstance(v, (int, float)) else None

    @property
    def answers(self) -> Dict[str, Any]:
        if isinstance(self.body, dict) and isinstance(self.body.get("answers"), dict):
            return self.body["answers"]
        return {}


class OpenRouterJev:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: str = MODEL, timeout: float = 180.0):
        env = load_env()
        self.api_key = api_key or env.get("OPENROUTER_API_KEY", "")
        self.base_url = (base_url or env.get("OPENROUTER_BASE_URL", BASE_URL)).rstrip("/")
        self.model = model
        self.timeout = timeout
        self.spent = 0.0
        self.calls = 0
        self._s = requests.Session()
        if not self.api_key:
            raise SystemExit("no OPENROUTER_API_KEY (put it in .env.local)")

    def ask(self, state: Any, questions: Dict[str, Any] | Sequence[Dict[str, Any]],
            model: Optional[str] = None) -> Result:
        if not isinstance(questions, dict):
            questions = merge(*questions)
        payload = {"model": model or self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        t0 = time.time()
        try:
            r = self._s.post(self.base_url + ENDPOINT, headers=headers,
                             json=payload, timeout=self.timeout)
        except Exception as exc:
            return Result(ok=False, status=-1, latency_s=time.time() - t0, error=repr(exc))
        dt = time.time() - t0
        try:
            body = r.json()
        except Exception:
            body = r.text[:2000]
        res = Result(ok=r.status_code == 200, status=r.status_code, latency_s=dt, body=body,
                     error=None if r.status_code == 200 else json.dumps(body)[:400])
        self.calls += 1
        if res.cost:
            self.spent += res.cost
        return res

    def budget_line(self) -> str:
        return f"# {self.calls} calls, ${self.spent:.6f} spent this run"

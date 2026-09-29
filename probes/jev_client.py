"""Minimal Jev client for the probes.

SCHEMA: pinned against jeff's src/jeff/core/schemas.py, whose docstring states it
matches typesafe_sdk/_schemas/models.py. Confirmed shape:

    POST /v1/systemone
    {"model": "jev-latest",
     "state": <str | dict | list>,
     "questions": {"<name>": {"type": "noul"|"choice"|"score",
                              "instructions": <str|dict|list>,
                              "criteria": ...}}}

`questions` is an OBJECT keyed by name, not a list, and each question carries
`instructions` plus a type-specific `criteria`:
    noul   -> {"true": ..., "false": ...} or omitted
    choice -> {"<option>": <description|null>, ...}   (>=1 option)
    score  -> ["<level>", ...]                        (ordered, >=1)

Response: {"model", "answers": {"<name>": ...}, "usage": {"input_tokens",
"output_tokens"}}. Billed input tokens are usage.input_tokens.

An official `typesafe-sdk` PyPI package exists; this client stays dependency-free
so the probes can run anywhere, and every probe takes --base-url so it can run
against a local jeff with no API key at all.

TRANSPORTS: the same request body is served by two endpoints, and the client picks
by base_url, so every probe reaches either one through --base-url alone.

  TypeSafe direct  https://api.typesafe.ai   POST /v1/systemone     model jev-latest
  OpenRouter       https://openrouter.ai/api POST /alpha/decisions  model typesafe/jev-1.13

Verified on OpenRouter 2026-09-28: `typesafe/jev-latest` does not exist there, a
list-valued `questions` is rejected ("expected record, received array"), and the
response carries usage.input_tokens / output_tokens / cost.

Auth comes from TYPESAFE_API_KEY or OPENROUTER_API_KEY, via the environment or
.env.local. Nothing here prints the key.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

DEFAULT_BASE_URL = "https://api.typesafe.ai"
ENDPOINT = "/v1/systemone"
DEFAULT_MODEL = "jev-latest"

OPENROUTER_HOST = "openrouter.ai"
OPENROUTER_ENDPOINT = "/alpha/decisions"
OPENROUTER_MODEL = "typesafe/jev-1.13"


def load_dotenv() -> Dict[str, str]:
    """Read .env.local from the project root; environment wins over the file."""
    env: Dict[str, str] = {}
    p = Path(__file__).resolve().parent.parent / ".env.local"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "JEV_BASE_URL", "OPENROUTER_BASE_URL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def noul(name: str, instructions: str, true_desc: str | None = None,
         false_desc: str | None = None) -> Dict[str, Any]:
    """Yes/no question -> probability in [0,1]. Returns (name, body)."""
    body: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true_desc is not None or false_desc is not None:
        body["criteria"] = {"true": true_desc, "false": false_desc}
    return {name: body}


def choice(name: str, instructions: str, options: Sequence[str] | Dict[str, Any]) -> Dict[str, Any]:
    """Pick one option -> choice + probabilities + confidence."""
    criteria = {o: None for o in options} if not isinstance(options, dict) else dict(options)
    return {name: {"type": "choice", "instructions": instructions, "criteria": criteria}}


def score(name: str, instructions: str, levels: Sequence[Any]) -> Dict[str, Any]:
    """Rate on ordered levels -> score + per-level probabilities."""
    return {name: {"type": "score", "instructions": instructions, "criteria": list(levels)}}


def merge(*questions: Dict[str, Any]) -> Dict[str, Any]:
    """Combine question dicts into the single `questions` object."""
    out: Dict[str, Any] = {}
    for q in questions:
        out.update(q)
    return out


@dataclass
class Response:
    ok: bool
    status: int
    latency_s: float
    body: Any = None
    error: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_tokens(self) -> Optional[int]:
        """Billed input tokens: usage.input_tokens per the pinned schema.

        Other spellings are still probed in case the hosted API differs from
        jeff; returns None rather than guessing a number.
        """
        for container in (self.usage, self.body if isinstance(self.body, dict) else {}):
            if not isinstance(container, dict):
                continue
            for k in ("input_tokens", "prompt_tokens", "tokens", "total_tokens", "billed_tokens"):
                v = container.get(k)
                if isinstance(v, int):
                    return v
            u = container.get("usage")
            if isinstance(u, dict):
                for k in ("input_tokens", "prompt_tokens", "tokens", "total_tokens"):
                    v = u.get(k)
                    if isinstance(v, int):
                        return v
        return None

    @property
    def answers(self) -> Dict[str, Any]:
        """The per-question answers, keyed by the name given in the request."""
        if isinstance(self.body, dict) and isinstance(self.body.get("answers"), dict):
            return self.body["answers"]
        return {}

    @property
    def cost(self) -> Optional[float]:
        """Billed USD, reported by OpenRouter; absent on the TypeSafe-direct API."""
        for container in (self.usage, self.body if isinstance(self.body, dict) else {}):
            if isinstance(container, dict):
                v = container.get("cost")
                if isinstance(v, (int, float)):
                    return float(v)
                u = container.get("usage")
                if isinstance(u, dict) and isinstance(u.get("cost"), (int, float)):
                    return float(u["cost"])
        return None


class JevClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        dry_run: bool = False,
        max_rpm: Optional[int] = 900,
        max_retries: int = 5,
    ):
        env = load_dotenv()
        self.base_url = base_url.rstrip("/")
        self.via_openrouter = OPENROUTER_HOST in self.base_url
        self.endpoint = OPENROUTER_ENDPOINT if self.via_openrouter else ENDPOINT
        self.is_local = not self.via_openrouter and "typesafe.ai" not in self.base_url

        if self.via_openrouter:
            self.api_key = api_key or env.get("OPENROUTER_API_KEY", "")
            # jev-latest exists only on the TypeSafe-direct API.
            self.model = OPENROUTER_MODEL if model in (None, DEFAULT_MODEL) else model
        else:
            self.api_key = api_key or env.get("TYPESAFE_API_KEY", "")
            self.model = model

        self.timeout = timeout
        self.dry_run = dry_run
        self._session = requests.Session()
        self.calls = 0
        self.spent = 0.0
        self.retries = 0
        self.throttled_s = 0.0

        # The documented limits are 1,200 requests/minute and 250k tokens/s, and
        # they are described as moving with demand. A scan issues thousands of
        # requests, so hitting them is the normal case, not the exception:
        # self-throttle, and back off on the 429 that arrives anyway.
        self.max_rpm = max_rpm
        self.max_retries = max_retries
        self._min_interval = 60.0 / max_rpm if max_rpm else 0.0
        self._last_sent = 0.0

        if not self.dry_run and not self.api_key and not self.is_local:
            raise SystemExit(
                "No API key. Put TYPESAFE_API_KEY or OPENROUTER_API_KEY in .env.local, "
                "or pass --dry-run to inspect payloads, or --base-url http://localhost:8000 "
                "to run against a local jeff server."
            )

    def ask(
        self,
        state: Any,
        questions: Dict[str, Any] | Sequence[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Response:
        if not isinstance(questions, dict):  # accept a list of single-entry dicts
            questions = merge(*questions)
        payload = {"model": model or self.model, "state": state, "questions": questions}
        if self.dry_run:
            blob = json.dumps(payload, ensure_ascii=False)
            return Response(ok=True, status=0, latency_s=0.0,
                            body={"dry_run": True, "payload_bytes": len(blob),
                                  "n_questions": len(questions)})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # latency_s measures the successful attempt only: time spent sleeping for
        # the rate limiter or backing off belongs in throttled_s, or every latency
        # figure this project reports would silently include queueing.
        attempt = 0
        while True:
            self._throttle()
            t0 = time.time()
            try:
                r = self._session.post(self.base_url + self.endpoint, headers=headers,
                                       json=payload, timeout=self.timeout)
            except Exception as exc:  # network error or timeout
                if attempt >= self.max_retries:
                    return Response(ok=False, status=-1, latency_s=time.time() - t0,
                                    error=repr(exc))
                self._backoff(attempt)
                attempt += 1
                continue
            dt = time.time() - t0

            if r.status_code in (429, 500, 502, 503, 504, 529) and attempt < self.max_retries:
                self._backoff(attempt, retry_after=r.headers.get("Retry-After"))
                attempt += 1
                continue

            try:
                body = r.json()
            except Exception:
                body = r.text[:2000]
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            res = Response(ok=r.status_code == 200, status=r.status_code, latency_s=dt,
                           body=body, usage=usage if isinstance(usage, dict) else {},
                           error=None if r.status_code == 200 else str(body)[:500])
            self.calls += 1
            if res.cost:
                self.spent += res.cost
            return res

    def _throttle(self) -> None:
        """Keep the send rate under max_rpm, so the limit is approached, not hit."""
        if not self._min_interval:
            return
        wait = self._last_sent + self._min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
            self.throttled_s += wait
        self._last_sent = time.time()

    def _backoff(self, attempt: int, retry_after: Optional[str] = None) -> None:
        delay = min(2.0 ** attempt, 30.0)
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)
        self.throttled_s += delay
        self.retries += 1

    def budget_line(self) -> str:
        extra = ""
        if self.retries or self.throttled_s > 0.5:
            extra = f", {self.retries} retries, {self.throttled_s:.1f}s throttled"
        return f"# {self.calls} calls, ${self.spent:.6f} billed this run{extra}"

    def models(self) -> Response:
        if self.dry_run:
            return Response(ok=True, status=0, latency_s=0.0, body={"dry_run": True})
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0 = time.time()
        r = self._session.get(self.base_url + "/v1/models", headers=headers, timeout=self.timeout)
        try:
            body = r.json()
        except Exception:
            body = r.text[:2000]
        return Response(ok=r.status_code == 200, status=r.status_code,
                        latency_s=time.time() - t0, body=body)


def add_common_args(ap) -> None:
    ap.add_argument("--base-url", default=os.environ.get("JEV_BASE_URL", DEFAULT_BASE_URL),
                    help="use http://localhost:8000 for a local jeff server (no key needed)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=None, help="prefer the TYPESAFE_API_KEY env var")
    ap.add_argument("--dry-run", action="store_true", help="build payloads, send nothing")
    ap.add_argument("--out", default=None, help="write raw results as JSON")


def save(path: Optional[str], obj: Any) -> None:
    if not path:
        return
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f"\n# wrote {p}")

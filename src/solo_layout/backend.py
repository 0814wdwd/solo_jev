"""Decision-model clients for AutoTrust JEV and the vllm-jev plugin."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import threading

import requests

JEV_REPOSITORY = "autotrust/JEV-9B"
JEV_REVISION = "b63f651ce8ed64481d3f5e73ecdb05f740042f01"


@dataclass(frozen=True)
class DecisionSpec:
    question: str
    kind: str = "noul"
    options: tuple = (False, True)

    @classmethod
    def create(cls, question, kind="noul", options=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if kind == "noul":
            if options is not None:
                raise ValueError("noul has fixed false/true options")
            labels = (False, True)
        elif kind == "score":
            if options is not None:
                raise ValueError("score has fixed options 0 through 5")
            labels = tuple(range(6))
        elif kind == "choice":
            if isinstance(options, str) or options is None:
                raise ValueError("choice requires a sequence of 2 to 16 strings")
            labels = tuple(options)
            if not 2 <= len(labels) <= 16 or any(not isinstance(o, str) or not o.strip() for o in labels):
                raise ValueError("choice requires 2 to 16 nonempty strings")
            if len(set(labels)) != len(labels) or any("\n" in o or "\r" in o for o in labels):
                raise ValueError("choice options must be unique single-line strings")
        else:
            raise ValueError("kind must be noul, choice, or score")
        return cls(question, kind, labels)

    def prompt(self, state):
        if self.kind == "choice":
            lines = [f"{chr(65+i)}) {o}" for i, o in enumerate(self.options)]
        elif self.kind == "noul":
            lines = ["false", "true"]
        else:
            lines = [str(o) for o in self.options]
        return (f"[kind] {self.kind}\n[state] {state}\n[question] {self.question}\n[options]\n"
                + "\n".join(lines) + "\n[decision]:")


@dataclass(frozen=True)
class DecisionResponse:
    probabilities: tuple[float, ...]
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    created_cache_tokens: int | None = None
    queue_time_ms: float | None = None
    time_to_first_token_ms: float | None = None
    generation_time_ms: float | None = None
    request_id: str | None = None


def _optional_count(mapping, key):
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"vLLM returned an invalid {key}")
    return value


def _optional_milliseconds(mapping, key):
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise RuntimeError(f"vLLM returned an invalid {key}")
    return float(value)


def _optional_count_sum(mapping, key):
    """Validate and total vllm-jev's per-candidate token counters."""
    values = mapping.get(key)
    if values is None:
        return None
    if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values):
        raise RuntimeError(f"vllm-jev returned invalid {key}")
    return sum(values)


class JevBackend:
    """Thread-safe connections; no GPU library or model weights on the client.

    With model_dir, read just decision_head.json and calibration.json locally.
    Otherwise download those two files at a pinned revision via the hub extra.
    """
    supports_cache_salt = True

    def __init__(self, base_url="http://127.0.0.1:8000", *, model="jev-decision",
                 model_dir=None, api_key=None, timeout=180):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        root = base_url.rstrip("/")
        self.endpoint = root + ("/completions" if root.endswith("/v1") else "/v1/completions")
        self.model, self.model_dir = model, Path(model_dir) if model_dir is not None else None
        self.api_key, self.timeout = api_key, timeout
        self._head = self._temperatures = None
        self._local = threading.local()
        self._sessions = []
        self._lock = threading.Lock()

    def prepare(self, spec):
        if self._head is not None:
            return
        with self._lock:
            if self._head is not None:
                return
            paths = ("adapter_vllm/decision_head.json", "calibration.json")
            if self.model_dir is not None:
                head_path, temperature_path = [self.model_dir / p for p in paths]
            else:
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError as exc:
                    raise ImportError("Pass model_dir or install solo-layout[hub] for JEV metadata") from exc
                head_path, temperature_path = [Path(hf_hub_download(
                    JEV_REPOSITORY, p, revision=JEV_REVISION)) for p in paths]
            head = json.loads(head_path.read_text())
            temperatures = json.loads(temperature_path.read_text())["per_kind"]
            if head["slots"]["template_version"] != "bare-v1":
                raise ValueError("only the bare-v1 JEV decision template is supported")
            for kind, expected in (("noul", 2), ("score", 6), ("choice", 16)):
                begin, end = head["slots"]["ranges"][kind]
                if end - begin != expected or end > len(head["verbalizer_ids"]) or end > len(head["bias"]):
                    raise ValueError("invalid decision-head slot metadata")
                if not math.isfinite(temperatures[kind]) or temperatures[kind] <= 0:
                    raise ValueError("invalid calibration temperature")
            self._temperatures, self._head = temperatures, head

    def _session(self):
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.trust_env = False
            if self.api_key:
                session.headers["Authorization"] = "Bearer " + self.api_key
            self._local.session = session
            with self._lock:
                self._sessions.append(session)
        return self._local.session

    def decide(self, state, spec, *, cache_salt=None):
        self.prepare(spec)
        start = self._head["slots"]["ranges"][spec.kind][0]
        ids = self._head["verbalizer_ids"][start:start + len(spec.options)]
        payload = {
            "model": self.model, "prompt": spec.prompt(state), "max_tokens": 1,
            "temperature": 1.0, "logprobs": len(ids), "allowed_token_ids": ids,
            "add_special_tokens": False, "return_tokens_as_token_ids": True,
        }
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt
        response = self._session().post(self.endpoint, json=payload, timeout=(10, self.timeout))
        response.raise_for_status()
        result = response.json()
        top = result["choices"][0]["logprobs"]["top_logprobs"][0]
        try:
            scores = [(top[f"token_id:{token}"] + self._head["bias"][start+i])
                      / self._temperatures[spec.kind] for i, token in enumerate(ids)]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("vLLM did not return every required decision-token logprob") from exc
        if not all(math.isfinite(v) for v in scores):
            raise RuntimeError("non-finite decision scores")
        weights = [math.exp(v - max(scores)) for v in scores]
        total = sum(weights)
        usage = result.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        metrics = result.get("metrics") or {}
        request_id = result.get("id")
        if request_id is not None and not isinstance(request_id, str):
            raise RuntimeError("vLLM returned an invalid request id")
        return DecisionResponse(
            probabilities=tuple(v / total for v in weights),
            prompt_tokens=_optional_count(usage, "prompt_tokens"),
            cached_tokens=_optional_count(details, "cached_tokens"),
            completion_tokens=_optional_count(usage, "completion_tokens"),
            created_cache_tokens=_optional_count(details, "created_cache_tokens"),
            queue_time_ms=_optional_milliseconds(metrics, "queue_time_ms"),
            time_to_first_token_ms=_optional_milliseconds(metrics, "time_to_first_token_ms"),
            generation_time_ms=_optional_milliseconds(metrics, "generation_time_ms"),
            request_id=request_id,
        )

    def close(self):
        with self._lock:
            for session in self._sessions:
                session.close()
            self._sessions.clear()


class VllmJevBackend:
    """Open-Jev decisions through vllm-jev's prefix-cache-aware Choice API.

    The plugin route accepts an explicit cache salt, unlike its System One
    compatibility route. Each SOLO output mode is represented as an ordered
    candidate list, and the returned probabilities are mapped back to the
    corresponding :class:`DecisionSpec` options.
    """
    supports_cache_salt = True

    def __init__(self, base_url="http://127.0.0.1:8795", *, api_key=None,
                 timeout=180, temperature=None, use_prefix_cache=True):
        if (isinstance(timeout, bool) or not isinstance(timeout, Real)
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be positive and finite")
        if (temperature is not None
                and (isinstance(temperature, bool) or not isinstance(temperature, Real)
                     or not math.isfinite(temperature) or temperature <= 0)):
            raise ValueError("temperature must be positive and finite")
        if not isinstance(use_prefix_cache, bool):
            raise ValueError("use_prefix_cache must be a boolean")
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.endpoint = root + "/plugins/vllm-jev/choice"
        self.api_key = api_key
        self.timeout = float(timeout)
        self.temperature = float(temperature) if temperature is not None else None
        self.use_prefix_cache = use_prefix_cache
        self._local = threading.local()
        self._sessions = []
        self._lock = threading.Lock()

    def prepare(self, spec):
        """No client-side model metadata is required by vllm-jev."""

    def _session(self):
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.trust_env = False
            if self.api_key:
                session.headers["Authorization"] = "Bearer " + self.api_key
            self._local.session = session
            with self._lock:
                self._sessions.append(session)
        return self._local.session

    @staticmethod
    def _labels(spec):
        if spec.kind == "noul":
            return ("no", "yes")
        return tuple(str(option) for option in spec.options)

    def decide(self, state, spec, *, cache_salt=None):
        if cache_salt is not None and not isinstance(cache_salt, str):
            raise ValueError("cache_salt must be a string or None")
        labels = self._labels(spec)
        payload = {
            # Keep the serialized record as text. Parsing it back to JSON could
            # change the field order that SOLO selected for prefix reuse.
            "state": state,
            "question": spec.question,
            "options": list(labels),
            "use_prefix_cache": self.use_prefix_cache,
        }
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        response = self._session().post(
            self.endpoint, json=payload, timeout=(10, self.timeout))
        response.raise_for_status()
        result = response.json()
        probabilities = result.get("probabilities")
        if not isinstance(probabilities, dict):
            raise RuntimeError("vllm-jev did not return candidate probabilities")
        try:
            values = [probabilities[label] for label in labels]
        except KeyError as exc:
            raise RuntimeError("vllm-jev did not return every candidate probability") from exc
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) or value < 0 for value in values):
            raise RuntimeError("vllm-jev returned invalid candidate probabilities")
        total = float(sum(values))
        if total <= 0:
            raise RuntimeError("vllm-jev returned zero probability mass")
        request_id = (response.headers.get("x-request-id")
                      or response.headers.get("x-typesafe-request-id"))
        return DecisionResponse(
            probabilities=tuple(float(value) / total for value in values),
            prompt_tokens=_optional_count_sum(result, "prompt_tokens"),
            cached_tokens=_optional_count_sum(result, "cached_tokens"),
            # vllm-jev scores candidates with a pooling head; it does not emit
            # an autoregressive completion token on this route.
            completion_tokens=0,
            request_id=request_id,
        )

    def close(self):
        with self._lock:
            for session in self._sessions:
                session.close()
            self._sessions.clear()

"""JEV's calibrated decision head over the vLLM completions API."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
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
                    raise ImportError("Pass model_dir or install solo-decision[hub] for JEV metadata") from exc
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
        return DecisionResponse(tuple(v / total for v in weights), usage.get("prompt_tokens"),
                                (usage.get("prompt_tokens_details") or {}).get("cached_tokens"))

    def close(self):
        with self._lock:
            for session in self._sessions:
                session.close()
            self._sessions.clear()


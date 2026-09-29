#!/usr/bin/env python3
"""Minimal smoke test: is Jev reachable through OpenRouter, and in what shape?

Two unknowns at once. (a) Does OpenRouter actually serve typesafe/jev, and under
which id. (b) What request shape does its Decisions API take -- the shape quoted
to us uses a LIST of questions with `key`/`options`, while Jev's own SDK schema
(pinned from jeff) uses an OBJECT keyed by name with `instructions`/`criteria`.
Those are incompatible, so we try both and let the server decide.

Reads the key from .env.local. Never prints it.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> dict:
    env = {}
    p = ROOT / ".env.local"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    for k in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def show(label: str, r: requests.Response, t: float) -> None:
    print(f"\n--- {label}: HTTP {r.status_code} in {t*1000:.0f}ms")
    try:
        body = r.json()
        print(json.dumps(body, ensure_ascii=False, indent=2)[:1400])
    except Exception:
        print(r.text[:800])


def main() -> None:
    env = load_env()
    key = env.get("OPENROUTER_API_KEY")
    base = env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api").rstrip("/")
    if not key:
        sys.exit("no OPENROUTER_API_KEY in .env.local")
    H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    print(f"# base={base}  key=***{key[-4:]} (len {len(key)})")

    # 0. Does the key work at all, and what does it have access to?
    t0 = time.time()
    r = requests.get(f"{base}/v1/key", headers=H, timeout=30)
    show("GET /v1/key", r, time.time() - t0)

    # 1. Is typesafe/jev in the catalogue?
    t0 = time.time()
    r = requests.get(f"{base}/v1/models", headers=H, timeout=60)
    dt = time.time() - t0
    print(f"\n--- GET /v1/models: HTTP {r.status_code} in {dt*1000:.0f}ms")
    jev_ids = []
    if r.status_code == 200:
        try:
            data = r.json().get("data", [])
            print(f"    {len(data)} models total")
            for m in data:
                mid = m.get("id", "")
                if "typesafe" in mid.lower() or "jev" in mid.lower():
                    jev_ids.append(mid)
                    print(f"    FOUND {mid}   pricing={m.get('pricing')}"
                          f"   ctx={m.get('context_length')}")
            if not jev_ids:
                print("    no typesafe/jev entry in /v1/models "
                      "(an alpha endpoint may not be listed here)")
        except Exception as exc:
            print(f"    parse error: {exc}")
    else:
        print(r.text[:500])

    # 2. The Decisions API, both candidate request shapes.
    state = "The customer says the product arrived broken."
    shapes = {
        "list_key_options (as quoted)": {
            "questions": [
                {"type": "choice", "key": "category",
                 "options": ["refund", "technical_support", "general_question"]}
            ]
        },
        "object_instructions_criteria (Jev SDK schema)": {
            "questions": {
                "category": {"type": "choice",
                             "instructions": "Which queue should this go to?",
                             "criteria": {"refund": None, "technical_support": None,
                                          "general_question": None}}
            }
        },
    }
    for model in ("typesafe/jev-1.13", "typesafe/jev-latest"):
        for label, extra in shapes.items():
            payload = {"model": model, "state": state, **extra}
            t0 = time.time()
            try:
                r = requests.post(f"{base}/alpha/decisions", headers=H,
                                  json=payload, timeout=60)
            except Exception as exc:
                print(f"\n--- POST /alpha/decisions {model} [{label}]: {exc!r}")
                continue
            show(f"POST /alpha/decisions  model={model}  shape={label}", r, time.time() - t0)
            if r.status_code == 200:
                print("    ^^ this combination works")


if __name__ == "__main__":
    main()

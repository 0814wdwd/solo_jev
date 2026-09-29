#!/usr/bin/env python3
"""Measure the fixed per-request billed overhead directly.

probe_tokenize's calibration puts the intercept at ~424 tokens, but that is an
extrapolation from states of 311..24251 proxy tokens, so its uncertainty at x=0
is large. The number matters a lot: it multiplies by request count, and the
per-row baseline issues one request per row, so the overhead alone can dominate
the comparison. So measure it near zero instead of extrapolating to it.

Also separates the two components of that overhead:
  - per-request scaffolding (state-independent)
  - per-question cost, by question type and option count

~14 tiny calls, a fraction of a cent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, choice, merge, noul, score  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

BASE = "https://openrouter.ai/api"


def main() -> None:
    c = JevClient(base_url=BASE)
    counter = get_counter("cl100k_base")

    print("# 1. near-zero states, one minimal noul: billed - 1.31*proxy = overhead")
    q1 = noul("q", "True.")
    print(f"{'state':>28s} {'proxy':>6s} {'billed':>7s} {'implied_overhead':>17s}")
    for label, state in (("'x'", "x"),
                         ("'ab cd ef'", "ab cd ef"),
                         ("one 8-col row", "r1: a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8"),
                         ("two 8-col rows", "r1: a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8\n"
                                            "r2: a=9,b=8,c=7,d=6,e=5,f=4,g=3,h=2")):
        r = c.ask(state, q1)
        if not r.ok:
            print(f"{label:>28s}  HTTP {r.status} {str(r.error)[:120]}")
            continue
        proxy = counter(state)
        print(f"{label:>28s} {proxy:6d} {r.input_tokens:7d} "
              f"{r.input_tokens - 1.3099 * proxy:17.1f}")

    print("\n# 2. cost of the question itself, state held at a single character")
    cases = {
        "noul 'True.'": q1,
        "noul long text": noul("q", "Every field in this row is internally consistent "
                                    "with every other field, including dates."),
        "choice 2 opts": choice("q", "Pick.", ["a", "b"]),
        "choice 8 opts": choice("q", "Pick.", [f"opt{i}" for i in range(8)]),
        "score 3 levels": score("q", "Rate.", ["low", "mid", "high"]),
        "score 8 levels": score("q", "Rate.", [f"l{i}" for i in range(8)]),
    }
    print(f"{'question':>20s} {'billed':>7s}")
    for label, q in cases.items():
        r = c.ask("x", q)
        print(f"{label:>20s} {r.input_tokens if r.ok else -1:7d}")

    print("\n# 3. marginal cost per added noul, state held at a single character")
    print(f"{'n_nouls':>8s} {'billed':>7s} {'marginal':>9s}")
    prev = None
    for n in (1, 2, 4, 8, 16, 32):
        qs = merge(*[noul(f"q{i}", "True.") for i in range(n)])
        r = c.ask("x", qs)
        if not r.ok:
            print(f"{n:8d}  HTTP {r.status} {str(r.error)[:120]}")
            continue
        marg = "" if prev is None else f"{(r.input_tokens - prev[1]) / (n - prev[0]):9.2f}"
        print(f"{n:8d} {r.input_tokens:7d} {marg:>9s}")
        prev = (n, r.input_tokens)

    print(f"\n{c.budget_line()}")


if __name__ == "__main__":
    main()

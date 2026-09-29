#!/usr/bin/env python3
"""Was that 31s a cold start, and what is warm latency actually like?

TypeSafe claims 70-500ms end to end. The first successful call through
OpenRouter took 31.1s. Before sizing any other probe we need to know which
number describes steady state, because it decides whether a few hundred requests
is minutes or hours.

Also confirms the billing identity cost == input_tokens/1e6 * $0.042 across
several payload sizes, and reports spend so the $3 key budget stays visible.

Cheap by construction: ~12 tiny calls, well under a cent.
"""
from __future__ import annotations

import statistics as stats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from or_client import POSTED_INPUT_PRICE_PER_MTOK, OpenRouterJev, choice, noul  # noqa: E402

SHORT = "The customer says the product arrived broken."
LONG = " ".join(
    f"r{i}: Year=2023 | Month={i % 12 + 1} | Origin=BDL | Dest=LGA | DepDelay={i} | ArrDelay={i + 3}"
    for i in range(40)
)


def main() -> None:
    c = OpenRouterJev()
    q = choice("category", "Which queue should this go to?",
               ["refund", "technical_support", "general_question"])

    print("# warm-up sequence: 6 identical short calls back to back")
    lats = []
    for i in range(6):
        r = c.ask(SHORT, q)
        if not r.ok:
            print(f"  call {i}: HTTP {r.status} {str(r.error)[:160]}")
            continue
        lats.append(r.latency_s)
        print(f"  call {i}: {r.latency_s*1000:8.0f}ms  in_tok={r.input_tokens:5d}"
              f"  out_tok={r.output_tokens}  cost=${r.cost:.3e}")
    if len(lats) >= 3:
        print(f"\n  first={lats[0]*1000:.0f}ms   median_of_rest="
              f"{stats.median(lats[1:])*1000:.0f}ms   min={min(lats)*1000:.0f}ms")
        if lats[0] > 3 * max(stats.median(lats[1:]), 0.001):
            print("  -> the first call was a cold start; steady state is the median")
        else:
            print("  -> no cold-start effect; this latency is steady state")

    print("\n# billing identity: does cost == input_tokens/1e6 * $0.042 ?")
    print(f"{'state':>8s} {'in_tok':>7s} {'cost':>12s} {'implied $/Mtok':>15s} {'latency_ms':>11s}")
    for label, state, qs in (
        ("short", SHORT, q),
        ("short+3q", SHORT, {**q, **noul("urgent", "Is this urgent?"),
                             **noul("angry", "Is the customer angry?")}),
        ("long40", LONG, q),
    ):
        r = c.ask(state, qs)
        if not r.ok:
            print(f"{label:>8s}  HTTP {r.status} {str(r.error)[:140]}")
            continue
        implied = r.cost / (r.input_tokens / 1e6) if r.input_tokens else float("nan")
        print(f"{label:>8s} {r.input_tokens:7d} {r.cost:12.3e} {implied:15.4f} "
              f"{r.latency_s*1000:11.0f}")
    print(f"\n# posted price is ${POSTED_INPUT_PRICE_PER_MTOK}/Mtok input, output free")
    print(c.budget_line())


if __name__ == "__main__":
    main()

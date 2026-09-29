#!/usr/bin/env python3
"""Probe 3: where does question-level parallelism stop being free?

TypeSafe documents that every question is evaluated in parallel against one
state, and a third party measured a 13-question request at 12.2x cheaper and 10x
faster than 13 separate calls. Packing a table into few large requests only works
if that flatness holds at the question counts a table scan produces -- hundreds,
not thirteen. This sweeps question count against a fixed state and finds the knee.

It also measures the thing the packer actually needs: cost and latency per
decision as a function of block size, including the point where the 64k total
budget (state + all questions) starts forcing smaller blocks.

Usage:
  TYPESAFE_API_KEY=... python3 probes/probe_parallel.py --max-q 256 --out results/par.json
  python3 probes/probe_parallel.py --dry-run
"""
from __future__ import annotations

import argparse
import statistics as stats
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, add_common_args, noul, save  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_cache import make_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--state-rows", type=int, default=200)
    ap.add_argument("--max-q", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3, help="repeats per question count")
    ap.add_argument("--tokenizer", default="cl100k_base")
    args = ap.parse_args()

    client = JevClient(api_key=args.api_key, base_url=args.base_url,
                       model=args.model, dry_run=args.dry_run)
    counter = get_counter(args.tokenizer)
    state = "\n".join(make_rows(args.state_rows, seed=11))
    state_tok = counter(state)

    counts: List[int] = []
    k = 1
    while k <= args.max_q:
        counts.append(k)
        k *= 2
    if args.max_q not in counts:
        counts.append(args.max_q)

    print(f"# probe_parallel  state_rows={args.state_rows} (~{state_tok} proxy tokens)"
          f"  base_url={args.base_url}")
    print(f"\n{'n_questions':>11s} {'median_ms':>10s} {'min_ms':>8s} {'billed_tok':>11s} "
          f"{'ms_per_decision':>16s} {'ok':>4s}")

    records: List[Dict[str, Any]] = []
    for n in counts:
        qs = [noul(f"q{i}", f"Row r{(i % args.state_rows) + 1} has a plausible departure delay.")
              for i in range(n)]
        lats, toks, oks = [], [], 0
        for rep in range(args.reps):
            r = client.ask(state, qs)
            records.append({"n_questions": n, "rep": rep, "ok": r.ok, "status": r.status,
                            "latency_s": r.latency_s, "billed_tokens": r.input_tokens,
                            "proxy_state_tokens": state_tok, "error": r.error})
            if r.ok:
                oks += 1
                lats.append(r.latency_s)
                if r.input_tokens is not None:
                    toks.append(r.input_tokens)
            elif not args.dry_run:
                print(f"  !! n={n} rep={rep} status={r.status} {str(r.error)[:140]}")
                if r.status in (400, 413, 422):
                    print(f"  -> rejected at n={n}; this is the hard question-count ceiling")
            time.sleep(0.2)
        if lats:
            med = stats.median(lats)
            print(f"{n:11d} {med*1000:10.1f} {min(lats)*1000:8.1f} "
                  f"{(stats.median(toks) if toks else -1):11.0f} "
                  f"{med*1000/n:16.2f} {oks:4d}")
        elif args.dry_run:
            print(f"{n:11d} {'(dry)':>10s} {'-':>8s} {'-':>11s} {'-':>16s} {oks:4d}")

    good = [r for r in records if r.get("ok") and r.get("latency_s")]
    if good and not args.dry_run:
        by_n: Dict[int, List[float]] = {}
        for r in good:
            by_n.setdefault(r["n_questions"], []).append(r["latency_s"])
        ns = sorted(by_n)
        base = stats.median(by_n[ns[0]])
        print("\n# flatness: latency relative to a single question")
        knee = None
        for n in ns:
            ratio = stats.median(by_n[n]) / base
            flag = ""
            if ratio > 2.0 and knee is None:
                knee = n
                flag = "  <- knee: parallelism no longer free"
            print(f"  n={n:5d}  x{ratio:6.2f}{flag}")
        if knee is None:
            print(f"  latency stayed within 2x up to n={ns[-1]}: "
                  f"pack as many questions as the 64k budget allows")

    save(args.out, {"args": vars(args) | {"api_key": None}, "records": records})


if __name__ == "__main__":
    main()

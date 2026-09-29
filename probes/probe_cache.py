#!/usr/bin/env python3
"""Probe 1: is there an undocumented cross-request cache?

This is the decisive experiment for the whole project. SOLO's mechanism is
cross-request prefix KV reuse. TypeSafe's docs describe no cache of any kind, so
if none exists, reordering can only pay through token count (see
jev_solo/bench_offline.py) and never through cache hits. If an undocumented
prefix cache does exist, SOLO transfers almost literally and the project changes
shape.

Four conditions, same state size throughout:
  repeat  -- identical state every time        (any cache should hit)
  suffix  -- same long prefix, last row edited (a *prefix* cache should hit)
  prefix  -- first row edited, rest identical  (a prefix cache should MISS)
  fresh   -- fully new state each time         (nothing can hit)

Read latency and, if the API reports them, billed input tokens. A prefix cache
shows up as: repeat ~ suffix << prefix ~ fresh. A whole-request (exact-match)
cache shows up as: repeat << suffix ~ prefix ~ fresh.

Usage:
  TYPESAFE_API_KEY=... python3 probes/probe_cache.py --reps 8 --state-rows 60
  python3 probes/probe_cache.py --dry-run                      # inspect payloads
  python3 probes/probe_cache.py --base-url http://localhost:8000  # against jeff
"""
from __future__ import annotations

import argparse
import random
import statistics as stats
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, add_common_args, noul, save  # noqa: E402


def make_rows(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    airports = ["BDL", "LGA", "ORD", "SFO", "LAX", "ATL", "DEN", "SEA", "JFK", "BOS"]
    out = []
    for i in range(n):
        out.append(
            f"r{i+1}: Year=2023 | Month={rng.randint(1,12)} | Origin={rng.choice(airports)} "
            f"| Dest={rng.choice(airports)} | DepDelay={rng.randint(-10,180)} "
            f"| ArrDelay={rng.randint(-10,200)} | Carrier={rng.choice(['AA','DL','UA','WN','9E'])} "
            f"| TailNum=N{rng.randint(100,999)}{rng.choice('ABCDEFG')}{rng.choice('XYZ')}"
        )
    return out


def variant(base: List[str], kind: str, rep: int) -> str:
    rows = list(base)
    if kind == "repeat":
        pass
    elif kind == "suffix":
        rows[-1] = rows[-1].replace("DepDelay=", f"DepDelay={rep}#", 1)
    elif kind == "prefix":
        rows[0] = rows[0].replace("DepDelay=", f"DepDelay={rep}#", 1)
    elif kind == "fresh":
        rows = make_rows(len(base), seed=10_000 + rep)
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--reps", type=int, default=8, help="requests per condition")
    ap.add_argument("--state-rows", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between requests")
    ap.add_argument("--ttl-gap", type=float, default=0.0,
                    help="if >0, repeat the 'repeat' condition once after this many seconds (TTL check)")
    args = ap.parse_args()

    client = JevClient(api_key=args.api_key, base_url=args.base_url,
                       model=args.model, dry_run=args.dry_run)
    base = make_rows(args.state_rows, seed=7)
    questions = [noul("q1", "Every row has an arrival delay consistent with its departure delay.")]

    print(f"# probe_cache  base_url={args.base_url}  model={args.model}"
          f"  state_rows={args.state_rows}  reps={args.reps}")
    if args.dry_run:
        print("# DRY RUN: nothing is sent")

    records: List[Dict] = []
    for kind in ("repeat", "suffix", "prefix", "fresh"):
        lats, toks = [], []
        for rep in range(args.reps):
            state = variant(base, kind, rep)
            r = client.ask(state, questions)
            records.append({"condition": kind, "rep": rep, "ok": r.ok, "status": r.status,
                            "latency_s": r.latency_s, "input_tokens": r.input_tokens,
                            "error": r.error, "state_chars": len(state)})
            if r.ok:
                lats.append(r.latency_s)
                if r.input_tokens is not None:
                    toks.append(r.input_tokens)
            elif not args.dry_run:
                print(f"  !! {kind} rep{rep}: status={r.status} {str(r.error)[:160]}")
            if args.sleep:
                time.sleep(args.sleep)
        if lats and not args.dry_run:
            print(f"  {kind:7s} n={len(lats)}  median={stats.median(lats)*1000:7.1f}ms"
                  f"  min={min(lats)*1000:7.1f}ms"
                  f"  first={lats[0]*1000:7.1f}ms"
                  f"  billed_tokens={stats.median(toks) if toks else 'not reported'}")
        elif args.dry_run:
            print(f"  {kind:7s} payload built, {len(variant(base, kind, 0))} state chars")

    if args.ttl_gap > 0 and not args.dry_run:
        print(f"\n# TTL check: sleeping {args.ttl_gap}s, then repeating the identical state")
        time.sleep(args.ttl_gap)
        r = client.ask(variant(base, "repeat", 0), questions)
        records.append({"condition": "repeat_after_gap", "rep": 0, "ok": r.ok,
                        "latency_s": r.latency_s, "input_tokens": r.input_tokens})
        print(f"  repeat_after_gap latency={r.latency_s*1000:.1f}ms  tokens={r.input_tokens}")

    if not args.dry_run:
        lat: Dict[str, List[float]] = {}
        tok: Dict[str, List[int]] = {}
        for rec in records:
            if not rec.get("ok"):
                continue
            if rec.get("latency_s"):
                lat.setdefault(rec["condition"], []).append(rec["latency_s"])
            if rec.get("input_tokens") is not None:
                tok.setdefault(rec["condition"], []).append(rec["input_tokens"])

        print("\n# verdict")
        verdict = "inconclusive"
        # Billed tokens are the primary signal: a credited cache shows up in the
        # bill, whereas latency on a shared multi-tenant endpoint is noisy enough
        # to hide a real hit (the mock server reproduces exactly that case).
        if {"repeat", "fresh"} <= tok.keys():
            rep_t = stats.median(tok["repeat"])
            fresh_t = stats.median(tok["fresh"])
            ratio_t = rep_t / fresh_t if fresh_t else float("nan")
            print(f"  billed tokens: repeat={rep_t:.0f} fresh={fresh_t:.0f} ratio={ratio_t:.3f}")
            if ratio_t < 0.9:
                suf = stats.median(tok.get("suffix", [fresh_t]))
                pre = stats.median(tok.get("prefix", [fresh_t]))
                print(f"                 suffix={suf:.0f} prefix={pre:.0f}")
                if suf / fresh_t < 0.9 and pre / fresh_t > 0.9:
                    verdict = "prefix cache, credited in the bill"
                elif suf / fresh_t > 0.9 and pre / fresh_t > 0.9:
                    verdict = "exact-match cache, credited in the bill"
                else:
                    verdict = "cache credited in the bill, shape unclear"
            else:
                verdict = "no cache credited in the bill"
        else:
            print("  billed tokens: not reported by this endpoint")

        if {"repeat", "fresh"} <= lat.keys():
            rep_m, fresh_m = stats.median(lat["repeat"]), stats.median(lat["fresh"])
            ratio = rep_m / fresh_m if fresh_m else float("nan")
            print(f"  latency: repeat/fresh = {ratio:.2f}"
                  f"  (repeat={rep_m*1000:.0f}ms fresh={fresh_m*1000:.0f}ms)")
            if verdict.startswith("no cache") and ratio < 0.85:
                verdict = ("latency suggests warming but the bill shows no credit "
                           "-- treat as no usable cache")
            elif verdict == "inconclusive" and ratio < 0.85:
                suf = stats.median(lat.get("suffix", [fresh_m]))
                pre = stats.median(lat.get("prefix", [fresh_m]))
                shape = "prefix-shaped" if suf < pre * 0.85 else "exact-match-shaped"
                verdict = f"latency-only evidence of a cache, {shape} (weak)"

        print(f"  => {verdict}")
        print("  Note: latency alone is weak here. On a shared endpoint a real "
              "prefix-cache hit can sit inside the noise; the bill cannot.")

    save(args.out, {"args": vars(args) | {"api_key": None}, "records": records})


if __name__ == "__main__":
    main()

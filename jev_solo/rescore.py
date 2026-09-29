#!/usr/bin/env python3
"""Re-score the offline benchmark under the measured billing model.

bench_offline.py counts proxy (cl100k) tokens. The probes measured what Jev
actually bills:

    billed  ~  overhead_per_request  +  slope_enc * proxy_state_tokens
                                     +  per_question * n_questions

with, from probes/probe_overhead.py:
    overhead_per_request = 260      (state "x", one minimal noul, minus the noul)
    per_question         = 9.00     (exactly linear over n = 1..32 nouls)

and slope_enc fitted per encoding from probes/probe_tokenize.py, because the
character mix differs by encoding and so does tokenizer efficiency.

The overhead term is the point: it is charged per *request*, and the per-row
baseline issues one request per row. Request count, not just token count, is part
of the objective -- which bench_offline.py's proxy counting could not see.

Usage:
  python3 -m jev_solo.rescore --bench results/flight_5k_all.json \
                              --tokenize results/or_tokenize.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

OVERHEAD_PER_REQUEST = 260.0
PER_QUESTION = 9.00
PRICE_PER_MTOK = 0.042


def fit_slopes(tokenize_json: Path, overhead: float) -> Dict[str, Tuple[float, float, int]]:
    """Per-encoding slope of billed vs proxy, with the intercept held at `overhead`.

    Returns {encoding: (slope, max_abs_resid_frac, n_points)}.
    """
    recs = json.loads(tokenize_json.read_text())["records"]
    by_enc: Dict[str, List[Tuple[float, float]]] = {}
    for r in recs:
        if r.get("phase") != "calibration" or not r.get("billed_tokens"):
            continue
        by_enc.setdefault(r["encoding"], []).append((float(r["proxy_tokens"]),
                                                     float(r["billed_tokens"])))
    out = {}
    for enc, pts in by_enc.items():
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts]) - overhead - PER_QUESTION  # one noul in that probe
        slope = float((x * y).sum() / (x * x).sum())  # least squares through the origin
        resid = np.abs(slope * x - y) / np.maximum(y, 1.0)
        out[enc] = (slope, float(resid.max()), len(pts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="results/flight_5k_all.json")
    ap.add_argument("--tokenize", default="results/or_tokenize.json")
    ap.add_argument("--overhead", type=float, default=OVERHEAD_PER_REQUEST)
    ap.add_argument("--per-question", type=float, default=PER_QUESTION)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bench = json.loads(Path(args.bench).read_text())
    slopes = fit_slopes(Path(args.tokenize), args.overhead)
    rows = bench["rows"]
    q_tokens_proxy = bench["q_tokens"]

    print(f"# billed = {args.overhead:.0f}/request + slope_enc * proxy_state "
          f"+ {args.per_question:.2f}/question")
    print(f"# table: {rows} rows x {bench['cols']} cols, one question per row\n")
    print(f"{'encoding':14s} {'slope':>7s} {'max_resid':>10s} {'n':>3s}")
    for enc, (s, resid, n) in sorted(slopes.items()):
        print(f"{enc:14s} {s:7.3f} {resid*100:9.1f}% {n:3d}")

    def billed(state_proxy: float, requests: int, n_questions: int, enc: str) -> float:
        slope = slopes.get(enc, (1.31, 0, 0))[0]
        return slope * state_proxy + args.overhead * requests + args.per_question * n_questions

    # Baseline: one request per row, row_kv prompt shape, no row ids.
    base_total_proxy = bench["baseline_per_row"]["tokens"]
    base_state_proxy = base_total_proxy - rows * q_tokens_proxy
    base_billed = billed(base_state_proxy, rows, rows, "row_kv")
    print(f"\n# baseline: {rows} requests, 1 row each")
    print(f"  proxy_state={base_state_proxy:>10.0f}  billed={base_billed:>11.0f}"
          f"  ${base_billed/1e6*PRICE_PER_MTOK:.4f}"
          f"   overhead share={args.overhead*rows/base_billed*100:.1f}%")

    results = []
    for r in bench["results"]:
        enc, plan = r["encoding"], r["planner"]
        b = billed(r["state_tokens"], r["requests"], rows, enc)
        results.append({**r, "billed_tokens": b, "billed_usd": b / 1e6 * PRICE_PER_MTOK,
                        "billed_vs_baseline": b / base_billed,
                        "overhead_share": args.overhead * r["requests"] / b})

    encs = ["row_kv", "csv_block", "csv_rle", "columnar_rle", "factored_rle"]
    plans = ["default", "ndv", "solo_greedy", "token_greedy", "token_greedy_eps"]
    tab = {(r["planner"], r["encoding"]): r for r in results}

    print(f"\n# percent of baseline billed tokens (lower is better)\n")
    print(f"{'encoding':14s}" + "".join(f"{p:>19s}" for p in plans))
    print("-" * (14 + 19 * len(plans)))
    for enc in encs:
        line = f"{enc:14s}"
        for p in plans:
            r = tab.get((p, enc))
            line += f"{r['billed_vs_baseline']*100:13.1f}% /{r['requests']:4d}" if r else " " * 19
        print(line)

    best = min(results, key=lambda r: r["billed_vs_baseline"])
    print(f"\n# best: {best['encoding']} + {best['planner']}")
    print(f"  billed={best['billed_tokens']:.0f} tokens "
          f"({best['billed_vs_baseline']*100:.1f}% of baseline, "
          f"{1/best['billed_vs_baseline']:.2f}x cheaper)")
    print(f"  requests {rows} -> {best['requests']} ({rows/best['requests']:.0f}x fewer)")
    print(f"  ${best['billed_usd']:.4f} vs ${base_billed/1e6*PRICE_PER_MTOK:.4f}")
    print(f"  per-request overhead is {best['overhead_share']*100:.1f}% of the packed cost, "
          f"{args.overhead*rows/base_billed*100:.1f}% of the baseline's")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "model": {"overhead_per_request": args.overhead,
                      "per_question": args.per_question,
                      "slopes": {k: v[0] for k, v in slopes.items()}},
            "baseline_billed": base_billed, "rows": rows, "results": results,
        }, indent=2))
        print(f"\n# wrote {p}")


if __name__ == "__main__":
    main()

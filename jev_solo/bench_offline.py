#!/usr/bin/env python3
"""Offline benchmark: what does a table cost as Jev input tokens?

No API key and no network. It answers the question that decides whether this
whole direction is worth pursuing: once the objective changes from "maximize
prefix KV reuse" to "minimize serialized input tokens", does reordering still
pay, and does SOLO's planner still win?

Usage:
  python3 -m jev_solo.bench_offline --csv /path/table.csv --rows 20000 \
      --planners default,ndv,solo_greedy,token_greedy,token_greedy_eps \
      --out results/flight_20k.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_solo.encodings import ENCODINGS, encode  # noqa: E402
from jev_solo.objective import validate  # noqa: E402
from jev_solo.pack import (  # noqa: E402
    STATE_BUDGET,
    TOTAL_BUDGET,
    pack_requests,
    pack_summary,
    per_row_baseline_tokens,
)
from jev_solo.plan import PLANNERS, plan  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

PRICE_PER_MTOK = 0.042


def load_csv(path: str, rows: int, delimiter: str | None = None) -> tuple[List[str], List[List[str]]]:
    if delimiter is None:
        delimiter = "\t" if "food" in Path(path).name.lower() else ","
    out: List[List[str]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = [h.strip().strip('"') for h in next(reader)]
        width = len(header)
        for row in reader:
            if len(row) != width:
                continue
            out.append(row)
            if rows and len(out) >= rows:
                break
    return header, out


def pack_generic(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    counter,
    encoding: str,
    q_tokens: int,
    chunk: int = 64,
) -> Dict[str, float]:
    """Budget-feasible packing for an arbitrary encoding, by chunked growth.

    Grows a block `chunk` rows at a time, re-encoding to check the budget, then
    backs off to the last feasible checkpoint. Approximate at the chunk
    boundary, exact on the reported token totals.
    """
    i, n = 0, len(rows)
    requests = 0
    total_state = 0
    total_q = 0
    sizes: List[int] = []
    while i < n:
        lo = 0
        last_ok_tokens = None
        k = 0
        while True:
            k_try = min(n - i, k + chunk if k else chunk)
            block = rows[i : i + k_try]
            toks = counter(encode(header, block, encoding, row_ids=True))
            if toks + q_tokens <= STATE_BUDGET and toks + k_try * q_tokens <= TOTAL_BUDGET:
                k = k_try
                last_ok_tokens = toks
                if i + k >= n:
                    break
            else:
                if k == 0:  # even one chunk overflows; shrink linearly
                    for k_lin in range(k_try - 1, 0, -1):
                        block = rows[i : i + k_lin]
                        toks = counter(encode(header, block, encoding, row_ids=True))
                        if toks + q_tokens <= STATE_BUDGET and toks + k_lin * q_tokens <= TOTAL_BUDGET:
                            k = k_lin
                            last_ok_tokens = toks
                            break
                    if k == 0:  # single row does not fit
                        k = 1
                        last_ok_tokens = counter(encode(header, rows[i : i + 1], encoding, row_ids=True))
                break
        del lo
        requests += 1
        sizes.append(k)
        total_state += int(last_ok_tokens or 0)
        total_q += k * q_tokens
        i += k
    return {
        "requests": requests,
        "state_tokens": total_state,
        "question_tokens": total_q,
        "total_tokens": total_state + total_q,
        "rows_per_request_mean": (sum(sizes) / max(1, len(sizes))),
        "rows_per_request_max": max(sizes) if sizes else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--planners", default="default,ndv,solo_greedy,token_greedy,token_greedy_eps")
    ap.add_argument("--encodings", default="row_kv,csv_block,factored_rle")
    ap.add_argument("--tokenizer", default="cl100k_base",
                    help="proxy for Jev's unpublished tokenizer; swap once probe_tokenize.py has run")
    ap.add_argument("--q-tokens", type=int, default=24, help="tokens per typed question")
    ap.add_argument("--request-overhead", type=int, default=0,
                    help="fixed billed tokens per request, independent of state size. "
                         "Measured via probes/probe_tokenize.py (the intercept of the "
                         "billed-vs-proxy fit). This term is what makes request count, "
                         "not just token count, part of the objective.")
    ap.add_argument("--token-scale", type=float, default=1.0,
                    help="multiply proxy tokens by this to approximate billed tokens "
                         "(the slope from probe_tokenize.py's calibration fit)")
    ap.add_argument("--plan-sample", type=int, default=20000)
    ap.add_argument("--validate-rows", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    counter = get_counter(args.tokenizer)
    header, rows = load_csv(args.csv, args.rows)
    print(f"# table: {Path(args.csv).name}  {len(rows)} rows x {len(header)} cols")
    print(f"# tokenizer proxy: {args.tokenizer}   price: ${PRICE_PER_MTOK}/Mtok (output free)")
    if args.request_overhead or args.token_scale != 1.0:
        print(f"# billed model: {args.token_scale} * proxy + {args.request_overhead}/request")

    def billed(proxy_tokens: int, requests: int) -> int:
        return int(round(proxy_tokens * args.token_scale + requests * args.request_overhead))

    base_raw, base_meta = per_row_baseline_tokens(header, rows, counter, q_tokens=args.q_tokens)
    base_tokens = billed(base_raw, base_meta["requests"])
    print(f"\n# baseline: one request per row (SOLO's current prompt shape)")
    print(f"  requests={base_meta['requests']:>8d}  tokens={base_tokens:>10d}"
          f"  tok/row={base_tokens/len(rows):7.1f}  ${base_tokens/1e6*PRICE_PER_MTOK:.4f}")

    results: List[Dict] = []
    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]

    for pname in planners:
        t0 = time.time()
        ordered, col_order, meta = plan(
            rows, pname, counter, plan_sample=args.plan_sample, header=header
        )
        sub_header = [header[c] for c in col_order]
        v = validate(header, rows[: args.validate_rows], col_order, counter)
        print(f"\n## planner={pname}  plan={meta['planning_time_ms']:.0f}ms"
              f"  sort={meta['sort_time_ms']:.0f}ms"
              f"  formula_rel_err={v['rel_error']*100:.2f}%")
        for enc in encodings:
            if enc == "factored_rle":
                reqs = list(pack_requests(sub_header, ordered, counter, q_tokens=args.q_tokens))
                summ = pack_summary(reqs)
            else:
                summ = pack_generic(sub_header, ordered, counter, enc, args.q_tokens)
            tot = summ["total_tokens"]
            row = {
                "planner": pname,
                "encoding": enc,
                "requests": summ["requests"],
                "total_tokens": tot,
                "state_tokens": summ["state_tokens"],
                "question_tokens": summ["question_tokens"],
                "tokens_per_row": tot / len(rows),
                "usd": tot / 1e6 * PRICE_PER_MTOK,
                "vs_baseline": tot / base_tokens,
                "rows_per_request_mean": summ.get("rows_per_request_mean"),
                "planning_time_ms": meta["planning_time_ms"],
                "formula_rel_error": v["rel_error"],
                "wall_s": time.time() - t0,
            }
            results.append(row)
            print(f"   {enc:14s} req={summ['requests']:>7d}  tokens={tot:>10d}"
                  f"  tok/row={row['tokens_per_row']:7.1f}  ${row['usd']:.4f}"
                  f"  = {row['vs_baseline']*100:5.1f}% of baseline")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "table": args.csv, "rows": len(rows), "cols": len(header),
            "tokenizer": args.tokenizer, "q_tokens": args.q_tokens,
            "price_per_mtok": PRICE_PER_MTOK,
            "baseline_per_row": {"tokens": base_tokens, **base_meta},
            "results": results,
        }, indent=2))
        print(f"\n# wrote {out}")


if __name__ == "__main__":
    main()

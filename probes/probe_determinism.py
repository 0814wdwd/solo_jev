#!/usr/bin/env python3
"""Is the same request answered the same way twice?

Every accuracy number in this project is a single measurement. If the model is not
deterministic, those numbers carry a run-to-run variance nobody has quantified, and
differences of a point or two between configurations mean nothing.

Three conditions:
  repeat          the identical request, several times
  reordered       the same rows, questions asked in a different order
  shifted-ids     the same rows and questions, row ids renumbered from an offset

The last two matter because they separate "the model is stochastic" from "the answer
depends on where a row sits in the request", which would be a much bigger problem for
a tool that packs rows into blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, merge, noul  # noqa: E402
from jev_solo import datasets  # noqa: E402
from jev_solo.encodings import encode_csv_rle  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

BASE = "https://openrouter.ai/api"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="flight")
    ap.add_argument("--predicate", default="delay_gt")
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="results/determinism.json")
    args = ap.parse_args()

    table = datasets.get(args.table)
    header, rows_all = table.load(20_000)
    pred = table.predicates[args.predicate]
    counter = get_counter("cl100k_base")
    client = JevClient(base_url=BASE, timeout=300, max_rpm=900)

    idx = [header.index(c) for c in pred.cols]
    usable = []
    for r in rows_all:
        if any(r[i] in ("", "NA", "NULL") for i in idx):
            continue
        usable.append([r[i] for i in idx])
        if len(usable) >= args.rows:
            break
    order, _ = plan_columns(usable, "solo_greedy", counter, header=pred.cols)
    ph = [pred.cols[i] for i in order]
    rows = lex_sort_rows(apply_order(usable, order))
    pins = tuple(range(len(ph)))
    state = encode_csv_rle(ph, rows, row_ids=True, pin=pins, label_pinned=True)

    def ask(qs):
        r = client.ask(state, qs)
        if not r.ok:
            print(f"  HTTP {r.status} {str(r.error)[:100]}")
            return None
        return {k: float(v["noul"]) for k, v in r.answers.items()
                if isinstance(v, dict) and "noul" in v}

    base_qs = merge(*[noul(f"row{i + 1}", pred.ask_verbose.format(rid=i + 1))
                      for i in range(len(rows))])

    print(f"# {args.table}/{args.predicate}, {len(rows)} rows, {args.reps} repeats")
    runs = [ask(base_qs) for _ in range(args.reps)]
    runs = [r for r in runs if r]
    ref = runs[0]
    exact = all(all(abs(r[k] - ref[k]) < 1e-12 for k in ref) for r in runs[1:])
    spread = max(max(abs(r[k] - ref[k]) for k in ref) for r in runs[1:]) if len(runs) > 1 else 0.0
    print(f"  repeat       identical={exact}  max |delta p| = {spread:.6f}")

    # questions asked in a different order
    keys = list(base_qs)
    shuffled = {k: base_qs[k] for k in np.random.default_rng(0).permutation(keys)}
    r2 = ask(shuffled)
    d2 = max(abs(r2[k] - ref[k]) for k in ref) if r2 else float("nan")
    print(f"  reordered    max |delta p| = {d2:.6f}")

    # same rows, ids renumbered from 101
    state_shift = encode_csv_rle(ph, rows, row_ids=True, pin=pins, label_pinned=True)
    state_shift = "\n".join(
        (f"r{int(ln.split(',')[0][1:]) + 100}," + ",".join(ln.split(",")[1:]))
        if ln.startswith("r") and ln.split(",")[0][1:].isdigit() else ln
        for ln in state_shift.split("\n"))
    qs_shift = merge(*[noul(f"row{i + 1}", pred.ask_verbose.format(rid=i + 101))
                       for i in range(len(rows))])
    rs = client.ask(state_shift, qs_shift)
    if rs.ok:
        got = {k: float(v["noul"]) for k, v in rs.answers.items()
               if isinstance(v, dict) and "noul" in v}
        d3 = max(abs(got[k] - ref[k]) for k in ref if k in got)
        print(f"  shifted-ids  max |delta p| = {d3:.6f}")
    else:
        d3 = float("nan")
        print(f"  shifted-ids  HTTP {rs.status}")

    print(f"\n{client.budget_line()}")
    verdict = ("deterministic" if exact and d2 < 1e-9
               else "stable but not bit-identical" if max(spread, d2) < 0.02
               else "varies run to run -- single measurements carry real noise")
    print(f"# verdict: {verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"identical": exact, "repeat_spread": spread, "reorder_delta": d2,
         "shifted_id_delta": d3, "verdict": verdict, "rows": len(rows),
         "reps": len(runs)}, indent=2))
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()

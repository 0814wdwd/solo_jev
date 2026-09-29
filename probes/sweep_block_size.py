#!/usr/bin/env python3
"""How much accuracy does each row you add to a block cost?

The end-to-end run exposed a knob the component experiments could not see. Those
experiments held the block at 120 rows and compared encodings; the packer, left to
minimise cost, fills the token budget instead and puts ~555 rows in a request. Measured
end to end that cost 9.2 points of balanced accuracy on a signed numeric comparison
and 0.1 on an easy predicate, against a per-row baseline that scores 100% on both.

So block size trades throughput against accuracy, and nobody has measured the curve.
This does: same rows, same predicate, same encoding, block size swept.

What the curve is for: picking a block size per predicate. A predicate that holds its
accuracy to 555 rows should be packed to the budget; one that decays should be capped,
paying some throughput to get the accuracy back. Without the curve, the packer's
default is a guess.

  python3 probes/sweep_block_size.py --table flight --predicates delay_gt,state_eq
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient  # noqa: E402
from jev_solo import datasets  # noqa: E402
from jev_solo.pipeline import PRICE_PER_MTOK, Scanner  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

BASE = "https://openrouter.ai/api"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="flight")
    ap.add_argument("--predicates", default="delay_gt")
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--blocks", default="15,60,120,240,600",
                    help="block sizes to sweep; None = fill the token budget")
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--out", default="results/block_sweep.json")
    args = ap.parse_args()

    table = datasets.get(args.table)
    header, rows = table.load(40_000)
    counter = get_counter("cl100k_base")
    client = JevClient(base_url=BASE, timeout=300, max_rpm=900)
    sizes = [int(b) for b in args.blocks.split(",") if b.strip()] + [None]

    records = []
    for pname in [p.strip() for p in args.predicates.split(",") if p.strip()]:
        pred = table.predicates[pname]
        usable = []
        for row in rows:
            vals = [row[header.index(c)] for c in pred.cols]
            if any(v in ("", "NA", "NULL") for v in vals):
                continue
            try:
                pred.truth(vals)
            except Exception:
                continue
            usable.append(row)
            if len(usable) >= args.rows:
                break

        print(f"\n## {pname}  ({pred.shape})  n={len(usable)}")
        print(f"   {'block':>7s} {'requests':>9s} {'rows/s':>8s} {'bal@0.5':>8s} "
              f"{'bal@fit':>8s} {'thresh':>7s} {'tokens':>9s} {'$/100k rows':>12s}")
        for size in sizes:
            scanner = Scanner(table, client, counter, pin=True, max_block_rows=size)
            res = scanner.run_packed(header, usable, pred)
            scanner.fit(res, frac=args.fit_frac)
            half = res.scores(False).get(pname, float("nan"))
            fit = res.scores(True).get(pname, float("nan"))
            per100k = res.billed_tokens / max(1, res.rows) * 100_000 / 1e6 * PRICE_PER_MTOK
            actual = res.rows / max(1, res.requests)
            label = f"{size}" if size else f"budget({actual:.0f})"
            print(f"   {label:>7s} {res.requests:9d} {res.rows/max(1e-9,res.wall_s):8.1f} "
                  f"{half*100:7.1f}% {fit*100:7.1f}% {res.thresholds[pname]:7.2f} "
                  f"{res.billed_tokens:9d} {per100k:12.2f}")
            records.append({
                "predicate": pname, "block_cap": size, "rows_per_request": actual,
                "requests": res.requests, "rows": res.rows,
                "billed_tokens": res.billed_tokens, "wall_s": res.wall_s,
                "rows_per_s": res.rows / max(1e-9, res.wall_s),
                "bal_half": half, "bal_fit": fit,
                "threshold": res.thresholds[pname],
                "usd_per_100k_rows": per100k, "failures": res.failures,
            })

    print(f"\n{client.budget_line()}")
    good = [r for r in records if r["bal_fit"] == r["bal_fit"]]
    if good:
        best = max(good, key=lambda r: r["bal_fit"])
        cheap = min(good, key=lambda r: r["usd_per_100k_rows"])
        print(f"# most accurate: block={best['block_cap']} at {best['bal_fit']*100:.1f}% "
              f"(${best['usd_per_100k_rows']:.2f}/100k rows)")
        print(f"# cheapest:      block={cheap['block_cap']} at {cheap['bal_fit']*100:.1f}% "
              f"(${cheap['usd_per_100k_rows']:.2f}/100k rows)")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"args": vars(args), "records": records}, indent=2))
    print(f"# wrote {p}")


if __name__ == "__main__":
    main()

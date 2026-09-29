#!/usr/bin/env python3
"""Measure both arms end to end, at a scale where neither has to be projected.

The README's throughput headline was arithmetic on 5,000-row component
measurements. This replaces it with a measurement: run the per-row baseline and the
packed pipeline over the same rows, same predicates, same client rate-limit policy,
and report wall clock as observed.

The baseline is the expensive arm — one request per row — so the scale is chosen to
keep it affordable rather than to flatter it. `--baseline-limit` caps how many rows
the baseline actually sends when a full run would be wasteful; the packed arm always
runs on every row, and the comparison then scales the baseline's *measured* rate
instead of guessing one.

  python3 probes/run_end_to_end.py --table flight --rows 2000 --predicates delay_gt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient  # noqa: E402
from jev_solo import datasets  # noqa: E402
from jev_solo.pipeline import PRICE_PER_MTOK, Scanner  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

BASE = "https://openrouter.ai/api"
REQ_PER_MIN = 1200  # vendor-documented cap, used only for the projection line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="flight")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--predicates", default="delay_gt")
    ap.add_argument("--baseline-limit", type=int, default=400,
                    help="rows the per-row baseline actually sends (it is one request "
                         "per row); its measured rate is scaled for the comparison")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--max-rpm", type=int, default=900,
                    help="same policy for both arms -- the rate limit is what is "
                         "being compared, so the arms must not get different policies")
    ap.add_argument("--pin", dest="pin", action="store_true", default=True)
    ap.add_argument("--no-pin", dest="pin", action="store_false")
    ap.add_argument("--project-to", type=int, default=100_000,
                    help="row count to project both arms to, for the headline")
    ap.add_argument("--out", default="results/end_to_end.json")
    args = ap.parse_args()

    table = datasets.get(args.table)
    header, rows = table.load(max(args.rows, 40_000))
    counter = get_counter("cl100k_base")
    client = JevClient(base_url=BASE, timeout=300, max_rpm=args.max_rpm)
    scanner = Scanner(table, client, counter, pin=args.pin)

    preds = [p.strip() for p in args.predicates.split(",") if p.strip()]
    if args.predicates == "ALL":
        preds = list(table.predicates)

    print(f"# table={table.name}  rows={args.rows}  predicates={preds}")
    print(f"# same client policy for both arms: max_rpm={args.max_rpm}, "
          f"pin={'on' if args.pin else 'off'}")

    out = {"args": vars(args), "arms": []}
    for pname in preds:
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

        emit = scanner.emit_report(header, usable, pred)
        print(f"\n## {pname}  ({pred.shape})  n={len(usable)}")
        print("   emit fraction of predicate columns: "
              + ", ".join(f"{k}={v*100:.0f}%" for k, v in emit.items())
              + ("   <- low: pinning is doing real work here"
                 if min(emit.values()) < 0.25 else ""))

        packed = scanner.run_packed(header, usable, pred)
        scanner.fit(packed)
        print("   " + packed.summary())
        print(f"   balanced accuracy: {packed.scores(False)[pname]*100:.1f}% at 0.5 "
              f"(n={packed.n_scored(False)[pname]}), "
              f"{packed.scores(True)[pname]*100:.1f}% at fitted "
              f"{packed.thresholds[pname]:.2f} (held out, n={packed.n_scored(True)[pname]})")

        base = None
        if not args.skip_baseline:
            base = scanner.run_baseline(header, usable, pred, limit=args.baseline_limit)
            scanner.fit(base)
            print("   " + base.summary())
            print(f"   balanced accuracy: {base.scores(False)[pname]*100:.1f}% at 0.5 "
                  f"(n={base.n_scored(False)[pname]}), "
                  f"{base.scores(True)[pname]*100:.1f}% at fitted "
                  f"{base.thresholds[pname]:.2f} (held out, n={base.n_scored(True)[pname]})")

        # Keep the probabilities: without them the run cannot be re-analysed
        # offline, and an earlier version of this script discarded them, which
        # meant its own accuracy numbers could not be recomputed after a fix.
        rec = {"predicate": pname, "emit": emit,
               "packed": vars(packed),
               "packed_bal_half": packed.scores(False).get(pname),
               "packed_bal_fit": packed.scores(True).get(pname)}
        if base:
            rec["baseline"] = vars(base)
            rec["baseline_bal_half"] = base.scores(False).get(pname)
            rec["baseline_bal_fit"] = base.scores(True).get(pname)

            # Per-row rates, measured on each arm, then scaled to --project-to.
            n = args.project_to
            b_s = base.wall_s / max(1, base.rows) * n
            p_s = packed.wall_s / max(1, packed.rows) * n
            b_req = base.requests / max(1, base.rows) * n
            p_req = packed.requests / max(1, packed.rows) * n
            b_tok = base.billed_tokens / max(1, base.rows) * n
            p_tok = packed.billed_tokens / max(1, packed.rows) * n
            quota_s = b_req / REQ_PER_MIN * 60
            print(f"\n   scaling each arm's MEASURED per-row rate to {n:,} rows:")
            print(f"     requests   {b_req:>12,.0f}  ->{p_req:>10,.0f}   "
                  f"({b_req/max(1,p_req):.0f}x fewer)")
            print(f"     tokens     {b_tok:>12,.0f}  ->{p_tok:>10,.0f}   "
                  f"({b_tok/max(1,p_tok):.1f}x fewer)")
            print(f"     cost       ${b_tok/1e6*PRICE_PER_MTOK:>11,.2f}  "
                  f"->${p_tok/1e6*PRICE_PER_MTOK:>9,.2f}")
            print(f"     wall clock {b_s/60:>11,.1f}m  ->{p_s/60:>9,.1f}m   "
                  f"({b_s/max(1e-9,p_s):.0f}x faster)")
            print(f"     the baseline's request count alone costs "
                  f"{quota_s/60:.0f} min of the documented {REQ_PER_MIN}/min quota")
            rec["projection"] = {"rows": n, "baseline_requests": b_req,
                                 "packed_requests": p_req, "baseline_tokens": b_tok,
                                 "packed_tokens": p_tok, "baseline_wall_s": b_s,
                                 "packed_wall_s": p_s, "quota_seconds": quota_s}
        out["arms"].append(rec)

    print(f"\n{client.budget_line()}")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"# wrote {p}")


if __name__ == "__main__":
    main()

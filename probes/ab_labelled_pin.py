#!/usr/bin/env python3
"""Does labelling the pinned cells recover what dittoing costs?

The end-to-end run showed the packed arm at 66.2% balanced accuracy on `delay_gt`
against a per-row baseline at 100%. Inspecting the actual payload showed why: a
dittoed row reads

    r2,,,,,,,,,,,,,,,,,2,-5.00,3,2023-01-03,N605LR,-8.00

so finding DepDelay means counting seventeen commas. Pinning had guaranteed the value
was present and not that the model could tell which column it was — the same positional
reading that makes columnar_rle unusable. `row_kv`, which the baseline uses, carries
"DepDelay=-5.00" and needs no counting.

So this is a defect in the encoding, not a property of the model, and the fix is to
label the pinned cells. Four arms at one block size, same rows, same questions:

    row_kv        every value labelled (the accuracy reference, most expensive)
    csv_block     no ditto: values all present, positional but no comma runs
    csv_rle+pin   pinned but unlabelled (what the end-to-end run measured)
    csv_rle+pin+label   pinned and labelled (the candidate fix)
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
from jev_solo.encodings import encode_csv_block, encode_csv_rle, encode_row_kv  # noqa: E402
from jev_solo.pipeline import bal_acc, fit_threshold  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

BASE = "https://openrouter.ai/api"

ARMS = {
    "row_kv": lambda h, r, pin: encode_row_kv(h, r, row_ids=True),
    "csv_block": lambda h, r, pin: encode_csv_block(h, r, row_ids=True),
    "csv_rle+pin": lambda h, r, pin: encode_csv_rle(h, r, row_ids=True, pin=pin,
                                                    label_pinned=False),
    "csv_rle+pin+label": lambda h, r, pin: encode_csv_rle(h, r, row_ids=True, pin=pin,
                                                          label_pinned=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="flight")
    ap.add_argument("--predicates", default="delay_gt,state_eq,taxi_sum")
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--block", type=int, default=120)
    ap.add_argument("--out", default="results/ab_labelled_pin.json")
    args = ap.parse_args()

    table = datasets.get(args.table)
    header, rows = table.load(40_000)
    counter = get_counter("cl100k_base")
    client = JevClient(base_url=BASE, timeout=300, max_rpm=900)
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

        keep = table.projection(header, pred)
        sh = [header[i] for i in keep]
        sub = [[r[i] for i in keep] for r in usable]
        order, _ = plan_columns(sub, "solo_greedy", counter, header=sh)
        ph = [sh[i] for i in order]
        ordered = lex_sort_rows(apply_order(sub, order))
        pins = tuple(order.index(sh.index(c)) for c in pred.cols)
        truth = [int(bool(pred.truth([r[p] for p in pins]))) for r in ordered]

        print(f"\n## {pname}  ({pred.shape})  n={len(ordered)}  block={args.block}")
        print(f"   predicate columns at positions {[p + 1 for p in pins]} of {len(ph)}")
        print(f"   {'arm':22s} {'bal@0.5':>8s} {'bal@fit':>8s} {'thresh':>7s} {'tokens':>8s}")
        for arm, enc in ARMS.items():
            probs, kept, billed = [], [], 0
            for s in range(0, len(ordered), args.block):
                block = ordered[s:s + args.block]
                state = enc(ph, block, pins)
                qs = merge(*[noul(f"row{i + 1}", pred.ask_verbose.format(rid=i + 1))
                             for i in range(len(block))])
                r = client.ask(state, qs)
                if not r.ok:
                    print(f"   {arm:22s} HTTP {r.status} {str(r.error)[:90]}")
                    break
                billed += r.input_tokens or 0
                for i in range(len(block)):
                    a = r.answers.get(f"row{i + 1}")
                    if isinstance(a, dict) and "noul" in a:
                        probs.append(float(a["noul"]))
                        kept.append(truth[s + i])
            if not probs:
                continue
            p = np.asarray(probs); y = np.asarray(kept, dtype=float)
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(p)); cut = len(idx) // 2
            t = fit_threshold(p[idx[:cut]], y[idx[:cut]])
            half = bal_acc((p >= 0.5).astype(float), y)
            fit = bal_acc((p[idx[cut:]] >= t).astype(float), y[idx[cut:]])
            print(f"   {arm:22s} {half*100:7.1f}% {fit*100:7.1f}% {t:7.2f} {billed:8d}")
            records.append({"predicate": pname, "arm": arm, "block": args.block,
                            "bal_half": half, "bal_fit": fit, "threshold": t,
                            "billed_tokens": billed, "n": len(p)})

    print(f"\n{client.budget_line()}")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"args": vars(args), "records": records}, indent=2))
    print(f"# wrote {p}")


if __name__ == "__main__":
    main()

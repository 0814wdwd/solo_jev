#!/usr/bin/env python3
"""Push the projection down: send the columns the predicate reads, not the whole row.

The end-to-end gap on `delay_gt` decomposes as ~1.5 points from the encoding and ~3.8
points from blocking itself — a blocked `row_kv` scores 96.2% where one-row-per-request
scores 100%. So compression is not where the loss is, and neither is the ditto
convention. What we had never questioned is the *width*: a predicate over two columns
was being handed a 20-column projection, because the component experiments inherited
SOLO's setup, where wide rows are the point.

A database would push the projection down. `Predicate.cols` already declares what the
question reads, so we can too. This measures what that is worth, on both axes at once:
fewer tokens per row *and* nothing irrelevant to read past.

Sweeps projection width against encoding at a fixed block size, so the only thing
changing is how many columns the model sees.
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
from jev_solo.encodings import encode_csv_rle, encode_row_kv  # noqa: E402
from jev_solo.pipeline import bal_acc, fit_threshold  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

BASE = "https://openrouter.ai/api"

ENCODERS = {
    "row_kv": lambda h, r, pin: encode_row_kv(h, r, row_ids=True),
    "csv_rle+pin+label": lambda h, r, pin: encode_csv_rle(h, r, row_ids=True, pin=pin,
                                                          label_pinned=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="flight")
    ap.add_argument("--predicates", default="delay_gt,state_eq")
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--block", type=int, default=120)
    ap.add_argument("--widths", default="0,4,10,20",
                    help="extra columns beyond the predicate's own; 0 = minimal projection")
    ap.add_argument("--out", default="results/ab_projection.json")
    args = ap.parse_args()

    table = datasets.get(args.table)
    header, rows_all = table.load(40_000)
    counter = get_counter("cl100k_base")
    client = JevClient(base_url=BASE, timeout=300, max_rpm=900)
    records = []

    for pname in [p.strip() for p in args.predicates.split(",") if p.strip()]:
        pred = table.predicates[pname]
        pcols = [header.index(c) for c in pred.cols]
        usable = []
        for row in rows_all:
            vals = [row[i] for i in pcols]
            if any(v in ("", "NA", "NULL") for v in vals):
                continue
            try:
                pred.truth(vals)
            except Exception:
                continue
            usable.append(row)
            if len(usable) >= args.rows:
                break

        print(f"\n## {pname}  ({pred.shape})  reads {pred.cols}  n={len(usable)}")
        print(f"   {'width':>6s} {'encoding':20s} {'bal@0.5':>8s} {'bal@fit':>8s} "
              f"{'tokens':>8s} {'tok/row':>8s}")
        for extra in [int(w) for w in args.widths.split(",")]:
            # Predicate's columns first, then a prefix of the rest for context.
            others = [i for i in range(len(header)) if i not in pcols][:extra]
            keep = sorted(set(pcols + others))
            sh = [header[i] for i in keep]
            sub = [[r[i] for i in keep] for r in usable]
            order, _ = plan_columns(sub, "solo_greedy", counter, header=sh)
            ph = [sh[i] for i in order]
            ordered = lex_sort_rows(apply_order(sub, order))
            pins = tuple(order.index(sh.index(c)) for c in pred.cols)
            truth = [int(bool(pred.truth([r[p] for p in pins]))) for r in ordered]

            for arm, enc in ENCODERS.items():
                probs, kept, billed = [], [], 0
                ok = True
                for s0 in range(0, len(ordered), args.block):
                    block = ordered[s0:s0 + args.block]
                    qs = merge(*[noul(f"row{i + 1}", pred.ask_verbose.format(rid=i + 1))
                                 for i in range(len(block))])
                    r = client.ask(enc(ph, block, pins), qs)
                    if not r.ok:
                        print(f"   {len(sh):6d} {arm:20s} HTTP {r.status} "
                              f"{str(r.error)[:70]}")
                        ok = False
                        break
                    billed += r.input_tokens or 0
                    for i in range(len(block)):
                        a = r.answers.get(f"row{i + 1}")
                        if isinstance(a, dict) and "noul" in a:
                            probs.append(float(a["noul"]))
                            kept.append(truth[s0 + i])
                if not ok or not probs:
                    continue
                p = np.asarray(probs); y = np.asarray(kept, dtype=float)
                idx = np.random.default_rng(0).permutation(len(p))
                cut = len(idx) // 2
                t = fit_threshold(p[idx[:cut]], y[idx[:cut]])
                half = bal_acc((p >= 0.5).astype(float), y)
                fit = bal_acc((p[idx[cut:]] >= t).astype(float), y[idx[cut:]])
                print(f"   {len(sh):6d} {arm:20s} {half*100:7.1f}% {fit*100:7.1f}% "
                      f"{billed:8d} {billed/len(p):8.1f}")
                records.append({"predicate": pname, "n_cols": len(sh),
                                "extra_cols": extra, "encoding": arm,
                                "bal_half": half, "bal_fit": fit, "threshold": t,
                                "billed_tokens": billed,
                                "tokens_per_row": billed / len(p), "n": len(p)})

    print(f"\n{client.budget_line()}")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"args": vars(args), "records": records}, indent=2))
    print(f"# wrote {p}")


if __name__ == "__main__":
    main()

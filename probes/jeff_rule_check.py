#!/usr/bin/env python3
"""Does the pinning rule hold on a second decision model, or is it a Jev quirk?

Everything in this repo was measured against one model. The rule it produces — pin
and label the columns a predicate reads — is justified by a mechanism (a value that
is elided has to be recovered from a distant row, and a value in a dittoed CSV slot
has to be located by counting) that should apply to any model reading a serialized
table. If it is instead a property of Jev, the tool's default is wrong for everyone
else.

`jeff` is an MIT self-hosted stand-in: GLiFormer 400M behind the same wire protocol.
It is a far weaker model — 75.5% on AG News against Jev's 90.5% — so its absolute
numbers are not comparable and are not the point. What transfers or fails to transfer
is the *direction and size* of the pinning effect.

The sharpest case is used: the synthetic table's cardinality-4 column, written on 7%
of rows, where Jev goes 50.6% -> 100.0% with pinning.

Start jeff first (see scripts/setup_jeff.sh), with its defaults raised:

    JEFF_API_KEYS=devkey JEFF_MAX_QUESTIONS=512 JEFF_MAX_STATE_CHARS=200000 uv run jeff
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
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.recalibrate import balanced_accuracy  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--table", default="synthetic")
    ap.add_argument("--predicates", default="low_card,a_gt_b")
    ap.add_argument("--rows", type=int, default=240)
    ap.add_argument("--block", type=int, default=60)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--out", default="results/jeff_rule_check.json")
    args = ap.parse_args()

    table = datasets.get(args.table)
    header, rows_all = table.load(40_000)
    counter = get_counter("cl100k_base")
    client = JevClient(base_url=args.base_url, model=args.model, timeout=600,
                       max_rpm=None)

    arms = {
        "row_kv": lambda h, r, pin: encode_row_kv(h, r, row_ids=True),
        "csv_rle": lambda h, r, pin: encode_csv_rle(h, r, row_ids=True,
                                                    label_pinned=False),
        "csv_rle+pin": lambda h, r, pin: encode_csv_rle(h, r, row_ids=True, pin=pin,
                                                        label_pinned=True),
    }
    records = []
    print(f"# second model at {args.base_url}; absolute numbers are not comparable "
          f"to Jev, the direction of the pinning effect is")

    for pname in [p.strip() for p in args.predicates.split(",") if p.strip()]:
        pred = table.predicates[pname]
        idx = [header.index(c) for c in pred.cols]
        keep = table.projection(header, pred)
        sh = [header[i] for i in keep]
        sub, y = [], []
        for r in rows_all:
            vals = [r[i] for i in idx]
            if any(v in ("", "NA") for v in vals):
                continue
            sub.append([r[i] for i in keep])
            y.append(int(bool(pred.truth(vals))))
            if len(sub) >= args.rows:
                break
        order, _ = plan_columns(sub, "solo_greedy", counter, header=sh)
        ph = [sh[i] for i in order]
        paired = sorted(zip([[r[i] for i in order] for r in sub], y),
                        key=lambda t: tuple(str(v) for v in t[0]))
        ordered = [a for a, _ in paired]
        truth = np.asarray([b for _, b in paired], dtype=float)
        pins = tuple(order.index(sh.index(c)) for c in pred.cols)

        print(f"\n## {pname}  n={len(ordered)}  base={truth.mean():.3f}")
        print(f"   {'arm':14s} {'bal@0.5':>8s} {'answered':>9s}")
        for arm, enc in arms.items():
            probs, kept = [], []
            for s0 in range(0, len(ordered), args.block):
                blk = ordered[s0:s0 + args.block]
                qs = merge(*[noul(f"row{i + 1}", pred.ask_verbose.format(rid=i + 1))
                             for i in range(len(blk))])
                r = client.ask(enc(ph, blk, pins), qs)
                if not r.ok:
                    print(f"   {arm:14s} HTTP {r.status} {str(r.error)[:90]}")
                    break
                for i in range(len(blk)):
                    a = r.answers.get(f"row{i + 1}")
                    if isinstance(a, dict) and "noul" in a:
                        probs.append(float(a["noul"]))
                        kept.append(truth[s0 + i])
            if not probs:
                continue
            bal = balanced_accuracy((np.asarray(probs) >= 0.5).astype(float),
                                    np.asarray(kept))
            print(f"   {arm:14s} {bal*100:7.1f}% {len(probs):5d}/{len(ordered):<4d}")
            records.append({"predicate": pname, "arm": arm, "bal_half": bal,
                            "n": len(probs), "base_rate": float(truth.mean())})

    by = {}
    for r in records:
        by.setdefault(r["predicate"], {})[r["arm"]] = r["bal_half"]
    print("\n# pinning effect on this model:")
    for pname, d in by.items():
        if "csv_rle" in d and "csv_rle+pin" in d:
            print(f"   {pname:12s} {d['csv_rle']*100:5.1f}% -> {d['csv_rle+pin']*100:5.1f}% "
                  f"({(d['csv_rle+pin']-d['csv_rle'])*100:+.1f} pts)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"base_url": args.base_url,
                                          "records": records}, indent=2))
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()

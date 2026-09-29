#!/usr/bin/env python3
"""Accuracy and calibration by encoding, at a size that can carry a conclusion.

Replaces probe_accuracy.py's n=40 / 2-predicate pilot with 360 rows per cell over
7 predicate shapes and 7 encodings, and records every returned probability so
calibration can be scored separately (jev_solo/calibration.py).

Two methodology fixes over the pilot:

  * NATURAL sampling, not 50/50 balanced. Balancing forces a 0.5 base rate, which
    makes any calibration measurement meaningless -- a model calibrated to the
    real rate looks broken on a balanced sample. Base rate is reported, and
    balanced accuracy is reported alongside accuracy so skew cannot flatter a
    result.
  * Blocked into 120-row requests. row_kv at 200 rows x 24 cols exceeds Jev's
    documented 32k state budget, so the widest encoding sets the block size and
    every encoding uses the same one.

Predicate shapes span the mechanism variable (where the referenced columns land
in the planned order, hence how often their values are emitted):
  numeric compare, categorical equality, numeric threshold, set membership,
  arithmetic over two columns, and a conjunction.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, merge, noul, save  # noqa: E402
from jev_solo.encodings import (  # noqa: E402
    encode_csv_block, encode_csv_rle, encode_columnar_rle, encode_factored_rle,
    encode_row_kv,
)
from jev_solo.objective import encode_columns, prefix_group_counts  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402
from jev_solo import datasets  # noqa: E402

BASE = "https://openrouter.ai/api"


# Tables, projections and predicates now live in jev_solo/datasets.py, so a caller
# can point this at their own workload instead of editing the probe. PREDICATES and
# load_table stay as thin aliases because probe_decompose.py imports them.
PREDICATES = {k: {"cols": v.cols, "shape": v.shape, "ask_verbose": v.ask_verbose,
                  "ask_terse": v.ask_terse, "truth": v.truth}
              for k, v in datasets.FLIGHT.predicates.items()}


def load_table(path: str, limit: int) -> tuple[List[str], List[List[str]]]:
    for spec in datasets.TABLES.values():
        if spec.path == path:
            return spec.load(limit)
    return datasets.TableSpec(name="ad-hoc", path=path, predicates={}).load(limit)


ENCODERS: Dict[str, Callable] = {
    "row_kv": lambda h, r, pin: encode_row_kv(h, r, row_ids=True),
    "csv_block": lambda h, r, pin: encode_csv_block(h, r, row_ids=True),
    "csv_rle": lambda h, r, pin: encode_csv_rle(h, r, row_ids=True),
    "csv_rle+pin": lambda h, r, pin: encode_csv_rle(h, r, row_ids=True, pin=pin),
    "columnar_rle": lambda h, r, pin: encode_columnar_rle(h, r, row_ids=True),
    "columnar_rle+pin": lambda h, r, pin: encode_columnar_rle(h, r, row_ids=True, pin=pin),
    "factored_rle": lambda h, r, pin: encode_factored_rle(h, r, row_ids=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-block", type=int, default=120,
                    help="row_kv at 24 cols must stay under the 32k state budget")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--cols", type=int, default=20)
    ap.add_argument("--planner", default="solo_greedy")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--terse", action="store_true",
                    help="compact predicate wording (r7: ArrDelay > DepDelay) instead of\nexplicit prose; a wording ablation, since phrasing moves accuracy a lot")
    ap.add_argument("--predicates", default="ALL")
    ap.add_argument("--encodings", default=",".join(ENCODERS))
    ap.add_argument("--random-sample", action="store_true",
                    help="sample rows randomly from the pool instead of taking the head. "
                         "The flight CSV is sorted by carrier and route, so the head gives "
                         "degenerate base rates (Distance>1000 and carrier in {AA,DL,UA} are "
                         "both 0/360 there). Random sampling fixes that, but lowers block "
                         "redundancy, which RAISES emit fractions and therefore flatters the "
                         "ditto encodings relative to a production full-table sort.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table", default="flight",
                    help=f"which table to run: {sorted(datasets.TABLES)}")
    ap.add_argument("--out", default="results/or_accuracy_scaled.json")
    args = ap.parse_args()

    counter = get_counter("cl100k_base")
    c = JevClient(base_url=BASE, timeout=300)
    table = datasets.get(args.table)
    table.width = args.cols
    header_all, rows_all = table.load(40000)
    need = args.rows_per_block * args.blocks
    print(f"# table={table.name}  {len(rows_all)} rows x {len(header_all)} cols")

    if args.predicates == "ALL":
        args.predicates = ",".join(table.predicates)
    preds = [p.strip() for p in args.predicates.split(",") if p.strip()]
    encs = [e.strip() for e in args.encodings.split(",") if e.strip()]
    records: List[dict] = []

    for pname in preds:
        pred = table.predicates[pname]
        spec = {"cols": pred.cols, "shape": pred.shape,
                "ask_verbose": pred.ask_verbose, "ask_terse": pred.ask_terse,
                "truth": pred.truth}
        idxs = [header_all.index(cn) for cn in pred.cols]
        keep = table.projection(header_all, pred)
        header = [header_all[i] for i in keep]

        # Natural order, only validity filtering -- no class balancing.
        usable: List[List[str]] = []
        for row in rows_all:
            vals = [row[i] for i in idxs]
            if any(v in ("", "NA", "NULL") for v in vals):
                continue
            try:
                spec["truth"](vals)
            except Exception:
                continue
            usable.append([row[i] for i in keep])
            if not args.random_sample and len(usable) >= need:
                break
        if args.random_sample:
            import random
            if len(usable) > need:
                usable = random.Random(args.seed).sample(usable, need)
        if len(usable) < need:
            print(f"!! {pname}: only {len(usable)} usable rows, wanted {need}")
        jcols = [header.index(cn) for cn in spec["cols"]]

        col_order, _ = plan_columns(usable, args.planner, counter, header=header)
        sub_header = [header[i] for i in col_order]
        pins = [col_order.index(j) for j in jcols]

        print(f"\n## {pname}  ({spec['shape']})  cols={spec['cols']}")
        blocks = []
        for b in range(args.blocks):
            chunk = usable[b * args.rows_per_block:(b + 1) * args.rows_per_block]
            if not chunk:
                continue
            ordered = lex_sort_rows(apply_order(chunk, col_order))
            truth = [bool(spec["truth"]([r[p] for p in pins])) for r in ordered]
            groups = prefix_group_counts(encode_columns(ordered), list(range(len(sub_header))))
            emit = [groups[p] / len(ordered) for p in pins]
            blocks.append((ordered, truth, emit))
        base_rate = sum(sum(t) for _, t, _ in blocks) / sum(len(t) for _, t, _ in blocks)
        emit_mean = [sum(e[k] for _, _, e in blocks) / len(blocks) for k in range(len(pins))]
        print(f"   base_rate={base_rate:.3f}   predicate cols at positions "
              f"{[p+1 for p in pins]}/{len(sub_header)}, emitted "
              f"{['%.0f%%' % (e*100) for e in emit_mean]} of rows")
        print(f"   {'encoding':18s} {'acc':>7s} {'bal_acc':>8s} {'billed':>8s} {'n':>5s}")

        for enc in encs:
            tot = cor = ans = 0
            tp = tn = fp = fn = 0
            billed = 0
            probs_all: List[float] = []
            truth_all: List[bool] = []
            failed = False
            for ordered, truth, _ in blocks:
                state = ENCODERS[enc](sub_header, ordered, tuple(pins))
                qs = merge(*[noul(f"row{i+1}", spec["ask_terse" if args.terse else "ask_verbose"].format(rid=i + 1))
                             for i in range(len(ordered))])
                r = c.ask(state, qs)
                if not r.ok:
                    print(f"   {enc:18s}  HTTP {r.status} {str(r.error)[:110]}")
                    failed = True
                    break
                billed += r.input_tokens or 0
                for i, t in enumerate(truth):
                    tot += 1
                    a = r.answers.get(f"row{i+1}")
                    if not isinstance(a, dict) or "noul" not in a:
                        continue
                    ans += 1
                    p = float(a["noul"])
                    probs_all.append(p)
                    truth_all.append(t)
                    pred = p >= args.threshold
                    cor += int(pred == t)
                    if pred and t:
                        tp += 1
                    elif pred and not t:
                        fp += 1
                    elif not pred and t:
                        fn += 1
                    else:
                        tn += 1
            if failed:
                continue
            acc = cor / max(1, ans)
            rec_pos = tp / max(1, tp + fn)
            rec_neg = tn / max(1, tn + fp)
            bal = (rec_pos + rec_neg) / 2
            print(f"   {enc:18s} {acc*100:6.1f}% {bal*100:7.1f}% {billed:8d} {ans:5d}")
            records.append({
                "table": table.name,
                "predicate": pname, "shape": spec["shape"], "encoding": enc,
                "wording": "terse" if args.terse else "verbose",
                "sampling": "random" if args.random_sample else "head",
                "n": tot, "answered": ans, "correct": cor, "accuracy": acc,
                "balanced_accuracy": bal, "recall_pos": rec_pos, "recall_neg": rec_neg,
                "base_rate": base_rate, "billed_tokens": billed,
                "predicate_positions": [p + 1 for p in pins],
                "n_cols": len(sub_header), "emit_fraction": emit_mean,
                "probs": probs_all, "truth": [int(t) for t in truth_all],
            })

    print(f"\n{c.budget_line()}")
    save(args.out, {"args": {k: v for k, v in vars(args).items()}, "records": records})


if __name__ == "__main__":
    main()

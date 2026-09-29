#!/usr/bin/env python3
"""The experiment that decides whether the token savings are real.

Every compressed encoding here wins by making the model recover values it never
restates: csv_rle and csv_block leave column names to a header row, csv_rle and
factored_rle leave unchanged cells to a ditto convention, columnar_rle transposes
the table and run-length encodes it. All of that is free in tokens and not
obviously free in comprehension. If accuracy drops, the savings are worthless.

Ground truth is computed from the CSV, not judged, so there is nothing to argue
about:
  gt_delay  : ArrDelay > DepDelay        (numeric comparison -- Jev's documented
                                          weak spot, and a relational table is
                                          mostly numbers)
  gt_state  : OriginState == DestState   (categorical equality)

One request per (encoding, task): the state holds N rows and carries N nouls, one
per row, so this also tests whether row ids survive each encoding -- if the model
cannot address rows, accuracy collapses to chance and we will see it.

Reports accuracy, billed tokens, and tokens per correct answer, which is the
number that actually matters: an encoding that is 2x cheaper and 20 points worse
is not a saving.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, merge, noul, save  # noqa: E402
from jev_solo.encodings import ENCODINGS, encode  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402
from jev_solo import datasets  # noqa: E402

CSV_PATH = datasets.FLIGHT.path  # portable: see jev_solo/datasets.py
BASE = "https://openrouter.ai/api"

TASKS = {
    "gt_delay": {
        "cols": ("ArrDelay", "DepDelay"),
        "instructions": "For row {rid}: the arrival delay is strictly greater than "
                        "the departure delay.",
        "truth": lambda a, b: float(a) > float(b),
    },
    "gt_state": {
        "cols": ("OriginState", "DestState"),
        "instructions": "For row {rid}: the origin state and the destination state "
                        "are the same.",
        "truth": lambda a, b: str(a).strip() == str(b).strip(),
    },
}


def load(path: str, want_rows: int, keep_cols: int):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = [h.strip().strip('"') for h in next(r)]
        out = []
        for row in r:
            if len(row) == len(header):
                out.append([c.strip().strip('"') for c in row])
            if len(out) >= want_rows * 40:
                break
    return header, out


def usable(header: Sequence[str], rows: List[List[str]], task: dict, n: int):
    """Rows where ground truth is well defined, balanced between true and false."""
    ia, ib = header.index(task["cols"][0]), header.index(task["cols"][1])
    pos, neg = [], []
    for row in rows:
        a, b = row[ia], row[ib]
        if a in ("", "NA") or b in ("", "NA"):
            continue
        try:
            t = task["truth"](a, b)
        except Exception:
            continue
        (pos if t else neg).append(row)
        if len(pos) >= n // 2 and len(neg) >= n // 2:
            break
    picked = pos[: n // 2] + neg[: n // 2]
    return picked, ia, ib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40, help="rows (= questions) per request")
    ap.add_argument("--cols", type=int, default=24)
    ap.add_argument("--planner", default="solo_greedy")
    ap.add_argument("--threshold", type=float, default=0.5, help="noul -> yes cutoff")
    ap.add_argument("--out", default="results/or_accuracy.json")
    args = ap.parse_args()

    counter = get_counter("cl100k_base")
    c = JevClient(base_url=BASE, timeout=300)
    header_all, rows_all = load(CSV_PATH, args.rows, args.cols)

    records: List[Dict] = []
    print(f"# accuracy by encoding   rows/request={args.rows}  planner={args.planner}")

    for task_name, task in TASKS.items():
        picked, ia, ib = usable(header_all, rows_all, task, args.rows)
        # Keep the two ground-truth columns plus a prefix of the rest, so the
        # state is wide enough to be realistic but small enough to be cheap.
        keep = sorted(set(list(range(args.cols)) + [ia, ib]))
        header = [header_all[i] for i in keep]
        sub = [[r[i] for i in keep] for r in picked]
        ja, jb = header.index(task["cols"][0]), header.index(task["cols"][1])

        col_order, _ = plan_columns(sub, args.planner, counter, header=header)
        ordered = lex_sort_rows(apply_order(sub, col_order))
        sub_header = [header[i] for i in col_order]
        truth = [bool(task["truth"](r[col_order.index(ja)], r[col_order.index(jb)]))
                 for r in ordered]

        print(f"\n## task={task_name}  n={len(ordered)}  "
              f"true={sum(truth)}/{len(truth)}")
        print(f"{'encoding':14s} {'acc':>7s} {'billed':>7s} {'answered':>9s} "
              f"{'tok/correct':>12s} {'latency_ms':>11s}")

        for enc in ENCODINGS:
            state = encode(sub_header, ordered, enc, row_ids=True)
            qs = merge(*[noul(f"row{i+1}", task["instructions"].format(rid=f"r{i+1}"))
                         for i in range(len(ordered))])
            r = c.ask(state, qs)
            if not r.ok:
                print(f"{enc:14s}  HTTP {r.status}  {str(r.error)[:120]}")
                records.append({"task": task_name, "encoding": enc, "ok": False,
                                "status": r.status, "error": str(r.error)[:400]})
                continue
            answers = r.answers
            correct = answered = 0
            preds = []
            for i, t in enumerate(truth):
                a = answers.get(f"row{i+1}")
                if not isinstance(a, dict) or "noul" not in a:
                    preds.append(None)
                    continue
                answered += 1
                p = float(a["noul"])
                pred = p >= args.threshold
                preds.append(p)
                correct += int(pred == t)
            acc = correct / max(1, answered)
            tpc = r.input_tokens / max(1, correct)
            print(f"{enc:14s} {acc*100:6.1f}% {r.input_tokens:7d} "
                  f"{answered:4d}/{len(truth):<4d} {tpc:12.1f} {r.latency_s*1000:11.0f}")
            records.append({"task": task_name, "encoding": enc, "ok": True,
                            "n": len(truth), "answered": answered, "correct": correct,
                            "accuracy": acc, "billed_tokens": r.input_tokens,
                            "tokens_per_correct": tpc, "latency_s": r.latency_s,
                            "proxy_tokens": counter(state), "probs": preds,
                            "truth": truth})

    print(f"\n{c.budget_line()}")
    save(args.out, {"rows_per_request": args.rows, "records": records})


if __name__ == "__main__":
    main()

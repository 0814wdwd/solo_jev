#!/usr/bin/env python3
"""Does pinning the predicate's columns recover the accuracy that ditto costs?

probe_accuracy.py found that csv_rle holds 95% on a predicate over two high-NDV
columns and collapses to 67.5% on a predicate over a low-NDV column -- the one
that sorts early and gets dittoed away. Mechanism check plus candidate fix:

  1. report where each predicate column lands in the planned order and how often
     its value is actually emitted (run count), to confirm the mechanism
  2. compare, on the SAME rows and questions:
       csv_block         no ditto at all      (the accurate, reorder-insensitive one)
       csv_rle           ditto everywhere     (the cheap, inaccurate one)
       csv_rle + pin     ditto except the predicate's columns

If pinning recovers accuracy at a cost between the two, the framework's knob is
predicate-aware compression, not column ordering.

~8 calls, under a cent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, merge, noul, save  # noqa: E402
from jev_solo.encodings import encode_csv_block, encode_csv_rle  # noqa: E402
from jev_solo.objective import encode_columns, prefix_group_counts  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402
from jev_solo import datasets  # noqa: E402
from probe_accuracy import TASKS, load, usable  # noqa: E402

BASE = "https://openrouter.ai/api"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--cols", type=int, default=24)
    ap.add_argument("--planner", default="solo_greedy")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/or_pinned.json")
    args = ap.parse_args()

    counter = get_counter("cl100k_base")
    c = JevClient(base_url=BASE, timeout=300)
    header_all, rows_all = load(datasets.FLIGHT.path, args.rows, args.cols)
    records: List[dict] = []

    for task_name, task in TASKS.items():
        picked, ia, ib = usable(header_all, rows_all, task, args.rows)
        keep = sorted(set(list(range(args.cols)) + [ia, ib]))
        header = [header_all[i] for i in keep]
        sub = [[r[i] for i in keep] for r in picked]
        ja, jb = header.index(task["cols"][0]), header.index(task["cols"][1])

        col_order, _ = plan_columns(sub, args.planner, counter, header=header)
        ordered = lex_sort_rows(apply_order(sub, col_order))
        sub_header = [header[i] for i in col_order]
        pa, pb = col_order.index(ja), col_order.index(jb)
        truth = [bool(task["truth"](r[pa], r[pb])) for r in ordered]

        # Mechanism: how often is each predicate column actually emitted?
        groups = prefix_group_counts(encode_columns(ordered), list(range(len(sub_header))))
        n = len(ordered)
        print(f"\n## task={task_name}  n={n}  true={sum(truth)}")
        for label, pos in (("A", pa), ("B", pb)):
            print(f"  predicate col {label} = {sub_header[pos]:<18s} "
                  f"position {pos + 1}/{len(sub_header)}  "
                  f"emitted {groups[pos]}/{n} rows "
                  f"({groups[pos] / n * 100:.0f}%)")

        qs = merge(*[noul(f"row{i+1}", task["instructions"].format(rid=f"r{i+1}"))
                     for i in range(n)])
        variants = {
            "csv_block": encode_csv_block(sub_header, ordered, row_ids=True),
            "csv_rle": encode_csv_rle(sub_header, ordered, row_ids=True),
            "csv_rle+pin": encode_csv_rle(sub_header, ordered, row_ids=True, pin=(pa, pb)),
        }
        print(f"  {'variant':14s} {'acc':>7s} {'billed':>7s} {'vs csv_block':>13s}")
        base_tok = None
        for name, state in variants.items():
            r = c.ask(state, qs)
            if not r.ok:
                print(f"  {name:14s}  HTTP {r.status} {str(r.error)[:120]}")
                continue
            correct = answered = 0
            for i, t in enumerate(truth):
                a = r.answers.get(f"row{i+1}")
                if isinstance(a, dict) and "noul" in a:
                    answered += 1
                    correct += int((float(a["noul"]) >= args.threshold) == t)
            acc = correct / max(1, answered)
            if base_tok is None:
                base_tok = r.input_tokens
            print(f"  {name:14s} {acc*100:6.1f}% {r.input_tokens:7d} "
                  f"{r.input_tokens / base_tok * 100:12.1f}%")
            records.append({"task": task_name, "variant": name, "accuracy": acc,
                            "answered": answered, "correct": correct, "n": n,
                            "billed_tokens": r.input_tokens,
                            "predicate_positions": [pa + 1, pb + 1],
                            "predicate_emit_frac": [groups[pa] / n, groups[pb] / n],
                            "latency_s": r.latency_s})

    print(f"\n{c.budget_line()}")
    save(args.out, {"records": records})


if __name__ == "__main__":
    main()

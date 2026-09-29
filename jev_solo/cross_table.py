#!/usr/bin/env python3
"""Does the encoding rule hold on a second schema, or was it a property of one table?

Every earlier conclusion came from flight. This pools the runs across tables and
asks three questions that a library has to answer before it can ship a default:

  1. Is pinning ever harmful? If it never is, the library should simply pin the
     predicate's columns by default, and users never need to reason about emit
     fraction at all. If it sometimes is, the rule needs a condition.
  2. Does the emit-fraction mechanism reproduce? Pin must be a large win where the
     predicate's columns are mostly dittoed away and a no-op where they are not --
     on each table independently.
  3. Does the cost/accuracy frontier agree across tables? A frontier that reorders
     per schema cannot be a default; it has to become a measurement step.

Thresholds are fitted per cell against balanced accuracy and averaged over several
train/test splits, because a single split moves a cell by 1-2 points.

Offline: reads only the saved probability records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

RUNS = {
    ("flight", "head"): "results/or_acc_scaled.json",
    ("flight", "random"): "results/or_acc_random.json",
    ("movies", "head"): "results/mv_acc_head.json",
    ("movies", "random"): "results/mv_acc_random.json",
}
ENCS = ["row_kv", "csv_block", "csv_rle", "csv_rle+pin",
        "columnar_rle", "columnar_rle+pin", "factored_rle"]
PAIRS = [("csv_rle", "csv_rle+pin"), ("columnar_rle", "columnar_rle+pin")]


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return 0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp))


def best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    best, best_score = 0.5, -1.0
    for t in np.unique(np.concatenate([[0.0], p, [1.0]])):
        s = bal_acc((p >= t).astype(float), y)
        if s > best_score + 1e-12:
            best_score, best = s, float(t)
    return best


def load(seeds: int) -> List[dict]:
    cells: List[dict] = []
    for (table, regime), path in RUNS.items():
        f = Path(path)
        if not f.exists():
            print(f"# missing {path} -- skipping {table}/{regime}")
            continue
        for r in json.loads(f.read_text())["records"]:
            if not r.get("probs") or not (0.0 < r.get("base_rate", 0.0) < 1.0):
                continue
            probs = np.asarray(r["probs"], dtype=float)
            y = np.asarray(r["truth"], dtype=float)
            fits, halves = [], []
            for seed in range(seeds):
                idx = np.random.default_rng(seed).permutation(len(y))
                h = len(idx) // 2
                tr, te = idx[:h], idx[h:]
                t = best_threshold(probs[tr], y[tr])
                fits.append(bal_acc((probs[te] >= t).astype(float), y[te]))
                halves.append(bal_acc((probs[te] >= 0.5).astype(float), y[te]))
            cells.append({
                "table": table, "regime": regime, "predicate": r["predicate"],
                "shape": r.get("shape", ""), "encoding": r["encoding"],
                "base_rate": float(y.mean()), "billed_tokens": r["billed_tokens"],
                "min_emit": min(r.get("emit_fraction") or [1.0]),
                "bal_fit": float(np.mean(fits)), "bal_half": float(np.mean(halves)),
            })
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/cross_table.json")
    args = ap.parse_args()
    cells = load(args.seeds)
    tables = sorted({c["table"] for c in cells})
    print(f"# {len(cells)} cells over {len(tables)} tables, thresholds fitted per cell "
          f"(balanced accuracy, {args.seeds} splits)\n")

    # ---- 1. is pinning ever harmful? -------------------------------------
    print("## 1. is pinning ever harmful?")
    worst = []
    for table in tables:
        for base, pinned in PAIRS:
            deltas = []
            for b in cells:
                if b["table"] != table or b["encoding"] != base:
                    continue
                for q in cells:
                    if (q["encoding"] == pinned and q["table"] == table
                            and q["predicate"] == b["predicate"]
                            and q["regime"] == b["regime"]):
                        deltas.append((q["bal_fit"] - b["bal_fit"],
                                       q["billed_tokens"] / b["billed_tokens"] - 1,
                                       b["predicate"], b["regime"], b["min_emit"]))
                        break
            if not deltas:
                continue
            neg = [d for d in deltas if d[0] < -0.01]
            print(f"   {table:8s} {base:14s} -> {pinned:18s} n={len(deltas):2d}  "
                  f"mean {mean(d[0] for d in deltas)*100:+6.1f} pts  "
                  f"tokens {mean(d[1] for d in deltas)*100:+5.1f}%  "
                  f"harmful in {len(neg)}/{len(deltas)}")
            worst.extend(neg)
    if worst:
        print("   cells where pinning cost more than 1 point:")
        for d, tok, pred, reg, emit in sorted(worst)[:6]:
            print(f"     {pred:12s} {reg:7s} emit={emit*100:3.0f}%  {d*100:+.1f} pts")
    else:
        print("   -> pinning never cost more than 1 point anywhere. "
              "A library should pin by default; callers need not reason about emit "
              "fraction to stay safe.")

    # ---- 2. the mechanism, per table -------------------------------------
    print("\n## 2. emit-fraction mechanism, each table independently")
    print(f"   {'table':8s} {'pair':34s} {'emit bucket':13s} {'n':>2s} "
          f"{'before':>7s} {'after':>7s} {'delta':>7s}")
    for table in tables:
        for base, pinned in PAIRS:
            for label, lo, hi in (("low (<25%)", -1.0, 0.25), ("high (>=75%)", 0.75, 2.0)):
                pairs = []
                for b in cells:
                    if (b["table"] != table or b["encoding"] != base
                            or not (lo <= b["min_emit"] < hi)):
                        continue
                    for q in cells:
                        if (q["encoding"] == pinned and q["table"] == table
                                and q["predicate"] == b["predicate"]
                                and q["regime"] == b["regime"]):
                            pairs.append((b, q))
                            break
                if not pairs:
                    continue
                bb = mean(b["bal_fit"] for b, _ in pairs)
                qq = mean(q["bal_fit"] for _, q in pairs)
                print(f"   {table:8s} {base + ' -> ' + pinned:34s} {label:13s} "
                      f"{len(pairs):2d} {bb*100:6.1f}% {qq*100:6.1f}% {(qq-bb)*100:+6.1f}")

    # ---- 3. does the frontier agree across tables? -----------------------
    print("\n## 3. cost/accuracy frontier, per table")
    out_rows = []
    for table in tables:
        sel_all = [c for c in cells if c["table"] == table]
        ref = mean(c["billed_tokens"] for c in sel_all if c["encoding"] == "row_kv")
        print(f"\n   {table}  (n={len(sel_all)} cells, "
              f"{len({c['predicate'] for c in sel_all})} predicates)")
        print(f"   {'encoding':18s} {'bal@0.5':>8s} {'bal@fit':>8s} {'worst':>7s} "
              f"{'tokens':>8s}")
        rows = []
        for enc in ENCS:
            sel = [c for c in sel_all if c["encoding"] == enc]
            if not sel:
                continue
            row = {"table": table, "encoding": enc,
                   "bal_half": mean(c["bal_half"] for c in sel),
                   "bal_fit": mean(c["bal_fit"] for c in sel),
                   "worst": min(c["bal_fit"] for c in sel),
                   "vs_row_kv": mean(c["billed_tokens"] for c in sel) / ref}
            rows.append(row)
            print(f"   {enc:18s} {row['bal_half']*100:7.1f}% {row['bal_fit']*100:7.1f}% "
                  f"{row['worst']*100:6.1f}% {row['vs_row_kv']*100:7.1f}%")
        front = [r for r in rows
                 if not any(o["vs_row_kv"] <= r["vs_row_kv"]
                            and o["bal_fit"] >= r["bal_fit"]
                            and o["encoding"] != r["encoding"] for o in rows)]
        print("   Pareto: " + ", ".join(
            f"{r['encoding']} ({r['bal_fit']*100:.1f}% @ {r['vs_row_kv']*100:.0f}%)"
            for r in sorted(front, key=lambda r: r["vs_row_kv"])))
        out_rows.extend(rows)
        out_rows[-1]["pareto"] = [r["encoding"] for r in front]

    # agreement across tables
    print("\n## do the tables agree on the ranking?")
    by_enc: Dict[str, Dict[str, float]] = {}
    for r in out_rows:
        by_enc.setdefault(r["encoding"], {})[r["table"]] = r["bal_fit"]
    print(f"   {'encoding':18s} " + " ".join(f"{t:>9s}" for t in tables) + "   spread")
    for enc in ENCS:
        d = by_enc.get(enc, {})
        if len(d) < len(tables):
            continue
        vals = [d[t] for t in tables]
        print(f"   {enc:18s} " + " ".join(f"{v*100:8.1f}%" for v in vals)
              + f"   {(max(vals)-min(vals))*100:5.1f} pts")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"cells": cells, "by_table": out_rows}, indent=2))
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()

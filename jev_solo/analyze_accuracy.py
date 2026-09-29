#!/usr/bin/env python3
"""Aggregate the accuracy runs into the accuracy-cost frontier and the mechanism test.

Two runs, two redundancy regimes, same 7 predicates and 7 encodings, 360 rows per
cell:

  head    rows taken consecutively from the file, which is sorted by carrier and
          route -- high block redundancy, like a production full-table sort, but
          two predicates come out degenerate (base rate 0)
  random  rows sampled at random -- representative base rates, lower redundancy,
          which RAISES emit fractions and so flatters the ditto encodings

Balanced accuracy is the headline, not accuracy: base rates run 0.10-0.56, so
plain accuracy rewards a model that just says no.

The mechanism test is the pin effect split by emit fraction. Pinning can only
matter where a predicate's columns are actually being dittoed away, so if the
mechanism is right, pin must be a large win at low emit and a no-op at high emit.
That is a falsifiable prediction with a built-in control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

RUNS = {"head": "results/or_acc_scaled.json", "random": "results/or_acc_random.json"}
ENCS = ["row_kv", "csv_block", "csv_rle", "csv_rle+pin",
        "columnar_rle", "columnar_rle+pin", "factored_rle"]


def load(runs: Dict[str, str]) -> List[dict]:
    out = []
    for label, path in runs.items():
        p = Path(path)
        if not p.exists():
            print(f"# missing {path}, skipping")
            continue
        for r in json.loads(p.read_text())["records"]:
            r["run"] = label
            r["degenerate"] = not (0.0 < r.get("base_rate", 0.0) < 1.0)
            r["min_emit"] = min(r.get("emit_fraction") or [1.0])
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/accuracy_frontier.json")
    args = ap.parse_args()
    recs = [r for r in load(RUNS) if not r["degenerate"]]
    print(f"# {len(recs)} non-degenerate cells "
          f"({len({r['predicate'] for r in recs})} predicates, "
          f"{len({r['run'] for r in recs})} runs), 360 rows each\n")

    # ---- accuracy-cost frontier -------------------------------------------
    print("## frontier: balanced accuracy vs billed tokens, pooled over both runs")
    print(f"{'encoding':18s} {'bal_acc':>8s} {'worst':>7s} {'acc':>7s} "
          f"{'billed':>9s} {'vs row_kv':>10s} {'cells':>6s}")
    ref = mean(r["billed_tokens"] for r in recs if r["encoding"] == "row_kv")
    rows = []
    for enc in ENCS:
        sel = [r for r in recs if r["encoding"] == enc]
        if not sel:
            continue
        bal = mean(r["balanced_accuracy"] for r in sel)
        worst = min(r["balanced_accuracy"] for r in sel)
        acc = mean(r["accuracy"] for r in sel)
        tok = mean(r["billed_tokens"] for r in sel)
        print(f"{enc:18s} {bal*100:7.1f}% {worst*100:6.1f}% {acc*100:6.1f}% "
              f"{tok:9.0f} {tok/ref*100:9.1f}% {len(sel):6d}")
        rows.append({"encoding": enc, "balanced_accuracy": bal, "worst": worst,
                     "accuracy": acc, "billed_tokens": tok, "vs_row_kv": tok / ref,
                     "cells": len(sel)})

    # Pareto set on (cost, balanced accuracy).
    front = [r for r in rows
             if not any(o["vs_row_kv"] <= r["vs_row_kv"]
                        and o["balanced_accuracy"] >= r["balanced_accuracy"]
                        and o["encoding"] != r["encoding"] for o in rows)]
    print("\n   Pareto-optimal: " + ", ".join(
        f"{r['encoding']} ({r['balanced_accuracy']*100:.1f}% @ {r['vs_row_kv']*100:.0f}%)"
        for r in sorted(front, key=lambda r: r["vs_row_kv"])))

    # ---- the mechanism test ----------------------------------------------
    print("\n## mechanism: the pin effect must appear only where emit fraction is low")
    for base, pinned in (("csv_rle", "csv_rle+pin"), ("columnar_rle", "columnar_rle+pin")):
        print(f"\n   {base} -> {pinned}")
        print(f"   {'emit bucket':14s} {'n':>3s} {'bal_acc base':>13s} "
              f"{'bal_acc pin':>12s} {'delta':>8s} {'token cost':>11s}")
        for label, lo, hi in (("low (<25%)", -1.0, 0.25), ("high (>=75%)", 0.75, 2.0)):
            pairs = []
            for b in recs:
                if b["encoding"] != base or not (lo <= b["min_emit"] < hi):
                    continue
                for q in recs:
                    if (q["encoding"] == pinned and q["predicate"] == b["predicate"]
                            and q["run"] == b["run"]):
                        pairs.append((b, q))
                        break
            if not pairs:
                continue
            bb = mean(b["balanced_accuracy"] for b, _ in pairs)
            qq = mean(q["balanced_accuracy"] for _, q in pairs)
            tk = mean(q["billed_tokens"] / b["billed_tokens"] for b, q in pairs)
            print(f"   {label:14s} {len(pairs):3d} {bb*100:12.1f}% {qq*100:11.1f}% "
                  f"{(qq-bb)*100:+7.1f} {(tk-1)*100:+10.1f}%")

    # ---- per-predicate difficulty ----------------------------------------
    print("\n## per predicate: balanced accuracy of the best safe encoding")
    print(f"{'predicate':12s} {'shape':28s} {'min_emit':>9s} {'row_kv':>7s} "
          f"{'csv_block':>10s} {'csv_rle+pin':>12s}")
    for pred in sorted({r["predicate"] for r in recs}):
        sel = [r for r in recs if r["predicate"] == pred]
        shape = sel[0]["shape"]
        emit = mean(r["min_emit"] for r in sel)
        def g(enc):
            s = [r["balanced_accuracy"] for r in sel if r["encoding"] == enc]
            return mean(s) * 100 if s else float("nan")
        print(f"{pred:12s} {shape:28s} {emit*100:8.0f}% {g('row_kv'):6.1f}% "
              f"{g('csv_block'):9.1f}% {g('csv_rle+pin'):11.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"frontier": rows,
                                          "pareto": [r["encoding"] for r in front]}, indent=2))
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()


# --- appended: refit the decision threshold offline -------------------------
# Every accuracy number above thresholds the noul at 0.5. The calibration work
# showed the model over-predicts (mean 0.357 against a 0.263 base rate), so 0.5 is
# simply the wrong cutoff, and probe_decompose showed that fitting it is worth more
# than any decomposition we tried (delay_gt on csv_rle+pin: 89.7% -> 98.8%).
# The saved probabilities let us redo the whole frontier with a fitted threshold at
# zero API cost. Fit on half the rows of each cell, score on the other half.

def refit_frontier(runs=None, seed: int = 0) -> None:
    import numpy as np

    recs = [r for r in load(runs or RUNS) if not r["degenerate"]]
    rng = np.random.default_rng(seed)
    print("\n## frontier with a per-cell fitted threshold (held-out half)")
    print(f"{'encoding':18s} {'bal@0.5':>8s} {'bal@fit':>8s} {'delta':>7s} "
          f"{'thresh':>7s} {'billed':>9s} {'vs row_kv':>10s}")
    ref = mean(r["billed_tokens"] for r in recs if r["encoding"] == "row_kv")
    rows = []
    for enc in ENCS:
        sel = [r for r in recs if r["encoding"] == enc]
        if not sel:
            continue
        b05, bfit, ths = [], [], []
        for r in sel:
            p = np.asarray(r["probs"], dtype=float)
            y = np.asarray(r["truth"], dtype=float)
            if len(p) < 40 or y.min() == y.max():
                continue
            idx = rng.permutation(len(p))
            tr, te = idx[:len(idx) // 2], idx[len(idx) // 2:]

            def bal(pred, yy):
                tp = float(((pred == 1) & (yy == 1)).sum())
                tn = float(((pred == 0) & (yy == 0)).sum())
                fp = float(((pred == 1) & (yy == 0)).sum())
                fn = float(((pred == 0) & (yy == 1)).sum())
                return 0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp))

            best_t, best_v = 0.5, -1.0
            for t in np.linspace(0.02, 0.98, 97):
                v = bal((p[tr] >= t).astype(float), y[tr])
                if v > best_v:
                    best_v, best_t = v, float(t)
            b05.append(bal((p[te] >= 0.5).astype(float), y[te]))
            bfit.append(bal((p[te] >= best_t).astype(float), y[te]))
            ths.append(best_t)
        if not b05:
            continue
        tok = mean(r["billed_tokens"] for r in sel)
        print(f"{enc:18s} {mean(b05)*100:7.1f}% {mean(bfit)*100:7.1f}% "
              f"{(mean(bfit)-mean(b05))*100:+6.1f} {mean(ths):7.2f} "
              f"{tok:9.0f} {tok/ref*100:9.1f}%")
        rows.append({"encoding": enc, "bal_at_half": mean(b05), "bal_at_fit": mean(bfit),
                     "threshold": mean(ths), "billed_tokens": tok, "vs_row_kv": tok / ref})
    front = [r for r in rows
             if not any(o["vs_row_kv"] <= r["vs_row_kv"]
                        and o["bal_at_fit"] >= r["bal_at_fit"]
                        and o["encoding"] != r["encoding"] for o in rows)]
    print("\n   Pareto-optimal with fitted thresholds: " + ", ".join(
        f"{r['encoding']} ({r['bal_at_fit']*100:.1f}% @ {r['vs_row_kv']*100:.0f}%)"
        for r in sorted(front, key=lambda r: r["vs_row_kv"])))
    Path("results/frontier_refit.json").write_text(json.dumps(
        {"rows": rows, "pareto": [r["encoding"] for r in front]}, indent=2))
    print("# wrote results/frontier_refit.json")

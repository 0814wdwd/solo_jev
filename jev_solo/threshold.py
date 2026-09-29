#!/usr/bin/env python3
"""Re-score every accuracy cell under a fitted decision threshold. Offline, no API.

The decomposition experiment found that the single biggest accuracy lever is not
decomposition at all -- it is the threshold. Jev over-predicts (mean 0.357 against
a 0.263 base rate), so 0.5 is simply the wrong cutoff, and fitting it lifted
delay_gt by +7.3 points at zero extra tokens.

But the same experiment found the fit degenerating to majority-class prediction on
a selective predicate (state_eq, base rate 0.128: balanced accuracy 50.0%), because
plain likelihood on ~23 positives prefers to answer "no" every time. Selective
predicates are the normal case in a database, so that failure mode matters more
than the headline.

This re-scores the saved probabilities under three rules, on a held-out half:

  p>=0.5      what every accuracy number so far used
  fit-acc     threshold chosen to maximize accuracy on the train half
  fit-bal     threshold chosen to maximize BALANCED accuracy (Youden's J) --
              the class-weighted objective that should survive low base rates

Zero API cost: everything comes from probs already recorded by
probe_accuracy_scaled.py. The question it answers is whether the encoding ranking
-- and therefore the whole cost/accuracy conclusion -- survives proper
thresholding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np

RUNS = {"head": "results/or_acc_scaled.json", "random": "results/or_acc_random.json"}
ENCS = ["row_kv", "csv_block", "csv_rle", "csv_rle+pin",
        "columnar_rle", "columnar_rle+pin", "factored_rle"]


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return 0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp))


def best_threshold(p: np.ndarray, y: np.ndarray, objective: str) -> float:
    """Scan candidate cutoffs; ties go to the middle of the best plateau."""
    cands = np.unique(np.concatenate([[0.0], p, [1.0]]))
    best, best_score = 0.5, -1.0
    for t in cands:
        pred = (p >= t).astype(float)
        score = float((pred == y).mean()) if objective == "acc" else bal_acc(pred, y)
        if score > best_score + 1e-12:
            best_score, best = score, float(t)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20,
                    help="number of train/test splits to average over. One split is not "
                         "enough: with 180 test rows per cell, split noise moves a cell by "
                         "a point or two, which was enough to flip which encodings looked "
                         "Pareto-optimal between two single-split runs.")
    ap.add_argument("--out", default="results/threshold_rescore.json")
    args = ap.parse_args()

    cells: List[dict] = []
    for run, path in RUNS.items():
        p = Path(path)
        if not p.exists():
            print(f"# missing {path}")
            continue
        for r in json.loads(p.read_text())["records"]:
            if not r.get("probs") or not (0.0 < r.get("base_rate", 0.0) < 1.0):
                continue
            probs = np.asarray(r["probs"], dtype=float)
            y = np.asarray(r["truth"], dtype=float)
            out = {"run": run, "predicate": r["predicate"], "encoding": r["encoding"],
                   "base_rate": float(y.mean()), "billed_tokens": r["billed_tokens"],
                   "min_emit": min(r.get("emit_fraction") or [1.0]),
                   "seeds": args.seeds}
            acc_of: Dict[str, List[float]] = {}
            bal_of: Dict[str, List[float]] = {}
            thr_of: Dict[str, List[float]] = {}
            deg_of: Dict[str, int] = {}
            for seed in range(args.seeds):
                idx = np.random.default_rng(seed).permutation(len(y))
                half = len(idx) // 2
                tr, te = idx[:half], idx[half:]
                for name, obj in (("half", None), ("fit_acc", "acc"), ("fit_bal", "bal")):
                    t = 0.5 if obj is None else best_threshold(probs[tr], y[tr], obj)
                    pred = (probs[te] >= t).astype(float)
                    bal_of.setdefault(name, []).append(bal_acc(pred, y[te]))
                    acc_of.setdefault(name, []).append(float((pred == y[te]).mean()))
                    thr_of.setdefault(name, []).append(t)
                    deg_of[name] = deg_of.get(name, 0) + int(pred.min() == pred.max())
            for name in ("half", "fit_acc", "fit_bal"):
                out[f"{name}_bal"] = float(np.mean(bal_of[name]))
                out[f"{name}_bal_sd"] = float(np.std(bal_of[name]))
                out[f"{name}_acc"] = float(np.mean(acc_of[name]))
                out[f"{name}_threshold"] = float(np.mean(thr_of[name]))
                out[f"{name}_degenerate"] = deg_of[name] / args.seeds > 0.5
                out[f"{name}_collapse_rate"] = deg_of[name] / args.seeds
            out["n_test"] = int(len(y) - len(y) // 2)
            cells.append(out)

    print(f"# {len(cells)} non-degenerate cells, held-out halves, "
          f"averaged over {args.seeds} splits\n")

    print("## effect of the threshold rule, averaged over all cells")
    print(f"{'rule':10s} {'bal_acc':>8s} {'acc':>7s} {'collapsed cells':>16s}")
    for name in ("half", "fit_acc", "fit_bal"):
        print(f"{name:10s} {mean(c[f'{name}_bal'] for c in cells)*100:7.1f}% "
              f"{mean(c[f'{name}_acc'] for c in cells)*100:6.1f}% "
              f"{sum(c[f'{name}_degenerate'] for c in cells):10d}/{len(cells):<5d}")
    print("   'collapsed' = the rule predicts one class for every test row, i.e. it "
          "gave up on discriminating")

    print("\n## does the encoding ranking survive proper thresholding?")
    ref = mean(c["billed_tokens"] for c in cells if c["encoding"] == "row_kv")
    print(f"{'encoding':18s} {'p>=0.5':>8s} {'fit_acc':>8s} {'fit_bal':>8s} "
          f"{'+-sd':>5s} {'worst_bal':>10s} {'tokens':>9s}")
    rows = []
    for enc in ENCS:
        sel = [c for c in cells if c["encoding"] == enc]
        if not sel:
            continue
        h = mean(c["half_bal"] for c in sel)
        fa = mean(c["fit_acc_bal"] for c in sel)
        fb = mean(c["fit_bal_bal"] for c in sel)
        worst = min(c["fit_bal_bal"] for c in sel)
        tok = mean(c["billed_tokens"] for c in sel)
        spread = mean(c["fit_bal_bal_sd"] for c in sel)
        print(f"{enc:18s} {h*100:7.1f}% {fa*100:7.1f}% {fb*100:7.1f}% "
              f"{spread*100:5.1f} {worst*100:9.1f}% {tok/ref*100:8.1f}%")
        rows.append({"encoding": enc, "half_bal": h, "fit_acc_bal": fa,
                     "fit_bal_bal": fb, "fit_bal_sd": spread, "worst_bal": worst,
                     "vs_row_kv": tok / ref})

    front = [r for r in rows
             if not any(o["vs_row_kv"] <= r["vs_row_kv"]
                        and o["fit_bal_bal"] >= r["fit_bal_bal"]
                        and o["encoding"] != r["encoding"] for o in rows)]
    print("\n   Pareto under fit_bal: " + ", ".join(
        f"{r['encoding']} ({r['fit_bal_bal']*100:.1f}% @ {r['vs_row_kv']*100:.0f}%)"
        for r in sorted(front, key=lambda r: r["vs_row_kv"])))

    print("\n## the selective-predicate failure: cells with base rate < 0.15")
    hard = [c for c in cells if c["base_rate"] < 0.15]
    print(f"{'rule':10s} {'bal_acc':>8s} {'collapsed':>11s}   (n={len(hard)} cells)")
    for name in ("half", "fit_acc", "fit_bal"):
        print(f"{name:10s} {mean(c[f'{name}_bal'] for c in hard)*100:7.1f}% "
              f"{sum(c[f'{name}_degenerate'] for c in hard):6d}/{len(hard):<4d}")

    print("\n## per predicate, fit_bal balanced accuracy")
    print(f"{'predicate':12s} {'base':>6s} {'emit':>6s} {'row_kv':>8s} "
          f"{'csv_block':>10s} {'csv_rle+pin':>12s}")
    for pred in sorted({c["predicate"] for c in cells}):
        sel = [c for c in cells if c["predicate"] == pred]
        def g(enc):
            s = [c["fit_bal_bal"] for c in sel if c["encoding"] == enc]
            return mean(s) * 100 if s else float("nan")
        print(f"{pred:12s} {mean(c['base_rate'] for c in sel):6.3f} "
              f"{mean(c['min_emit'] for c in sel)*100:5.0f}% {g('row_kv'):7.1f}% "
              f"{g('csv_block'):9.1f}% {g('csv_rle+pin'):11.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"cells": cells, "by_encoding": rows,
                                          "pareto": [r["encoding"] for r in front]},
                                         indent=2))
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()

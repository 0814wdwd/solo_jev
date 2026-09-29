#!/usr/bin/env python3
"""Can question decomposition lift the two predicates that resist every encoding?

`state_eq` tops out at 85.6% and `delay_gt` at 93.0% even under row_kv, the most
accurate and most expensive encoding. Both are ordinary SQL predicates, so that
ceiling is the project's real limit. A published third-party result got 62.6% ->
95% on phishing by splitting one question into five atomic ones and combining them
with logistic regression, at equal cost because Jev bills input only and evaluates
questions in parallel. This tests whether that transfers to relational predicates.

Three strategies per (predicate, encoding):

  A single      ask the predicate directly, thresholded at 0.5 -- today's baseline
  A+fit         the SAME single probability, but with a fitted threshold. Without
                this arm, B and C get credit for something that is not
                decomposition at all: their combiner also absorbs the bias we
                measured (mean predicted 0.357 against a 0.263 base rate), so
                A -> A+fit isolates the calibration effect and A+fit -> B/C
                isolates decomposition proper.
  B ensemble    several atomic / reworded nouls, combined by logistic regression
  C extract     ask the model only to READ each operand into a coarse band with a
                `score` question, then evaluate the predicate in code

C is the one that matches the diagnosis. Jev is documented weak on numbers, and
`delay_gt` is a signed numeric comparison, so C moves the arithmetic out of the
model and leaves it only perception. If C wins, the framework's rule is "the
engine computes, the model only reads".

B and C fit a combiner, so every number reported is on a held-out half. A needs no
fitting but is scored on the same half, so the comparison is fair.

The question that decides whether any of this matters for cost: can a CHEAPER
encoding plus decomposition reach row_kv's accuracy? That would buy the savings
back without paying for them in errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, merge, noul, save, score  # noqa: E402
from jev_solo.encodings import encode_csv_block, encode_csv_rle, encode_row_kv  # noqa: E402
from jev_solo.objective import encode_columns, prefix_group_counts  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402
from jev_solo import datasets  # noqa: E402
from probe_accuracy_scaled import PREDICATES, load_table  # noqa: E402

BASE = "https://openrouter.ai/api"
CSV_PATH = datasets.FLIGHT.path  # portable: see jev_solo/datasets.py

DELAY_BANDS = ["below -30", "-30 to -10", "-10 to 0", "0 to 15", "15 to 60", "above 60"]

# Atomic / reworded sub-questions. Each takes the row id and returns one noul.
ENSEMBLE: Dict[str, List[Tuple[str, str]]] = {
    "delay_gt": [
        ("arr_pos", "For row r{rid}: the arrival delay is greater than zero."),
        ("dep_pos", "For row r{rid}: the departure delay is greater than zero."),
        ("worsened", "For row r{rid}: the delay got worse between departure and arrival."),
        ("arr_big", "For row r{rid}: the arrival delay is greater than 30."),
    ],
    "state_eq": [
        ("same", "For row r{rid}: the origin state and the destination state are the same."),
        ("intrastate", "For row r{rid}: this is an intrastate flight, beginning and "
                       "ending in one state."),
        ("differs", "For row r{rid}: the origin state is different from the "
                    "destination state."),
        ("codes_eq", "For row r{rid}: the two-letter origin state code is identical to "
                     "the two-letter destination state code."),
    ],
}

ENCODERS: Dict[str, Callable] = {
    "row_kv": lambda h, r, pin: encode_row_kv(h, r, row_ids=True),
    "csv_block": lambda h, r, pin: encode_csv_block(h, r, row_ids=True),
    "csv_rle+pin": lambda h, r, pin: encode_csv_rle(h, r, row_ids=True, pin=pin),
}


def logistic_fit(X: np.ndarray, y: np.ndarray, iters: int = 800, lr: float = 0.5,
                 l2: float = 1e-3) -> np.ndarray:
    """Plain gradient-descent logistic regression with an intercept."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[0.0, w[1:]]
        w -= lr * g
    return w


def logistic_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1 / (1 + np.exp(-Xb @ w))


def balanced_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = float(((pred == 1) & (truth == 1)).sum())
    tn = float(((pred == 0) & (truth == 0)).sum())
    fp = float(((pred == 1) & (truth == 0)).sum())
    fn = float(((pred == 0) & (truth == 1)).sum())
    return 0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-per-block", type=int, default=120)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--cols", type=int, default=20)
    ap.add_argument("--planner", default="solo_greedy")
    ap.add_argument("--predicates", default="delay_gt,state_eq")
    ap.add_argument("--encodings", default="row_kv,csv_block,csv_rle+pin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", action="store_true",
                    help="consecutive rows from the head of the carrier-sorted file: the "
                         "high-redundancy regime that a production full-table sort "
                         "produces, and the only regime where state_eq is actually hard "
                         "(71.5% vs 99.7% balanced accuracy under row_kv)")
    ap.add_argument("--out", default="results/or_decompose.json")
    args = ap.parse_args()

    counter = get_counter("cl100k_base")
    c = JevClient(base_url=BASE, timeout=600)
    header_all, rows_all = load_table(CSV_PATH, 40000)
    need = args.rows_per_block * args.blocks
    rng = np.random.default_rng(args.seed)
    records: List[dict] = []

    for pname in [p.strip() for p in args.predicates.split(",") if p.strip()]:
        spec = PREDICATES[pname]
        idxs = [header_all.index(cn) for cn in spec["cols"]]
        keep = sorted(set(list(range(args.cols)) + idxs))
        header = [header_all[i] for i in keep]

        usable = []
        for row in rows_all:
            vals = [row[i] for i in idxs]
            if any(v in ("", "NA", "NULL") for v in vals):
                continue
            try:
                spec["truth"](vals)
            except Exception:
                continue
            usable.append([row[i] for i in keep])
        usable = (usable[:need] if args.head
                  else [usable[i] for i in rng.permutation(len(usable))[:need]])
        jcols = [header.index(cn) for cn in spec["cols"]]
        col_order, _ = plan_columns(usable, args.planner, counter, header=header)
        sub_header = [header[i] for i in col_order]
        pins = [col_order.index(j) for j in jcols]

        blocks = []
        for b in range(args.blocks):
            chunk = usable[b * args.rows_per_block:(b + 1) * args.rows_per_block]
            if chunk:
                ordered = lex_sort_rows(apply_order(chunk, col_order))
                truth = [bool(spec["truth"]([r[p] for p in pins])) for r in ordered]
                blocks.append((ordered, truth))
        emit = [prefix_group_counts(encode_columns(b[0]), list(range(len(sub_header))))[p]
                / len(b[0]) for p in pins for b in blocks[:1]]
        base_rate = sum(sum(t) for _, t in blocks) / sum(len(t) for _, t in blocks)
        print(f"\n## {pname}  base_rate={base_rate:.3f}  "
              f"emit={['%.0f%%' % (e * 100) for e in emit]}  n={need}")
        print(f"   {'encoding':13s} {'strategy':10s} {'bal_acc':>8s} {'acc':>7s} "
              f"{'billed':>8s} {'n_test':>7s}")

        for enc in [e.strip() for e in args.encodings.split(",") if e.strip()]:
            # Collect per-row features for each strategy, across all blocks.
            feats: Dict[str, List[List[float]]] = {"A": [], "B": [], "C": []}
            truths: List[int] = []
            billed: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
            broke = False

            for ordered, truth in blocks:
                state = ENCODERS[enc](sub_header, ordered, tuple(pins))
                n = len(ordered)

                # A: the predicate as one question.
                qa = merge(*[noul(f"row{i+1}", spec["ask_verbose"].format(rid=i + 1))
                             for i in range(n)])
                ra = c.ask(state, qa)
                # B: atomic / reworded nouls.
                qb = merge(*[noul(f"{key}_{i+1}", tpl.format(rid=i + 1))
                             for i in range(n) for key, tpl in ENSEMBLE[pname]])
                rb = c.ask(state, qb)
                # C: read each operand into a band, compare in code.
                if pname == "delay_gt":
                    qc = merge(*[
                        q for i in range(n) for q in (
                            score(f"arr_{i+1}", f"For row r{i+1}: which band does the "
                                                f"arrival delay fall in?", DELAY_BANDS),
                            score(f"dep_{i+1}", f"For row r{i+1}: which band does the "
                                                f"departure delay fall in?", DELAY_BANDS),
                        )])
                else:
                    qc = merge(*[
                        q for i in range(n) for q in (
                            score(f"rel_{i+1}", f"For row r{i+1}: how far apart are the "
                                                f"origin and the destination?",
                                  ["same state", "neighbouring states",
                                   "same country, far apart"]),
                            noul(f"one_{i+1}", f"For row r{i+1}: only one state is named "
                                               f"anywhere in the row."),
                        )])
                rc = c.ask(state, qc)

                for tag, r in (("A", ra), ("B", rb), ("C", rc)):
                    if not r.ok:
                        print(f"   {enc:13s} {tag:10s}  HTTP {r.status} "
                              f"{str(r.error)[:100]}")
                        broke = True
                    else:
                        billed[tag] += r.input_tokens or 0
                if broke:
                    break

                for i, t in enumerate(truth):
                    a = ra.answers.get(f"row{i+1}", {})
                    feats["A"].append([float(a.get("noul", 0.5))])
                    feats["B"].append([
                        float(rb.answers.get(f"{key}_{i+1}", {}).get("noul", 0.5))
                        for key, _ in ENSEMBLE[pname]])
                    if pname == "delay_gt":
                        av = rc.answers.get(f"arr_{i+1}", {})
                        dv = rc.answers.get(f"dep_{i+1}", {})
                        a_s = float(av.get("score", 2.5))
                        d_s = float(dv.get("score", 2.5))
                        feats["C"].append([a_s, d_s, a_s - d_s])
                    else:
                        rv = rc.answers.get(f"rel_{i+1}", {})
                        ov = rc.answers.get(f"one_{i+1}", {})
                        feats["C"].append([float(rv.get("score", 1.0)),
                                           float(ov.get("noul", 0.5))])
                    truths.append(int(t))

            if broke:
                continue

            y = np.asarray(truths, dtype=float)
            idx = rng.permutation(len(y))
            half = len(idx) // 2
            tr, te = idx[:half], idx[half:]

            for tag in ("A", "A+fit", "B", "C"):
                X = np.asarray(feats["A" if tag == "A+fit" else tag], dtype=float)
                if tag == "A":  # no fitting: threshold the single probability
                    pred = (X[te, 0] >= 0.5).astype(float)
                else:
                    w = logistic_fit(X[tr], y[tr])
                    pred = (logistic_predict(w, X[te]) >= 0.5).astype(float)
                bal = balanced_accuracy(pred, y[te])
                acc = float((pred == y[te]).mean())
                bt = billed["A" if tag == "A+fit" else tag]
                print(f"   {enc:13s} {tag:10s} {bal*100:7.1f}% {acc*100:6.1f}% "
                      f"{bt:8d} {len(te):7d}")
                records.append({"predicate": pname, "encoding": enc, "strategy": tag,
                                "balanced_accuracy": bal, "accuracy": acc,
                                "billed_tokens": bt, "n_test": int(len(te)),
                                "base_rate": base_rate, "emit_fraction": emit,
                                "n_features": int(X.shape[1]),
                                "fitted": tag != "A",
                                "sampling": "head" if args.head else "random"})

    print(f"\n{c.budget_line()}")
    save(args.out, {"args": vars(args), "records": records})


if __name__ == "__main__":
    main()

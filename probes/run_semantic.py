#!/usr/bin/env python3
"""The value case: predicates the shown columns do not determine.

A gradient-boosted tree scores 97-100% on every computable predicate in this repo,
for free, so those predicates are a poor advert for a decision model however well it
does on them. The interesting question is what happens when the answer is not in the
data shown: the model sees a film's title and is asked what kind of film it is, while
ground truth comes from a genre flag it never sees.

Compared against a TF-IDF classifier over the same titles, which is the honest cheap
alternative for a text input, and which scores 50-59% here.

Ground truth is a genre flag on a public film table; multi-genre films mean a title
can legitimately be several things at once, so these numbers are about relative
ability, not an absolute ceiling.
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
from jev_solo.encodings import encode_csv_rle  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.recalibrate import balanced_accuracy, fit as fit_cal  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402

BASE = "https://openrouter.ai/api"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1200)
    ap.add_argument("--block", type=int, default=60)
    ap.add_argument("--predicates", default="is_action,is_comedy,is_drama,is_horror")
    ap.add_argument("--out", default="results/semantic.json")
    args = ap.parse_args()

    table = datasets.get("movies")
    header, rows_all = table.load(40_000)
    counter = get_counter("cl100k_base")
    client = JevClient(base_url=BASE, timeout=300, max_rpm=900)
    tfidf = {r["predicate"].lower(): r["tfidf"]
             for r in json.loads(Path("results/baselines.json").read_text())["records"]
             if r.get("kind") == "semantic"} if Path("results/baselines.json").exists() else {}
    records = []

    print(f"{'predicate':12s} {'n':>5s} {'base':>6s} {'jev@0.5':>8s} {'jev@fit':>8s} "
          f"{'tfidf':>7s} {'AUC':>6s} {'tok/row':>8s}")
    for pname in [p.strip() for p in args.predicates.split(",") if p.strip()]:
        pred = table.predicates[pname]
        show = [header.index(c) for c in pred.cols]
        label = [header.index(c) for c in pred.label_cols]

        shown, truth = [], []
        for r in rows_all:
            if any(not r[i] for i in show) or any(r[i] in ("", "NA") for i in label):
                continue
            shown.append([r[i] for i in show])
            truth.append(int(bool(pred.truth([r[i] for i in label]))))
            if len(truth) >= args.rows:
                break
        if len(set(truth)) < 2:
            print(f"{pname:12s} degenerate, skipped")
            continue

        order, _ = plan_columns(shown, "solo_greedy", counter, header=pred.cols)
        # Keep rows paired with their labels through the sort.
        paired = sorted(zip([[r[i] for i in order] for r in shown], truth),
                        key=lambda t: tuple(str(v) for v in t[0]))
        ordered = [a for a, _ in paired]
        y = np.asarray([b for _, b in paired], dtype=float)
        ph = [pred.cols[i] for i in order]
        pins = tuple(range(len(ph)))  # everything shown is what the question reads

        probs, billed = [], 0
        for s0 in range(0, len(ordered), args.block):
            blk = ordered[s0:s0 + args.block]
            qs = merge(*[noul(f"row{i + 1}", pred.ask_verbose.format(rid=i + 1))
                         for i in range(len(blk))])
            r = client.ask(encode_csv_rle(ph, blk, row_ids=True, pin=pins,
                                          label_pinned=True), qs)
            if not r.ok:
                print(f"{pname:12s} HTTP {r.status} {str(r.error)[:80]}")
                break
            billed += r.input_tokens or 0
            for i in range(len(blk)):
                a = r.answers.get(f"row{i + 1}")
                probs.append(float(a["noul"]) if isinstance(a, dict) and "noul" in a
                             else 0.5)
        if len(probs) != len(y):
            continue
        p = np.asarray(probs)
        cal = fit_cal(p, y, platt=True)
        idx = np.random.default_rng(0).permutation(len(p))
        te = idx[len(idx) // 2:]
        half = balanced_accuracy((p >= 0.5).astype(float), y)
        fit = balanced_accuracy((p[te] >= cal.threshold).astype(float), y[te])
        print(f"{pname:12s} {len(y):5d} {y.mean():6.3f} {half*100:7.1f}% {fit*100:7.1f}% "
              f"{(tfidf.get(pname) or float('nan'))*100:6.1f}% "
              f"{(cal.auc if cal.auc else float('nan')):6.3f} {billed/len(p):8.1f}")
        records.append({"predicate": pname, "n": int(len(y)), "base_rate": float(y.mean()),
                        "jev_half": half, "jev_fit": fit, "auc": cal.auc,
                        "tfidf": tfidf.get(pname), "tokens_per_row": billed / len(p),
                        "threshold": cal.threshold})

    print(f"\n{client.budget_line()}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"records": records}, indent=2))
    print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The throughput headline, measured at 100k rows instead of scaled from 5k.

Also the first use of the public API end to end, which is the point: if `Scan` cannot
do this, the library does not work.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from jev_solo import Scan, datasets
from jev_solo.recalibrate import balanced_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=100_000)
ap.add_argument("--predicate", default="weekend")
ap.add_argument("--calibrate-rows", type=int, default=600)
ap.add_argument("--out", default="results/run_100k.json")
a = ap.parse_args()

table = datasets.get("flight")
header, rows = table.load(a.rows + a.calibrate_rows + 5000)
pred = table.predicates[a.predicate]
idx = [header.index(c) for c in pred.cols]
usable = [r for r in rows if all(r[i] not in ("", "NA", "NULL") for i in idx)]
cal_rows, run_rows = usable[:a.calibrate_rows], usable[a.calibrate_rows:a.calibrate_rows + a.rows]
print(f"# flight/{a.predicate}: calibrating on {len(cal_rows)}, scanning {len(run_rows):,}")

scan = Scan(table, [a.predicate])
t0 = time.time()
cals = scan.calibrate(cal_rows, header=header)
print(f"# calibrated in {time.time()-t0:.1f}s: {cals[a.predicate].summary()}")

res = scan.run(run_rows, header=header)
print(res.report())

pins_truth = [int(bool(pred.truth([r[i] for i in idx]))) for r in run_rows]
# run() sorts internally; recompute truth in the sorted order it used
planned, ordered, pins = scan._layout(header, run_rows, pred)
y = np.asarray([int(bool(pred.truth([r[p] for p in pins]))) for r in ordered], dtype=float)
p = res.probabilities[a.predicate]
ok = ~np.isnan(p)
print(f"  balanced accuracy: {balanced_accuracy((p[ok] >= 0.5).astype(float), y[ok])*100:.1f}% at 0.5, "
      f"{balanced_accuracy((p[ok] >= cals[a.predicate].threshold).astype(float), y[ok])*100:.1f}% "
      f"at the calibrated threshold {cals[a.predicate].threshold:.2f}")
print(f"  base rate {y.mean():.3f}, {int(ok.sum()):,} of {len(y):,} rows answered")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({
    "rows": res.rows, "requests": res.requests, "billed_tokens": res.billed_tokens,
    "usd": res.usd, "wall_s": res.wall_s, "throttled_s": res.throttled_s,
    "rows_per_s": res.rows_per_s, "failures": res.failures,
    "threshold": cals[a.predicate].threshold,
    "bal_half": balanced_accuracy((p[ok] >= 0.5).astype(float), y[ok]),
    "bal_fit": balanced_accuracy((p[ok] >= cals[a.predicate].threshold).astype(float), y[ok]),
}, indent=2, default=float))
print(f"# wrote {a.out}")

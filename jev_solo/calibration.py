#!/usr/bin/env python3
"""Score Jev's returned probabilities as probabilities, not as labels.

Everything in probe_accuracy*.py thresholds the noul at 0.5 and asks whether the
label is right. That is the wrong question for a database: a selectivity estimate,
an approximate count, or any cost-based decision consumes the probability itself.
So this scores calibration.

Reports, overall and per encoding / predicate / emit-fraction bucket:
  base rate, mean predicted, ECE (equal-width bins), MCE, Brier score,
  and the sign of the miscalibration (over- or under-confident)

Plus two things that make the numbers interpretable:
  * a NOISE FLOOR: the ECE that finite-sample noise alone produces for perfectly
    calibrated predictions at these bin counts. Reporting ECE without it is how
    small samples get mistaken for miscalibration.
  * a TEMPERATURE REFIT on logits, with a train/test split, so we can say whether
    the miscalibration is a fixable monotone squashing or something worse.

An independent study put Jev's ECE at 0.107 against a 0.024 floor, with choice
and score overconfident and noul underconfident. This checks that on relational
predicates, where the answers are grounded in a state we control.

Usage:
  python3 -m jev_solo.calibration --acc results/or_acc_scaled.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

EPS = 1e-6


def ece_mce(probs: np.ndarray, truth: np.ndarray, bins: int = 10) -> Tuple[float, float, list]:
    """Equal-width-bin expected and maximum calibration error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(probs)
    ece = 0.0
    mce = 0.0
    rows = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (probs >= lo) & (probs < hi) if b < bins - 1 else (probs >= lo) & (probs <= hi)
        k = int(sel.sum())
        if k == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan")))
            continue
        conf = float(probs[sel].mean())
        acc = float(truth[sel].mean())
        gap = abs(conf - acc)
        ece += k / n * gap
        mce = max(mce, gap)
        rows.append((lo, hi, k, conf, acc))
    return ece, mce, rows


def noise_floor(probs: np.ndarray, bins: int = 10, trials: int = 200,
                seed: int = 0) -> float:
    """ECE from sampling noise alone, if the predictions were perfectly calibrated.

    Resamples labels from the predicted probabilities themselves, so any ECE at or
    below this is indistinguishable from a perfectly calibrated model at this n.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        fake = (rng.random(len(probs)) < probs).astype(float)
        e, _, _ = ece_mce(probs, fake, bins)
        out.append(e)
    return float(np.mean(out))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def platt_refit(probs: np.ndarray, truth: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """Fit a*logit(p)+b on half the data, score ECE on the other half.

    Temperature alone (b fixed at 0) can only rescale spread. What we measure here
    is mostly a BIAS -- mean predicted probability sits well above the base rate --
    and no temperature can move a bias, which is why the temperature-only fit
    barely helps. Platt scaling adds the intercept, so comparing the two says
    whether the damage is an affine distortion (fixable post hoc) or a ranking
    failure (not fixable).
    """
    if len(probs) < 40:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(probs))
    half = len(idx) // 2
    tr, te = idx[:half], idx[half:]
    z_tr, y_tr = _logit(probs[tr]), truth[tr]

    best = (1.0, 0.0, float("inf"))
    for a in np.linspace(0.1, 3.0, 59):
        for b in np.linspace(-4.0, 2.0, 61):
            q = np.clip(1 / (1 + np.exp(-(a * z_tr + b))), EPS, 1 - EPS)
            nll = -float(np.mean(y_tr * np.log(q) + (1 - y_tr) * np.log(1 - q)))
            if nll < best[2]:
                best = (float(a), float(b), nll)
    a, b, _ = best
    z_te = _logit(probs[te])
    before, _, _ = ece_mce(probs[te], truth[te])
    after, _, _ = ece_mce(1 / (1 + np.exp(-(a * z_te + b))), truth[te])
    # AUC: does the ordering carry signal at all? Affine maps cannot change it.
    order = np.argsort(probs[te])
    y = truth[te][order]
    pos, neg = y.sum(), len(y) - y.sum()
    auc = float("nan")
    if pos and neg:
        ranks = np.arange(1, len(y) + 1)
        auc = float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return {"scale": a, "intercept": b, "ece_before": before, "ece_after": after,
            "auc": auc, "n_test": int(len(te))}


def temperature_refit(probs: np.ndarray, truth: np.ndarray,
                      seed: int = 0) -> Dict[str, float]:
    """Fit one temperature on half the data, score ECE on the other half."""
    if len(probs) < 40:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(probs))
    half = len(idx) // 2
    tr, te = idx[:half], idx[half:]
    z_tr, y_tr = _logit(probs[tr]), truth[tr]

    best_t, best_nll = 1.0, float("inf")
    for t in np.concatenate([np.linspace(0.2, 5.0, 97)]):
        q = 1 / (1 + np.exp(-z_tr / t))
        q = np.clip(q, EPS, 1 - EPS)
        nll = -float(np.mean(y_tr * np.log(q) + (1 - y_tr) * np.log(1 - q)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)

    z_te = _logit(probs[te])
    before, _, _ = ece_mce(probs[te], truth[te])
    after_p = 1 / (1 + np.exp(-z_te / best_t))
    after, _, _ = ece_mce(after_p, truth[te])
    return {"temperature": best_t, "ece_before": before, "ece_after": after,
             "n_test": int(len(te))}


def pool(records: Sequence[dict], key=None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[List[float], List[int]]] = {}
    for r in records:
        if not r.get("probs"):
            continue
        k = "ALL" if key is None else str(r[key])
        p, t = out.setdefault(k, ([], []))
        p.extend(r["probs"])
        t.extend(r["truth"])
    return {k: (np.asarray(p, dtype=float), np.asarray(t, dtype=float))
            for k, (p, t) in out.items()}


def report(title: str, groups: Dict[str, Tuple[np.ndarray, np.ndarray]],
           bins: int, floor_trials: int) -> List[dict]:
    print(f"\n## {title}")
    print(f"{'group':20s} {'n':>6s} {'base':>6s} {'mean_p':>7s} {'ECE':>7s} "
          f"{'floor':>7s} {'MCE':>6s} {'Brier':>7s} {'sign':>14s}")
    rows = []
    for k, (p, t) in sorted(groups.items(), key=lambda kv: -len(kv[1][0])):
        if len(p) < 10:
            continue
        e, m, _ = ece_mce(p, t, bins)
        fl = noise_floor(p, bins, trials=floor_trials)
        brier = float(np.mean((p - t) ** 2))
        gap = float(p.mean() - t.mean())
        # gap is mean predicted minus base rate: a BIAS, not spread overconfidence.
        sign = "over-predicts" if gap > 0.02 else ("under-predicts" if gap < -0.02 else "~unbiased")
        verdict = "" if e <= fl * 1.5 else " *"
        print(f"{k:20s} {len(p):6d} {t.mean():6.3f} {p.mean():7.3f} {e:7.3f}{verdict:2s}"
              f"{fl:6.3f} {m:6.3f} {brier:7.3f} {sign:>14s}")
        rows.append({"group": k, "n": int(len(p)), "base_rate": float(t.mean()),
                     "mean_pred": float(p.mean()), "ece": e, "noise_floor": fl,
                     "mce": m, "brier": brier, "sign": sign,
                     "above_floor": bool(e > fl * 1.5)})
    print("   * = ECE more than 1.5x the noise floor, i.e. real miscalibration")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="results/or_acc_scaled.json")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--floor-trials", type=int, default=200)
    ap.add_argument("--out", default="results/calibration.json")
    args = ap.parse_args()

    data = json.loads(Path(args.acc).read_text())
    recs_all = [r for r in data["records"] if r.get("probs")]
    if not recs_all:
        raise SystemExit("no probability records in that file")
    # A cell whose ground truth is constant carries no calibration information:
    # every predicted probability is compared against a single class.
    recs = [r for r in recs_all if 0.0 < r.get("base_rate", 0.0) < 1.0]
    dropped = sorted({r["predicate"] for r in recs_all if r not in recs})
    if dropped:
        print(f"# excluded degenerate predicates (base rate 0 or 1): {', '.join(dropped)}")

    n_total = sum(len(r["probs"]) for r in recs)
    print(f"# {len(recs)} cells, {n_total} predictions, bins={args.bins}")

    out = {"overall": report("overall", pool(recs), args.bins, args.floor_trials),
           "by_encoding": report("by encoding", pool(recs, "encoding"),
                                 args.bins, args.floor_trials),
           "by_predicate": report("by predicate", pool(recs, "predicate"),
                                  args.bins, args.floor_trials)}

    # Distinguish "squashed but monotone" from "genuinely wrong".
    p, t = pool(recs)["ALL"]
    fit = temperature_refit(p, t)
    pl = platt_refit(p, t)
    if fit:
        print(f"\n## post-hoc recalibration (fit on half, scored on the other half)")
        print(f"   temperature only  T={fit['temperature']:.2f}       "
              f"ECE {fit['ece_before']:.3f} -> {fit['ece_after']:.3f}")
        out["temperature_refit"] = fit
    if pl:
        print(f"   Platt (scale+bias) a={pl['scale']:.2f} b={pl['intercept']:+.2f}  "
              f"ECE {pl['ece_before']:.3f} -> {pl['ece_after']:.3f}   "
              f"AUC={pl['auc']:.3f}   n_test={pl['n_test']}")
        out["platt_refit"] = pl
        if pl["ece_after"] < pl["ece_before"] * 0.5:
            print("   -> an affine map on the logit fixes most of it: the damage is a "
                  "bias/scale distortion, so a per-(question type, encoding) Platt fit "
                  "is a real option")
        else:
            print("   -> even scale+bias does not fix it; the ordering itself is weak "
                  "(see AUC), so these probabilities cannot carry a selectivity estimate")

    # Does compression damage the probabilities, not just the labels?
    buckets: Dict[str, Tuple[List[float], List[int]]] = {}
    for r in recs:
        emit = min(r.get("emit_fraction") or [1.0])
        key = ("emit<25%" if emit < 0.25 else
               "emit 25-75%" if emit < 0.75 else "emit>=75%")
        b = buckets.setdefault(key, ([], []))
        b[0].extend(r["probs"])
        b[1].extend(r["truth"])
    out["by_emit_fraction"] = report(
        "by emit fraction of the predicate's columns",
        {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in buckets.items()},
        args.bins, args.floor_trials)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()

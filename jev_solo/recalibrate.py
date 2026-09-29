"""Turn Jev's probabilities into usable ones.

Measured on relational predicates: Jev over-predicts (mean 0.357 against a 0.263
base rate) and its ECE is 0.100 against a 0.007 resampling noise floor. A
temperature alone barely helps, because what is wrong is a bias and no temperature
moves a bias. An affine map on the logit does: scale 1.35, intercept -1.10 takes
held-out ECE from 0.097 to 0.025, with AUC 0.954. The ranking was always sound;
the mapping was skewed.

So there are two separate things to fit, and they answer different questions:

  Threshold  -- for a verdict. Fit against BALANCED accuracy, never likelihood:
                on a selective predicate the likelihood objective prefers
                answering "no" to everything, which scores well on accuracy and
                decides nothing. Measured: it collapses on 45% of splits.
  Platt      -- for a probability. Needed only if the number itself is consumed,
                e.g. a selectivity estimate or an expected-count.

Both are fitted per predicate, from a labelled sample the caller supplies, and
both are free: they reuse probabilities the API already returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return 0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        k = int(sel.sum())
        if k:
            ece += k / len(p) * abs(float(p[sel].mean()) - float(y[sel].mean()))
    return ece


@dataclass
class Calibrator:
    """A fitted threshold and (optionally) an affine probability correction."""

    threshold: float = 0.5
    scale: float = 1.0
    intercept: float = 0.0
    n_fit: int = 0
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None
    auc: Optional[float] = None
    degenerate: bool = False

    def verdict(self, p) -> np.ndarray:
        return (np.asarray(p, dtype=float) >= self.threshold).astype(bool)

    def probability(self, p) -> np.ndarray:
        z = _logit(np.asarray(p, dtype=float))
        return 1.0 / (1.0 + np.exp(-(self.scale * z + self.intercept)))

    def summary(self) -> str:
        s = f"threshold={self.threshold:.2f}"
        if self.scale != 1.0 or self.intercept != 0.0:
            s += f", platt(a={self.scale:.2f}, b={self.intercept:+.2f})"
        if self.ece_before is not None and self.ece_after is not None:
            s += f", ECE {self.ece_before:.3f}->{self.ece_after:.3f}"
        if self.auc is not None:
            s += f", AUC={self.auc:.3f}"
        if self.degenerate:
            s += "  [no signal: labels are single-class or too few]"
        return s


def fit_threshold(p: np.ndarray, y: np.ndarray) -> float:
    best, best_score = 0.5, -1.0
    for t in np.unique(np.concatenate([[0.0], p, [1.0]])):
        s = balanced_accuracy((p >= t).astype(float), y)
        if s > best_score + 1e-12:
            best_score, best = s, float(t)
    return best


def _auc(p: np.ndarray, y: np.ndarray) -> Optional[float]:
    order = np.argsort(p)
    ys = y[order]
    pos, neg = ys.sum(), len(ys) - ys.sum()
    if not pos or not neg:
        return None
    ranks = np.arange(1, len(ys) + 1)
    return float((ranks[ys == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fit(
    probs: Sequence[float],
    truth: Sequence[int],
    platt: bool = True,
    holdout_frac: float = 0.5,
    seed: int = 0,
) -> Calibrator:
    """Fit on part of a labelled sample and report quality on the rest.

    Reporting on the held-out part matters: a cutoff scored on the rows that chose
    it is an in-sample number, worth 0.7-0.9 points in our own measurements.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(truth, dtype=float)
    if len(p) < 20 or y.min() == y.max():
        return Calibrator(degenerate=True, n_fit=len(p))

    idx = np.random.default_rng(seed).permutation(len(p))
    cut = max(1, int(len(idx) * (1 - holdout_frac)))
    tr, te = idx[:cut], idx[cut:]
    cal = Calibrator(threshold=fit_threshold(p[tr], y[tr]), n_fit=int(len(tr)))

    if platt and len(te) >= 20:
        z, yt = _logit(p[tr]), y[tr]
        best = (1.0, 0.0, float("inf"))
        for a in np.linspace(0.1, 3.0, 59):
            for b in np.linspace(-4.0, 2.0, 61):
                q = np.clip(1 / (1 + np.exp(-(a * z + b))), EPS, 1 - EPS)
                nll = -float(np.mean(yt * np.log(q) + (1 - yt) * np.log(1 - q)))
                if nll < best[2]:
                    best = (float(a), float(b), nll)
        cal.scale, cal.intercept, _ = best
        cal.ece_before = expected_calibration_error(p[te], y[te])
        cal.ece_after = expected_calibration_error(cal.probability(p[te]), y[te])
        cal.auc = _auc(p[te], y[te])
    return cal

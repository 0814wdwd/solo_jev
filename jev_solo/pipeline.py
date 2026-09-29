"""The whole thing, end to end — the one path that had never been run.

Every other number in this project comes from a probe that measures one component:
cost offline, accuracy from one script, thresholds from another. Nothing had ever
gone table -> pack -> encode -> send -> threshold -> verdicts in a single run, which
means the product itself was unmeasured. This is that path.

Two arms, under the *same* client rate-limit policy, because the rate limit is the
thing being compared and giving the arms different policies would fake the result:

  baseline   one request per row, SOLO's current prompt shape (row_kv, no row ids,
             one question). This is what the existing pipeline sends.
  packed     rows packed into budget-feasible requests, encoded with the predicate's
             columns pinned, one question per row, threshold fitted per predicate.

Reports what a user actually needs to decide anything: wall clock, requests, billed
tokens, dollars, and accuracy against ground truth computed from the table — plus how
much of the wall clock was spent waiting on the rate limiter, since that is where the
packed arm wins and hiding it would overstate the model's own speed.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))

from jev_solo.datasets import Predicate, TableSpec  # noqa: E402
from jev_solo.encodings import encode_csv_rle, encode_row_kv  # noqa: E402
from jev_solo.objective import encode_columns, prefix_group_counts  # noqa: E402
from jev_solo.pack import pack_requests  # noqa: E402
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns  # noqa: E402
from jev_solo.tokens import TokenCounter, get_counter  # noqa: E402

PRICE_PER_MTOK = 0.042


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    return 0.5 * (tp / max(1.0, tp + fn) + tn / max(1.0, tn + fp))


def fit_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """Cutoff maximising balanced accuracy.

    Balanced, not raw accuracy: on a selective predicate the accuracy objective
    prefers answering "no" to everything, which scores well and decides nothing.
    """
    best, best_score = 0.5, -1.0
    for t in np.unique(np.concatenate([[0.0], p, [1.0]])):
        s = bal_acc((p >= t).astype(float), y)
        if s > best_score + 1e-12:
            best_score, best = s, float(t)
    return best


@dataclass
class ArmResult:
    arm: str
    rows: int
    requests: int
    billed_tokens: int
    wall_s: float
    throttled_s: float
    failures: int
    probs: Dict[str, List[float]] = field(default_factory=dict)
    truth: Dict[str, List[int]] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)

    @property
    def usd(self) -> float:
        return self.billed_tokens / 1e6 * PRICE_PER_MTOK

    @property
    def model_s(self) -> float:
        """Wall clock excluding time spent waiting on our own rate limiter."""
        return self.wall_s - self.throttled_s

    def scores(self, fitted: bool = True) -> Dict[str, float]:
        out = {}
        for name, ps in self.probs.items():
            p = np.asarray(ps, dtype=float)
            y = np.asarray(self.truth[name], dtype=float)
            if len(p) == 0:
                continue
            t = self.thresholds.get(name, 0.5) if fitted else 0.5
            out[name] = bal_acc((p >= t).astype(float), y)
        return out

    def summary(self) -> str:
        rps = self.rows / self.wall_s if self.wall_s else 0.0
        return (f"{self.arm:9s} rows={self.rows:6d} requests={self.requests:6d} "
                f"tokens={self.billed_tokens:9d} ${self.usd:7.4f} "
                f"wall={self.wall_s:7.1f}s (throttled {self.throttled_s:6.1f}s) "
                f"{rps:8.1f} rows/s  failed={self.failures}")


class Scanner:
    """Runs a set of per-row predicates over a table on a decision model."""

    def __init__(
        self,
        table: TableSpec,
        client: Any,
        counter: Optional[TokenCounter] = None,
        planner: str = "solo_greedy",
        pin: bool = True,
        q_tokens: int = 24,
        max_block_rows: Optional[int] = 120,
    ):
        self.table = table
        self.client = client
        self.counter = counter or get_counter("cl100k_base")
        self.planner = planner
        self.pin = pin
        self.q_tokens = q_tokens
        # 120, not "fill the token budget", and the difference is measured. Past
        # ~120 rows per request the bill barely moves (92,327 -> 90,560 tokens from
        # 120 to 600 rows, 2%) while balanced accuracy falls 3.3 points on a signed
        # numeric comparison. Filling the budget buys throughput, not money, so it
        # has to be asked for rather than assumed. Set None to fill the budget.
        self.max_block_rows = max_block_rows

    # ---- shared setup ----------------------------------------------------
    def _prepare(self, header: Sequence[str], rows: Sequence[Sequence[str]],
                 pred: Predicate):
        keep = self.table.projection(header, pred)
        sub_header = [header[i] for i in keep]
        sub = [[r[i] for i in keep] for r in rows]
        jcols = [sub_header.index(c) for c in pred.cols]
        order, _ = plan_columns(sub, self.planner, self.counter, header=sub_header)
        ordered = lex_sort_rows(apply_order(sub, order))
        planned_header = [sub_header[i] for i in order]
        pins = [order.index(j) for j in jcols]
        truth = [int(bool(pred.truth([r[p] for p in pins]))) for r in ordered]
        return planned_header, ordered, pins, truth

    # ---- arm 1: one request per row --------------------------------------
    def run_baseline(self, header: Sequence[str], rows: Sequence[Sequence[str]],
                     pred: Predicate, limit: Optional[int] = None) -> ArmResult:
        from jev_client import noul

        planned_header, ordered, _, truth = self._prepare(header, rows, pred)
        if limit and limit < len(ordered):
            # Stride across the sorted order, never take a prefix of it. Rows are
            # lexicographically sorted, so the first N share their leading columns:
            # a prefix of a sorted table is a block where the predicate's column may
            # be constant, which pins balanced accuracy at 50% by construction and
            # says nothing about the model.
            step = len(ordered) / limit
            idx = [min(len(ordered) - 1, int(i * step)) for i in range(limit)]
            ordered = [ordered[i] for i in idx]
            truth = [truth[i] for i in idx]
        res = ArmResult("baseline", len(ordered), 0, 0, 0.0, 0.0, 0)
        probs: List[float] = []
        kept: List[int] = []
        t_wall = time.time()
        before_throttle = self.client.throttled_s
        # No row ids and no block: exactly what the existing per-row pipeline sends.
        question = pred.ask_verbose.replace("row r{rid}", "this row").replace("r{rid}", "this row")
        for row, t in zip(ordered, truth):
            state = encode_row_kv(planned_header, [row], row_ids=False)
            r = self.client.ask(state, noul("q", question))
            res.requests += 1
            if not r.ok:
                res.failures += 1
                continue
            res.billed_tokens += r.input_tokens or 0
            a = r.answers.get("q")
            if isinstance(a, dict) and "noul" in a:
                probs.append(float(a["noul"]))
                kept.append(t)
        res.wall_s = time.time() - t_wall
        res.throttled_s = self.client.throttled_s - before_throttle
        res.probs[pred.name] = probs
        res.truth[pred.name] = kept
        return res

    # ---- arm 2: packed, pinned, thresholded ------------------------------
    def run_packed(self, header: Sequence[str], rows: Sequence[Sequence[str]],
                   pred: Predicate) -> ArmResult:
        from jev_client import merge, noul

        planned_header, ordered, pins, truth = self._prepare(header, rows, pred)
        res = ArmResult("packed", len(ordered), 0, 0, 0.0, 0.0, 0)
        probs: List[float] = []
        kept: List[int] = []
        t_wall = time.time()
        before_throttle = self.client.throttled_s

        for req in pack_requests(planned_header, ordered, self.counter,
                                 q_tokens=self.q_tokens,
                                 max_rows=self.max_block_rows):
            block = ordered[req.start:req.end]
            state = encode_csv_rle(planned_header, block, row_ids=True,
                                   pin=tuple(pins) if self.pin else ())
            qs = merge(*[noul(f"row{i + 1}", pred.ask_verbose.format(rid=i + 1))
                         for i in range(len(block))])
            r = self.client.ask(state, qs)
            res.requests += 1
            if not r.ok:
                res.failures += len(block)
                continue
            res.billed_tokens += r.input_tokens or 0
            for i in range(len(block)):
                a = r.answers.get(f"row{i + 1}")
                if isinstance(a, dict) and "noul" in a:
                    probs.append(float(a["noul"]))
                    kept.append(truth[req.start + i])
        res.wall_s = time.time() - t_wall
        res.throttled_s = self.client.throttled_s - before_throttle
        res.probs[pred.name] = probs
        res.truth[pred.name] = kept
        return res

    # ---- threshold fitting ------------------------------------------------
    @staticmethod
    def fit(res: ArmResult, frac: float = 0.5, seed: int = 0) -> Dict[str, float]:
        """Fit a cutoff per predicate on `frac` of the rows; the rest is held out."""
        rng = np.random.default_rng(seed)
        for name, ps in res.probs.items():
            p = np.asarray(ps, dtype=float)
            y = np.asarray(res.truth[name], dtype=float)
            if len(p) < 20 or y.min() == y.max():
                res.thresholds[name] = 0.5
                continue
            idx = rng.permutation(len(p))
            cut = int(len(idx) * frac)
            res.thresholds[name] = fit_threshold(p[idx[:cut]], y[idx[:cut]])
        return res.thresholds

    def emit_report(self, header: Sequence[str], rows: Sequence[Sequence[str]],
                    pred: Predicate) -> Dict[str, float]:
        """Emit fraction of the predicate's columns — the compression-risk diagnostic."""
        planned_header, ordered, pins, _ = self._prepare(header, rows, pred)
        groups = prefix_group_counts(encode_columns(ordered),
                                     list(range(len(planned_header))))
        return {planned_header[p]: groups[p] / max(1, len(ordered)) for p in pins}

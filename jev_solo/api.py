"""The library surface: run per-row predicates over a table, and be told what it cost.

Until now the capability lived in probe scripts and could not be imported. This is
the path a user takes:

    from jev_solo import Scan, datasets

    scan = Scan(datasets.get("flight"), ["delay_gt", "weekend"])
    scan.calibrate(rows[:600])        # labelled sample -> threshold (+ Platt)
    result = scan.run(rows)
    print(result.report())
    verdicts = result.verdicts["delay_gt"]

Defaults are the measured ones, not guesses: pinned and labelled columns, a 60-row
block, the predicate's own columns projected down. Each is documented at the point
it is set, with the measurement that chose it.

The report deliberately states the measured saving *for this table* instead of a
headline. The saving is schema-dependent — 41% of `row_kv` tokens on flight, 65% on
movies — so a promised number would be wrong for somebody.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .datasets import Predicate, TableSpec
from .encodings import encode_csv_rle, encode_row_kv
from .objective import encode_columns, prefix_group_counts
from .pack import pack_requests
from .plan import plan
from .recalibrate import Calibrator, fit as fit_calibrator
from .tokens import TokenCounter, get_counter

PRICE_PER_MTOK = 0.042
REQ_PER_MIN = 1200  # vendor-documented cap, for the quota line in the report


@dataclass
class ScanResult:
    table: str
    rows: int
    requests: int
    billed_tokens: int
    wall_s: float
    throttled_s: float
    failures: int
    verdicts: Dict[str, np.ndarray] = field(default_factory=dict)
    probabilities: Dict[str, np.ndarray] = field(default_factory=dict)
    calibrated: Dict[str, np.ndarray] = field(default_factory=dict)
    emit_fraction: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def usd(self) -> float:
        return self.billed_tokens / 1e6 * PRICE_PER_MTOK

    @property
    def rows_per_s(self) -> float:
        return self.rows / self.wall_s if self.wall_s else 0.0

    def report(self) -> str:
        per_row = self.billed_tokens / max(1, self.rows)
        quota_min = self.rows / REQ_PER_MIN  # one request per row, for contrast
        lines = [
            f"{self.rows:,} rows x {len(self.probabilities)} predicates on {self.table}",
            f"  {self.requests:,} requests, {self.billed_tokens:,} billed tokens "
            f"(${self.usd:.4f}), {per_row:.1f} tokens/row",
            f"  {self.wall_s:.1f}s wall ({self.throttled_s:.1f}s waiting on the rate "
            f"limiter), {self.rows_per_s:.0f} rows/s",
            f"  one request per row would have used {self.rows:,} requests: "
            f"{quota_min:.0f} min of the {REQ_PER_MIN}/min quota on request count alone",
        ]
        if self.failures:
            lines.append(f"  {self.failures} rows had no answer")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


class Scan:
    def __init__(
        self,
        table: TableSpec,
        predicates: Sequence[str],
        client: Any = None,
        *,
        # Measured defaults. Each is set where its evidence is recorded.
        block_rows: Optional[int] = 60,
        pin: bool = True,
        project: bool = True,
        planner: str = "solo_greedy",
        q_tokens: int = 24,
        counter: Optional[TokenCounter] = None,
    ):
        self.table = table
        self.predicate_names = list(predicates)
        self.client = client or self._default_client()
        self.block_rows = block_rows
        self.pin = pin
        # Send the columns the predicate declares, not the whole row. 1.7x cheaper
        # and accuracy-neutral on flight; a database would push the projection down
        # and there is no reason not to.
        self.project = project
        self.planner = planner
        self.q_tokens = q_tokens
        self.counter = counter or get_counter("cl100k_base")
        self.calibrators: Dict[str, Calibrator] = {}

    @staticmethod
    def _default_client():
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probes"))
        from jev_client import JevClient  # noqa: E402

        return JevClient(base_url="https://openrouter.ai/api", timeout=300, max_rpm=900)

    # ---- layout -----------------------------------------------------------
    def _layout(self, header: Sequence[str], rows: Sequence[Sequence[str]],
                pred: Predicate):
        if self.project:
            keep = sorted({header.index(c) for c in pred.cols})
        else:
            keep = self.table.projection(header, pred)
        sub_header = [header[i] for i in keep]
        sub = [[r[i] for i in keep] for r in rows]
        ordered, order, _ = plan(sub, self.planner, self.counter, header=sub_header)
        planned = [sub_header[i] for i in order]
        pins = tuple(order.index(sub_header.index(c)) for c in pred.cols)
        return planned, ordered, pins

    def _ask_blocks(self, planned, ordered, pins, pred, result: ScanResult):
        from jev_client import merge, noul  # noqa: E402

        def enc(h, blk):
            return encode_csv_rle(h, blk, row_ids=True,
                                  pin=pins if self.pin else (), label_pinned=True)

        probs: List[float] = []
        for req in pack_requests(planned, ordered, self.counter, q_tokens=self.q_tokens,
                                 max_rows=self.block_rows, encode_fn=enc):
            block = ordered[req.start:req.end]
            qs = merge(*[noul(f"row{i + 1}", pred.ask_verbose.format(rid=i + 1))
                         for i in range(len(block))])
            r = self.client.ask(enc(planned, block), qs)
            result.requests += 1
            if not r.ok:
                result.failures += len(block)
                probs.extend([float("nan")] * len(block))
                continue
            result.billed_tokens += r.input_tokens or 0
            for i in range(len(block)):
                a = r.answers.get(f"row{i + 1}")
                probs.append(float(a["noul"]) if isinstance(a, dict) and "noul" in a
                             else float("nan"))
        return np.asarray(probs, dtype=float)

    # ---- public API -------------------------------------------------------
    def calibrate(self, rows: Sequence[Sequence[str]], header: Optional[Sequence[str]] = None,
                  platt: bool = True) -> Dict[str, Calibrator]:
        """Fit a threshold (and optionally Platt) per predicate on labelled rows.

        Ground truth comes from the table via `Predicate.truth`, so "labelled" means
        rows where the predicate can actually be evaluated — no annotation needed for
        the predicates shipped here. Bring your own `truth` for your own predicates.
        """
        header = list(header) if header else self.table.load(1)[0]
        for name in self.predicate_names:
            pred = self.table.predicates[name]
            planned, ordered, pins = self._layout(header, rows, pred)
            truth = [int(bool(pred.truth([r[p] for p in pins]))) for r in ordered]
            tmp = ScanResult(self.table.name, len(ordered), 0, 0, 0.0, 0.0, 0)
            p = self._ask_blocks(planned, ordered, pins, pred, tmp)
            ok = ~np.isnan(p)
            self.calibrators[name] = fit_calibrator(p[ok], np.asarray(truth)[ok],
                                                    platt=platt)
        return self.calibrators

    def run(self, rows: Sequence[Sequence[str]],
            header: Optional[Sequence[str]] = None) -> ScanResult:
        header = list(header) if header else self.table.load(1)[0]
        res = ScanResult(self.table.name, len(rows), 0, 0, 0.0, 0.0, 0)
        t0 = time.time()
        before = getattr(self.client, "throttled_s", 0.0)

        for name in self.predicate_names:
            pred = self.table.predicates[name]
            planned, ordered, pins = self._layout(header, rows, pred)

            groups = prefix_group_counts(encode_columns(ordered),
                                         list(range(len(planned))))
            emit = {planned[p]: groups[p] / max(1, len(ordered)) for p in pins}
            res.emit_fraction[name] = emit
            if not self.pin and min(emit.values()) < 0.25:
                res.warnings.append(
                    f"{name}: predicate columns are written on "
                    f"{min(emit.values())*100:.0f}% of rows and pinning is off. "
                    f"Measured, that costs up to 37 points of balanced accuracy.")

            p = self._ask_blocks(planned, ordered, pins, pred, res)
            res.probabilities[name] = p
            cal = self.calibrators.get(name)
            if cal is None:
                res.warnings.append(
                    f"{name}: not calibrated, so verdicts use a 0.5 cutoff. Fitted "
                    f"cutoffs on this workload span 0.09-0.99; call calibrate() first.")
                cal = Calibrator()
            res.verdicts[name] = cal.verdict(np.nan_to_num(p, nan=0.0))
            res.calibrated[name] = cal.probability(np.nan_to_num(p, nan=0.5))

        res.wall_s = time.time() - t0
        res.throttled_s = getattr(self.client, "throttled_s", 0.0) - before
        return res

"""Column-order planners.

solo_greedy reproduces SOLO's Algorithm 1 objective (minimize G^(k), i.e.
maximize prefix sharing) on the same greedy skeleton. token_greedy replaces that
objective with the Jev bill: at each step it picks the column whose own emission
cost G(prefix+a) * w_a is smallest, so a cheap-but-high-NDV column is no longer
treated the same as an expensive one.

Group ids are re-compacted every step (np.unique with return_inverse), which is
the int64-overflow fix the handover doc requires and which the copy in
solo_code/core/code_data_faster.py does not yet have.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .objective import encode_columns, prefix_group_counts
from .tokens import TokenCounter, column_value_weights, rendered_field_weights

PLANNERS = ("default", "random", "ndv", "solo_greedy", "token_greedy", "token_greedy_eps")


def apply_order(rows: Sequence[Sequence[str]], col_order: Sequence[int]) -> List[List[str]]:
    return [[row[c] for c in col_order] for row in rows]


def lex_sort_rows(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    return sorted((list(r) for r in rows), key=lambda r: tuple(str(v) for v in r))


def _greedy(
    codes: np.ndarray,
    weights: Sequence[float] | None,
    eps: float | None = None,
) -> List[int]:
    """Shared greedy skeleton.

    weights=None            -> SOLO's unweighted objective (minimize G).
    weights, eps=None       -> myopic token objective: minimize G(prefix+a)*w_a.
                               Kept as an ablation; it loses, and provably so:
                               every later column has G >= g_a, so the remaining
                               cost is bounded below by g_a * sum(w over
                               remaining), whose sum factor is the same for all
                               candidates. The lower-bound criterion therefore
                               reduces to argmin g_a -- SOLO's own rule.
    weights, eps=0.05       -> second-order correction. Among candidates within
                               (1+eps) of the best group count, take the most
                               expensive column, because G^(k) is
                               non-decreasing and the rearrangement inequality
                               wants large w paired with small G.
    """
    n_rows, n_cols = codes.shape
    remaining = list(range(n_cols))
    order: List[int] = []
    group_ids = np.zeros(n_rows, dtype=np.int64)

    while remaining:
        best_col = None
        best_score = None
        best_ids = None
        best_raw_g = None
        scored = []
        for a in remaining:
            col = codes[:, a]
            k = int(col.max()) + 1 if n_rows else 1
            combined = group_ids * k + col
            uniq, inv = np.unique(combined, return_inverse=True)
            g = int(uniq.size)
            # SOLO: fewest groups. token_greedy: cheapest emission for this column.
            ids = inv.astype(np.int64)
            score = g if weights is None else g * float(weights[a])
            if eps is not None:
                scored.append((a, g, ids))
            if best_raw_g is None or g < best_raw_g:
                best_raw_g = g
            if best_score is None or score < best_score:
                best_score = score
                best_col = a
                best_ids = ids
        if eps is not None and weights is not None:
            # Re-scan: among near-tied group counts, prefer the costliest column.
            threshold = best_raw_g * (1.0 + eps)
            cand = [(a, g, ids) for a, g, ids in scored if g <= threshold]
            best_col, _, best_ids = max(cand, key=lambda t: float(weights[t[0]]))
        order.append(best_col)
        remaining.remove(best_col)
        group_ids = best_ids
    return order


def plan_columns(
    rows: Sequence[Sequence[str]],
    planner: str,
    counter: TokenCounter,
    seed: int = 0,
    plan_sample: int = 20000,
    header: Sequence[str] | None = None,
    eps_tie: float = 0.05,
) -> Tuple[List[int], Dict[str, float]]:
    """Return (col_order, meta). Planning may run on a row sample; scoring never does."""
    if not rows:
        return [], {"planning_time_ms": 0.0, "plan_rows": 0}
    n_cols = len(rows[0])
    if header is None:
        header = [f"c{i}" for i in range(n_cols)]
    started = time.time()

    plan_rows = rows
    if plan_sample and len(rows) > plan_sample:
        idx = random.Random(seed).sample(range(len(rows)), plan_sample)
        plan_rows = [rows[i] for i in idx]

    if planner == "default":
        order = list(range(n_cols))
    elif planner == "random":
        order = list(range(n_cols))
        random.Random(seed).shuffle(order)
    elif planner == "ndv":
        codes = encode_columns(plan_rows)
        ndv = [int(np.unique(codes[:, c]).size) for c in range(n_cols)]
        order = sorted(range(n_cols), key=lambda c: ndv[c])
    elif planner == "solo_greedy":
        order = _greedy(encode_columns(plan_rows), None)
    elif planner in ("token_greedy", "token_greedy_eps"):
        weights = rendered_field_weights(
            plan_rows, header, counter, sample=min(4000, len(plan_rows)), seed=seed
        )
        eps = eps_tie if planner == "token_greedy_eps" else None
        order = _greedy(encode_columns(plan_rows), weights, eps=eps)
    else:
        raise ValueError(f"unknown planner: {planner!r}; known: {PLANNERS}")

    return order, {
        "planning_time_ms": (time.time() - started) * 1000.0,
        "plan_rows": len(plan_rows),
    }


def plan(
    rows: Sequence[Sequence[str]],
    planner: str,
    counter: TokenCounter,
    seed: int = 0,
    plan_sample: int = 20000,
    sort_rows: bool = True,
    header: Sequence[str] | None = None,
    eps_tie: float = 0.05,
) -> Tuple[List[List[str]], List[int], Dict[str, float]]:
    """Full plan: column order, then lexicographic row sort under that order."""
    col_order, meta = plan_columns(
        rows, planner, counter, seed=seed, plan_sample=plan_sample, header=header, eps_tie=eps_tie
    )
    started = time.time()
    ordered = apply_order(rows, col_order)
    if sort_rows:
        ordered = lex_sort_rows(ordered)
    meta["sort_time_ms"] = (time.time() - started) * 1000.0
    return ordered, col_order, meta


def group_count_curve(rows: Sequence[Sequence[str]], col_order: Sequence[int]) -> List[int]:
    return prefix_group_counts(encode_columns(rows), col_order)

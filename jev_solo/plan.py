"""Column-order planners and stable integer row ordering.

The greedy rules are unchanged. Candidates are counted by scanning contiguous
prefix groups with a reusable value-indexed stamp array. Only the chosen column
is partitioned, and its row permutation is reused by full-table planning.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ._integer import PrefixGroups
from .objective import encode_columns, prefix_group_counts
from .tokens import TokenCounter, rendered_field_weights

PLANNERS = ("default", "random", "ndv", "solo_greedy", "token_greedy", "token_greedy_eps")


def apply_order(rows: Sequence[Sequence[str]], col_order: Sequence[int]) -> List[List[str]]:
    return [[row[c] for c in col_order] for row in rows]


def lex_sort_rows(rows: Sequence[Sequence[str]]) -> List[List[str]]:
    """Stable string-lexicographic order via compact integer partitions."""
    if rows and any(len(row) != len(rows[0]) for row in rows):
        # The public helper historically also accepted ragged sequences.
        return sorted((list(row) for row in rows), key=lambda row: tuple(str(v) for v in row))
    state = PrefixGroups(encode_columns(rows))
    for c in range(state.codes.shape[1]):
        state.refine(c)
    return [list(rows[int(i)]) for i in state.permutation]


def _greedy_state(
    codes: np.ndarray,
    weights: Sequence[float] | None,
    eps: float | None = None,
) -> Tuple[List[int], PrefixGroups]:
    """Original greedy choices, with scalar scores and shared workspace.

    No weights: minimize G(prefix+a).
    Weights without eps: minimize G(prefix+a) * w_a.
    Weights with eps: among G <= (1+eps) * min_G, choose the largest w_a.
    Ties retain the first remaining column, as in the original implementation.
    """
    state = PrefixGroups(codes)
    remaining = list(range(codes.shape[1]))
    order: List[int] = []
    while remaining:
        best_col = None
        best_score = None
        scored = []
        for a in remaining:
            g = state.count(a)
            score = g if weights is None else g * float(weights[a])
            scored.append((a, g))
            if best_score is None or score < best_score:
                best_col, best_score = a, score
        if eps is not None and weights is not None:
            threshold = min(g for _, g in scored) * (1.0 + eps)
            best_col = max((a for a, g in scored if g <= threshold),
                           key=lambda a: float(weights[a]))
        order.append(best_col)
        remaining.remove(best_col)
        state.refine(best_col)
    return order, state


def _greedy(
    codes: np.ndarray,
    weights: Sequence[float] | None,
    eps: float | None = None,
) -> List[int]:
    return _greedy_state(codes, weights, eps)[0]


def _plan_columns(
    rows: Sequence[Sequence[str]],
    planner: str,
    counter: TokenCounter,
    seed: int,
    plan_sample: int,
    header: Sequence[str] | None,
    eps_tie: float,
):
    """Also return encoded data/permutation for full-table planning reuse."""
    if not rows:
        return [], {"planning_time_ms": 0.0, "plan_rows": 0}, None, None
    n_cols = len(rows[0])
    if header is None:
        header = [f"c{i}" for i in range(n_cols)]
    started = time.perf_counter()

    plan_rows = rows
    if plan_sample and len(rows) > plan_sample:
        idx = random.Random(seed).sample(range(len(rows)), plan_sample)
        plan_rows = [rows[i] for i in idx]

    codes = None
    permutation = None
    if planner == "default":
        order = list(range(n_cols))
    elif planner == "random":
        order = list(range(n_cols))
        random.Random(seed).shuffle(order)
    elif planner == "ndv":
        codes = encode_columns(plan_rows)
        ndv = codes.max(axis=0).astype(np.intp) + 1
        order = sorted(range(n_cols), key=lambda c: ndv[c])
    elif planner in ("solo_greedy", "token_greedy", "token_greedy_eps"):
        weights = None
        eps = None
        if planner != "solo_greedy":
            weights = rendered_field_weights(
                plan_rows, header, counter, sample=min(4000, len(plan_rows)), seed=seed
            )
            eps = eps_tie if planner == "token_greedy_eps" else None
        codes = encode_columns(plan_rows)
        order, state = _greedy_state(codes, weights, eps)
        permutation = state.permutation
    else:
        raise ValueError(f"unknown planner: {planner!r}; known: {PLANNERS}")

    meta = {"planning_time_ms": (time.perf_counter() - started) * 1000.0,
            "plan_rows": len(plan_rows)}
    return order, meta, codes, permutation


def plan_columns(
    rows: Sequence[Sequence[str]],
    planner: str,
    counter: TokenCounter,
    seed: int = 0,
    plan_sample: int = 20000,
    header: Sequence[str] | None = None,
    eps_tie: float = 0.05,
) -> Tuple[List[int], Dict[str, float]]:
    """Return (col_order, meta). Planning may run on a row sample."""
    order, meta, _, _ = _plan_columns(
        rows, planner, counter, seed, plan_sample, header, eps_tie
    )
    return order, meta


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
    """Column planning and stable lexicographic materialization in one pass.

    A greedy permutation is reusable only if it was computed on every row. A
    sampled plan instead partitions the full encoded table under the selected
    column order. Original cells are materialized once, in the final order.
    """
    col_order, meta, codes, permutation = _plan_columns(
        rows, planner, counter, seed, plan_sample, header, eps_tie
    )
    started = time.perf_counter()
    if not sort_rows:
        ordered = apply_order(rows, col_order)
    else:
        if permutation is None or meta["plan_rows"] != len(rows):
            if codes is None or meta["plan_rows"] != len(rows):
                codes = encode_columns(rows)
            state = PrefixGroups(codes)
            for c in col_order:
                state.refine(c)
            permutation = state.permutation
        ordered = [[rows[int(i)][c] for c in col_order] for i in permutation]
    meta["sort_time_ms"] = (time.perf_counter() - started) * 1000.0
    return ordered, col_order, meta


def group_count_curve(rows: Sequence[Sequence[str]], col_order: Sequence[int]) -> List[int]:
    return prefix_group_counts(encode_columns(rows), col_order)

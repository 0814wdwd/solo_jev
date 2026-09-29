"""The token objective, and the bridge from SOLO's G^(k) to Jev's bill.

For rows sorted lexicographically under column order (c_1..c_m), the number of
value *runs* in column position k equals G^(k), the number of distinct
combinations of the first k columns. The factored encoding emits exactly one
value per run, so

    value_tokens = sum_k  G^(k) * w_{c_k}

with w_c the mean token cost of one value in column c. That is SOLO's own
G^(k) quantity, weighted by per-column token cost -- which is why SOLO's
planning machinery transfers while its objective does not.

analytic_factored_cost() computes that sum plus the fixed overheads actually
present in the encoding (column names, row ids, separators, preamble).
validate() checks the prediction against real serialization + tokenization,
because an objective nobody checked is just a hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .encodings import FACTORED_PREAMBLE, encode
from .tokens import TokenCounter


def encode_columns(rows: Sequence[Sequence[str]]) -> np.ndarray:
    """Factorize each column independently into compact int codes."""
    if not rows:
        return np.zeros((0, 0), dtype=np.int64)
    n_rows, n_cols = len(rows), len(rows[0])
    out = np.empty((n_rows, n_cols), dtype=np.int64)
    for c in range(n_cols):
        col = [str(r[c]) for r in rows]
        _, inv = np.unique(np.asarray(col, dtype=object), return_inverse=True)
        out[:, c] = inv
    return out


def prefix_group_counts(codes: np.ndarray, col_order: Sequence[int]) -> List[int]:
    """G^(1..m) along the given column order.

    Group ids are re-compacted at every step, so the combined key never
    overflows int64 -- the failure the handover doc flags for the original
    greedy_column_order_torch.
    """
    n_rows = codes.shape[0]
    group_ids = np.zeros(n_rows, dtype=np.int64)
    counts: List[int] = []
    for c in col_order:
        col = codes[:, c]
        k = int(col.max()) + 1 if n_rows else 1
        combined = group_ids * k + col
        uniq, inv = np.unique(combined, return_inverse=True)
        group_ids = inv.astype(np.int64)
        counts.append(int(uniq.size))
    return counts


@dataclass
class CostBreakdown:
    total: int
    value_tokens: int
    colname_tokens: int
    rowid_tokens: int
    overhead_tokens: int
    group_counts: List[int] = field(default_factory=list)
    detail: Dict[str, float] = field(default_factory=dict)

    def per_row(self, n_rows: int) -> float:
        return self.total / max(1, n_rows)

    def usd(self, price_per_mtok: float = 0.042) -> float:
        return self.total / 1e6 * price_per_mtok


def analytic_factored_cost(
    header: Sequence[str],
    codes: np.ndarray,
    col_order: Sequence[int],
    field_weights: Sequence[float],
    counter: TokenCounter,
    row_ids: bool = True,
) -> CostBreakdown:
    """Predict factored-encoding tokens without serializing the block.

    field_weights[c] is the mean token cost of the *rendered* fragment
    " | Name=value" for column c (see tokens.rendered_field_weights). Using
    rendered fragments rather than summing name/value/separator pieces removes
    a systematic ~16% overcount from cross-boundary token merges.
    """
    n_rows = codes.shape[0]
    groups = prefix_group_counts(codes, col_order)

    # One emitted field per run; a run in position k costs one rendered field.
    field_tokens = 0.0
    for k, c in enumerate(col_order):
        field_tokens += groups[k] * float(field_weights[c])

    # field_weights include the " | " separator, but the first field emitted on a
    # line has none -- it follows the row id or the "^ " marker. One separator per
    # row is therefore charged and never written.
    field_tokens -= n_rows * counter(" | ")

    rowid_tokens = 0.0
    if row_ids:
        # Measure the rendered prefix as one string. Charging "\nrN: " and the "^"
        # ditto marker separately overcounts by ~1.8x, because the tokenizer merges
        # across that boundary -- the same piecewise-counting error that
        # rendered_field_weights exists to avoid, which this term had kept.
        probe = min(n_rows, 2000)
        first = counter(f"\nr1: ")
        rest = sum(counter(f"\nr{i + 1}: ^ ") for i in range(1, probe))
        per_row = (first + rest) / max(1, probe)
        rowid_tokens = per_row * n_rows

    overhead = counter(FACTORED_PREAMBLE)
    total = field_tokens + rowid_tokens + overhead
    return CostBreakdown(
        total=int(round(total)),
        value_tokens=int(round(field_tokens)),
        colname_tokens=0,
        rowid_tokens=int(round(rowid_tokens)),
        overhead_tokens=int(round(overhead)),
        group_counts=groups,
        detail={"note": "value_tokens holds rendered-field tokens (name+value+sep)"},
    )


def measured_cost(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    counter: TokenCounter,
    encoding: str = "factored_rle",
    row_ids: bool = True,
) -> int:
    """Ground truth: actually serialize the block and tokenize it."""
    return counter(encode(header, rows, encoding, row_ids=row_ids))


def validate(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_order: Sequence[int],
    counter: TokenCounter,
    row_ids: bool = True,
) -> Dict[str, float]:
    """Compare analytic prediction with measured tokens on sorted rows."""
    from .plan import apply_order, lex_sort_rows
    from .tokens import rendered_field_weights

    ordered = lex_sort_rows(apply_order(rows, col_order))
    sub_header = [header[c] for c in col_order]
    codes = encode_columns(ordered)
    weights = rendered_field_weights(ordered, sub_header, counter, sample=len(ordered))
    pred = analytic_factored_cost(
        sub_header, codes, list(range(len(sub_header))), weights, counter, row_ids=row_ids
    )
    meas = measured_cost(sub_header, ordered, counter, "factored_rle", row_ids=row_ids)
    return {
        "predicted": pred.total,
        "measured": meas,
        "abs_error": abs(pred.total - meas),
        "rel_error": abs(pred.total - meas) / max(1, meas),
    }

"""Packing a sorted table into budget-feasible Jev requests.

Documented limits (docs.typesafe.ai, Jev 1.13): 64k tokens per request across
state and all questions, and 32k for state plus the single longest question.
With state = a block of rows and one question per row, both bind:

    state(K) + K * q_tokens <= 64k        (total budget)
    state(K) + q_tokens     <= 32k        (state budget)

Rows are packed in sorted order, which is what makes the factored encoding pay:
adjacent rows share prefixes, so each added row costs only its changed suffix.
Packing and reordering are therefore not independent -- the plan decides how
cheap the packing can get.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from .encodings import FACTORED_PREAMBLE
from .tokens import TokenCounter

TOTAL_BUDGET = 64_000
STATE_BUDGET = 32_000


def _lcp(a: Sequence[str], b: Sequence[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and str(a[i]) == str(b[i]):
        i += 1
    return i


def row_delta_tokens(
    header: Sequence[str],
    row: Sequence[str],
    prev: Sequence[str] | None,
    counter: TokenCounter,
    row_id: str,
) -> int:
    """Token cost of appending one row to a factored block."""
    if prev is None:
        start = 0
    else:
        start = _lcp(prev, row)
        if start >= len(row):
            start = len(row) - 1
    body = " | ".join(f"{header[k]}={row[k]}" for k in range(start, len(row)))
    marker = "^ " if start > 0 else ""
    return counter(f"\n{row_id}: {marker}{body}")


@dataclass
class Request:
    start: int
    end: int  # exclusive
    state_tokens: int
    question_tokens: int

    @property
    def n_rows(self) -> int:
        return self.end - self.start

    @property
    def total_tokens(self) -> int:
        return self.state_tokens + self.question_tokens


def pack_requests(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    counter: TokenCounter,
    q_tokens: int = 24,
    questions_per_row: int = 1,
    total_budget: int = TOTAL_BUDGET,
    state_budget: int = STATE_BUDGET,
    row_ids: bool = True,
    max_rows: Optional[int] = None,
    encode_fn: Optional[Callable[[Sequence[str], Sequence[Sequence[str]]], str]] = None,
    stride: int = 32,
) -> Iterator[Request]:
    """Greedily pack consecutive sorted rows into feasible requests.

    `max_rows` caps the block independently of the token budget. Filling to the
    budget minimises cost and is not free: block size is a throughput/accuracy
    knob, not an implementation detail, so the packer must let the caller set it.

    `encode_fn` is the encoder whose output will actually be sent. Pass it, or the
    budget is accounted against the wrong serialization: the incremental fallback
    models `factored_rle`, which on flight costs 1.79x what `csv_rle` costs for the
    same block, so a caller sending csv_rle under-packs by that factor and never
    reaches the budget it thinks it is filling.
    """
    preamble = counter(FACTORED_PREAMBLE)
    per_row_q = q_tokens * questions_per_row

    i = 0
    n = len(rows)

    if encode_fn is not None:
        # Grow by `stride` rows, measuring the real encoding, then back off to the
        # last block that fits. Approximate only at the stride boundary; the token
        # count reported is the measured one.
        while i < n:
            k = 0
            fitted_tokens = 0
            cap = n - i if max_rows is None else min(n - i, max_rows)
            while k < cap:
                probe = min(cap, k + stride)
                tokens = counter(encode_fn(header, rows[i:i + probe]))
                if (tokens + q_tokens <= state_budget
                        and tokens + probe * per_row_q <= total_budget):
                    k, fitted_tokens = probe, tokens
                    continue
                # shrink one row at a time from the last good point
                for trial in range(probe - 1, k, -1):
                    tokens = counter(encode_fn(header, rows[i:i + trial]))
                    if (tokens + q_tokens <= state_budget
                            and tokens + trial * per_row_q <= total_budget):
                        k, fitted_tokens = trial, tokens
                        break
                break
            if k == 0:  # even one row does not fit: emit it alone and report it
                k = 1
                fitted_tokens = counter(encode_fn(header, rows[i:i + 1]))
            yield Request(start=i, end=i + k, state_tokens=fitted_tokens,
                          question_tokens=k * per_row_q)
            i += k
        return

    while i < n:
        state = preamble
        prev = None
        k = 0
        start = i
        while i < n:
            if max_rows is not None and k >= max_rows:
                break
            rid = f"r{k + 1}" if row_ids else ""
            delta = row_delta_tokens(header, rows[i], prev, counter, rid)
            new_state = state + delta
            new_k = k + 1
            if new_state + q_tokens > state_budget:
                break
            if new_state + new_k * per_row_q > total_budget:
                break
            state = new_state
            prev = rows[i]
            k = new_k
            i += 1
        if k == 0:  # a single row cannot fit: emit it alone and report it
            rid = "r1" if row_ids else ""
            state = preamble + row_delta_tokens(header, rows[start], None, counter, rid)
            i = start + 1
            k = 1
        yield Request(start=start, end=start + k, state_tokens=state, question_tokens=k * per_row_q)


def pack_summary(requests: Sequence[Request]) -> dict:
    if not requests:
        return {"requests": 0, "rows": 0, "state_tokens": 0, "question_tokens": 0, "total_tokens": 0}
    rows = sum(r.n_rows for r in requests)
    state = sum(r.state_tokens for r in requests)
    quest = sum(r.question_tokens for r in requests)
    sizes = [r.n_rows for r in requests]
    return {
        "requests": len(requests),
        "rows": rows,
        "state_tokens": state,
        "question_tokens": quest,
        "total_tokens": state + quest,
        "rows_per_request_mean": rows / len(requests),
        "rows_per_request_min": min(sizes),
        "rows_per_request_max": max(sizes),
    }


def per_row_baseline_tokens(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    counter: TokenCounter,
    q_tokens: int = 24,
    questions_per_row: int = 1,
) -> Tuple[int, dict]:
    """One request per row, SOLO's current prompt shape: no row ids, no sharing.

    This is what the existing pipeline sends, and the honest comparison point.
    """
    total = 0
    for row in rows:
        body = " | ".join(f"{h}={v}" for h, v in zip(header, row))
        total += counter(body) + q_tokens * questions_per_row
    return total, {"requests": len(rows), "rows": len(rows), "total_tokens": total}

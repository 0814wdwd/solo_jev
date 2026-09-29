"""Robustness of the offline layer against tables that are not well behaved.

Real tables carry NULLs, unicode, embedded separators, duplicate rows, single rows
that dwarf a budget, and columns with one distinct value. Each of those has already
produced a silent defect in this project once: a comma broke column alignment on
every flight row, and an empty cell was read as the value above it. These pin the
behaviour so the next one fails loudly instead.

Nothing here needs an API key or a network.
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_solo.encodings import ENCODINGS, NULL_SENTINEL, encode, encode_csv_rle
from jev_solo.objective import encode_columns, prefix_group_counts
from jev_solo.pack import STATE_BUDGET, pack_requests
from jev_solo.plan import PLANNERS, apply_order, lex_sort_rows, plan_columns
from jev_solo.recalibrate import Calibrator, fit
from jev_solo.tokens import get_counter

COUNTER = get_counter("cl100k_base")


def _parse(state, preamble_lines=1):
    """Strip the convention preamble only; row 0 of the result is the header."""
    return list(csv.reader(io.StringIO("\n".join(state.split("\n")[preamble_lines:]))))


# ---- pathological tables ------------------------------------------------

def test_all_null_column_round_trips():
    header = ["a", "b", "c"]
    rows = [["1", "", "x"], ["2", "", "y"], ["3", "", "z"]]
    state = encode_csv_rle(header, rows, row_ids=True)
    parsed = _parse(state)
    assert {len(r) for r in parsed} == {len(header) + 1}
    for i in range(len(rows)):
        assert parsed[i + 1][2] == NULL_SENTINEL, \
            "an all-empty column must stay distinguishable from ditto"


def test_duplicate_rows_stay_separately_addressable():
    header = ["a", "b"]
    rows = [["1", "x"]] * 5
    state = encode_csv_rle(header, rows, row_ids=True)
    for i in range(len(rows)):
        assert f"r{i + 1}," in state, f"duplicate row {i} lost its id"


def test_single_column_table():
    header = ["only"]
    rows = [[str(i % 3)] for i in range(50)]
    for enc in ENCODINGS:
        state = encode(header, rows, enc, row_ids=True)
        assert state, f"{enc} produced nothing for a one-column table"


def test_constant_column_has_one_group():
    rows = [["k", str(i)] for i in range(40)]
    groups = prefix_group_counts(encode_columns(rows), [0, 1])
    assert groups[0] == 1, "a constant column must form exactly one group"
    assert groups[1] == 40


def test_unicode_and_control_characters_survive():
    header = ["city", "note"]
    rows = [["北京, 中国", "行\t列"], ["Kraków", "naïve — em"], ["東京", "a\nb"]]
    state = encode_csv_rle(header, rows, row_ids=True)
    parsed = _parse(state)
    assert {len(r) for r in parsed} == {len(header) + 1}
    for i, row in enumerate(rows):
        assert parsed[i + 1][1:] == row


def test_a_row_larger_than_the_budget_is_emitted_alone_not_dropped():
    header = ["big"]
    # Random text, not a repeated character: tiktoken compresses "x" * 200000 into
    # about 25k tokens, which fits the budget and would make this test vacuous.
    rng = np.random.default_rng(7)
    huge = " ".join(str(int(v)) for v in rng.integers(100_000, 999_999, 40_000))
    rows = [[huge], ["small"], ["also small"]]
    enc = lambda h, blk: encode_csv_rle(h, blk, row_ids=True)
    reqs = list(pack_requests(header, rows, COUNTER, q_tokens=24, encode_fn=enc))
    assert sum(r.n_rows for r in reqs) == len(rows), "packing must not drop rows"
    oversized = [r for r in reqs if r.start == 0]
    assert oversized and oversized[0].n_rows == 1, \
        "an oversized row must be emitted alone rather than silently split"
    assert oversized[0].state_tokens > STATE_BUDGET, \
        "and its real size must be reported, not clipped to the budget"


def test_empty_table_is_handled():
    assert list(pack_requests(["a"], [], COUNTER)) == []
    order, meta = plan_columns([], "solo_greedy", COUNTER)
    assert order == []


# ---- planners -----------------------------------------------------------

@pytest.mark.parametrize("planner", PLANNERS)
def test_every_planner_returns_a_permutation(planner):
    rng = np.random.default_rng(0)
    header = [f"c{i}" for i in range(9)]
    rows = [[str(rng.integers(0, 4)) for _ in header] for _ in range(200)]
    order, _ = plan_columns(rows, planner, COUNTER, header=header)
    assert sorted(order) == list(range(len(header))), \
        f"{planner} did not return a permutation of the columns"


def test_planning_is_deterministic():
    rng = np.random.default_rng(1)
    header = [f"c{i}" for i in range(8)]
    rows = [[str(rng.integers(0, 5)) for _ in header] for _ in range(300)]
    a, _ = plan_columns(rows, "solo_greedy", COUNTER, header=header, seed=0)
    b, _ = plan_columns(rows, "solo_greedy", COUNTER, header=header, seed=0)
    assert a == b, "same input and seed must give the same plan"


# ---- calibration --------------------------------------------------------

def test_calibrator_refuses_single_class_labels():
    cal = fit([0.2] * 50, [1] * 50)
    assert cal.degenerate, "single-class labels carry no calibration information"
    assert cal.threshold == 0.5


def test_calibrator_is_a_noop_by_default():
    cal = Calibrator()
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(cal.probability(p), p, atol=1e-6)
    assert list(cal.verdict(p)) == [False, True, True]


def test_platt_reduces_calibration_error_on_a_biased_signal():
    rng = np.random.default_rng(2)
    n = 2000
    y = (rng.random(n) < 0.3).astype(int)
    p = np.clip(0.55 + 0.30 * y + rng.normal(0, 0.10, n), 0.01, 0.99)  # biased high
    cal = fit(p, y, platt=True)
    assert cal.ece_before is not None and cal.ece_after is not None
    assert cal.ece_after < cal.ece_before, \
        f"Platt made calibration worse: {cal.ece_before:.3f} -> {cal.ece_after:.3f}"
    assert cal.auc and cal.auc > 0.8, "the ranking should survive the correction"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

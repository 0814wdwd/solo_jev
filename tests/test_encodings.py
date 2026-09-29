"""Tests for the serialization layer.

These exist because an audit found a real defect that every measurement had been
running on: values containing commas were written unquoted, so a csv-family state
declared N header fields and emitted N+1 data fields, shifting every later column
out of alignment. On the flight table that was every single row
(`OriginCityName` = "Hartford, CT"). Nothing errored; the model just read a
table whose columns did not line up with its header.

The rule these tests encode: a state is only correct if it can be parsed back
into the rows it came from. Anything less and the model is reading something we
did not intend, silently.

Run: python3 -m pytest tests/ -q      (or: python3 tests/test_encodings.py)
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from jev_solo.encodings import (
    CSV_RLE_PREAMBLE, encode, encode_csv_block, encode_csv_rle, ENCODINGS,
)
from jev_solo.objective import encode_columns, prefix_group_counts
from jev_solo.pack import STATE_BUDGET, TOTAL_BUDGET, pack_requests
from jev_solo.plan import apply_order, lex_sort_rows, plan_columns
from jev_solo.tokens import get_counter

# Cells that break naive joining: separators, quotes, newlines, empties, unicode.
HEADER = ["Origin", "City", "Note", "Delay"]
NASTY = [
    ["BDL", "Hartford, CT", "on time", "12"],
    ["LGA", 'New "York", NY', "a | b", "0"],
    ["SFO", "San Francisco, CA", "line1\nline2", "-5"],
    ["ORD", "Chicago, IL", "", "33"],
    ["PEK", "北京, 中国", "unicode, comma", "7"],
]


def _parse_csv(state: str, has_preamble: bool) -> list[list[str]]:
    lines = state.split("\n")
    if has_preamble:
        lines = lines[1:]
    return list(csv.reader(io.StringIO("\n".join(lines))))


def test_csv_block_roundtrips():
    state = encode_csv_block(HEADER, NASTY, row_ids=True)
    parsed = _parse_csv(state, has_preamble=False)
    widths = {len(r) for r in parsed}
    assert widths == {len(HEADER) + 1}, f"ragged csv: field counts {widths}"
    for i, row in enumerate(NASTY):
        assert parsed[i + 1][0] == f"r{i + 1}"
        assert parsed[i + 1][1:] == row, f"row {i} did not round-trip"


def test_csv_rle_roundtrips_after_filling_dittos():
    rows = lex_sort_rows(NASTY)
    state = encode_csv_rle(HEADER, rows, row_ids=True)
    parsed = _parse_csv(state, has_preamble=True)
    assert {len(r) for r in parsed} == {len(HEADER) + 1}
    prev = None
    for i, row in enumerate(rows):
        cells = parsed[i + 1][1:]
        from jev_solo.encodings import NULL_SENTINEL
        cells = ["" if c == NULL_SENTINEL else c for c in cells]
        raw = parsed[i + 1][1:]
        filled = ([c if raw[j] != "" else prev[j] for j, c in enumerate(cells)]
                  if prev else cells)
        assert filled == row, f"row {i} did not reconstruct: {filled} != {row}"
        prev = filled


def test_pin_always_restates_its_columns():
    rows = lex_sort_rows(NASTY)
    pin = (0, 3)
    state = encode_csv_rle(HEADER, rows, row_ids=True, pin=pin)
    parsed = _parse_csv(state, has_preamble=True)
    for i in range(len(rows)):
        for k in pin:
            assert parsed[i + 1][1 + k] != "", \
                f"pinned column {HEADER[k]} was elided on row {i}"


def test_recommended_encodings_make_every_row_addressable():
    for enc in ("row_kv", "row_json", "csv_block", "csv_rle", "factored_rle"):
        state = encode(HEADER, NASTY, enc, row_ids=True)
        for i in range(len(NASTY)):
            assert f"r{i + 1}" in state, f"{enc} lost the id of row {i}"


def test_columnar_rle_is_known_to_be_unaddressable():
    """Documents a defect rather than asserting correctness.

    columnar_rle announces "ids r1..rN" once and then lists values column by
    column, so an individual row id never appears next to its values: the model
    has to count positions to answer a question about row 7. That is the
    structural reason it scores near chance (64.1% balanced accuracy) while
    looking cheapest on tokens, and why pinning helps it at *every* emit level
    rather than only at low ones. Pinning a column restores per-row ids for that
    column, which is what rescues it.
    """
    state = encode(HEADER, NASTY, "columnar_rle", row_ids=True)
    missing = [i for i in range(len(NASTY)) if f"r{i + 1}=" not in state]
    assert missing, "columnar_rle unexpectedly became addressable; revisit the docs"

    from jev_solo.encodings import encode_columnar_rle
    pinned = encode_columnar_rle(HEADER, NASTY, row_ids=True, pin=(3,))
    for i in range(len(NASTY)):
        assert f"r{i + 1}=" in pinned, "pinning should restore per-row addressing"


def test_runs_equal_prefix_group_counts():
    """The identity the whole cost model rests on: emitted values per column
    position equal G^(k) once rows are sorted."""
    rng = np.random.default_rng(0)
    rows = [[str(rng.integers(0, k)) for k in (2, 3, 5, 40)] for _ in range(300)]
    counter = get_counter("chars4")
    order, _ = plan_columns(rows, "solo_greedy", counter, header=["a", "b", "c", "d"])
    ordered = lex_sort_rows(apply_order(rows, order))
    groups = prefix_group_counts(encode_columns(ordered), list(range(len(order))))
    for k in range(len(order)):
        runs = 1 + sum(1 for i in range(1, len(ordered))
                       if ordered[i][k] != ordered[i - 1][k]
                       or ordered[i][:k] != ordered[i - 1][:k])
        assert runs == groups[k], f"column {k}: {runs} runs but G^(k)={groups[k]}"


def test_packing_respects_the_documented_budgets():
    rng = np.random.default_rng(1)
    header = [f"c{i}" for i in range(12)]
    rows = lex_sort_rows([[str(rng.integers(0, 50)) for _ in header] for _ in range(4000)])
    counter = get_counter("cl100k_base")
    reqs = list(pack_requests(header, rows, counter, q_tokens=24))
    assert sum(r.n_rows for r in reqs) == len(rows), "packing dropped or duplicated rows"
    assert [r.start for r in reqs] == sorted(r.start for r in reqs)
    for r in reqs:
        if r.n_rows > 1:  # a single oversized row is reported, not silently split
            assert r.state_tokens + 24 <= STATE_BUDGET, "state budget exceeded"
            assert r.total_tokens <= TOTAL_BUDGET, "total budget exceeded"


def test_compressed_encodings_are_smaller_than_the_labelled_one():
    rng = np.random.default_rng(2)
    header = [f"col{i}" for i in range(15)]
    rows = lex_sort_rows([[str(rng.integers(0, 4)) for _ in header] for _ in range(200)])
    counter = get_counter("cl100k_base")
    sizes = {e: counter(encode(header, rows, e, row_ids=True)) for e in ENCODINGS}
    assert sizes["csv_block"] < sizes["row_kv"], sizes
    assert sizes["csv_rle"] < sizes["csv_block"], sizes


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


def test_pinned_cells_carry_their_column_name():
    """Presence is not identification.

    Pinning guarantees a predicate's value is written on every row. It does not
    guarantee the model can tell which column that value belongs to: in a dittoed
    row the value sits behind a run of empty fields and has to be located by
    counting commas. Labelling closes that gap, and measured on flight it is worth
    +9.9 points of balanced accuracy on an arithmetic predicate at a 10% token cost.
    """
    rows = lex_sort_rows(NASTY)
    pin = (0, 3)
    labelled = encode_csv_rle(HEADER, rows, row_ids=True, pin=pin, label_pinned=True)
    bare = encode_csv_rle(HEADER, rows, row_ids=True, pin=pin, label_pinned=False)
    # Parse rather than split on newlines: a quoted cell may legitimately contain
    # one, and splitting would tear a valid row in half.
    parsed = _parse_csv(labelled, has_preamble=True)
    assert {len(r) for r in parsed} == {len(HEADER) + 1}, \
        f"labelling broke the field count: {[len(r) for r in parsed]}"
    for record in parsed[1:]:
        for k in pin:
            assert record[1 + k].startswith(f"{HEADER[k]}="), \
                f"pinned column {HEADER[k]} unlabelled: {record[1 + k]!r}"
    assert len(labelled) > len(bare), "labelling should cost something"

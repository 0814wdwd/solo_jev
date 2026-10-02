"""Independent regressions for integer planning and stable row materialization."""
from __future__ import annotations

import random

import numpy as np
import pytest

from jev_solo._integer import PrefixGroups
from jev_solo.encodings import ENCODINGS, encode
from jev_solo.objective import encode_columns, prefix_group_counts
from jev_solo.plan import PLANNERS, _greedy, apply_order, lex_sort_rows, plan, plan_columns
from jev_solo.tokens import get_counter, rendered_field_weights


def _groups(codes, prefix):
    # Tuple sets are an independent oracle: no packed keys, integer stamps,
    # sorting assumptions, or overflow-sensitive multiplication.
    return len({tuple(int(row[c]) for c in prefix) for row in codes})


def _reference_greedy(codes, weights=None, eps=None):
    remaining = list(range(codes.shape[1]))
    prefix = []
    while remaining:
        counts = {c: _groups(codes, prefix + [c]) for c in remaining}
        if eps is not None and weights is not None:
            threshold = min(counts.values()) * (1.0 + eps)
            chosen = max((c for c in remaining if counts[c] <= threshold),
                         key=lambda c: float(weights[c]))
        else:
            chosen = min(remaining, key=lambda c: counts[c] if weights is None
                         else counts[c] * float(weights[c]))
        prefix.append(chosen)
        remaining.remove(chosen)
    return prefix


@pytest.mark.parametrize("mode", ["solo", "weighted", "eps"])
def test_greedy_matches_tuple_set_oracle_including_ties(mode):
    rng = np.random.default_rng(7)
    for n in (0, 1, 2, 6, 30, 80):
        for m in (0, 1, 4, 9):
            codes = rng.integers(0, 5, size=(n, m), dtype=np.int64)
            weights = None if mode == "solo" else rng.integers(0, 5, m).tolist()
            eps = 0.2 if mode == "eps" else None
            assert _greedy(codes, weights, eps) == _reference_greedy(codes, weights, eps)


def test_group_scanning_counts_values_repeated_across_interleaved_prefixes():
    codes = np.array([[0, 2, 1], [1, 2, 0], [0, 1, 0], [1, 1, 1],
                      [0, 2, 0], [1, 2, 1]], dtype=np.uint8)
    state = PrefixGroups(codes)
    prefix = []
    for chosen in (0, 1, 2):
        for candidate in range(3):
            assert state.count(candidate) == _groups(codes, prefix + [candidate])
        prefix.append(chosen)
        assert state.refine(chosen) == _groups(codes, prefix)
        expected = sorted(range(len(codes)), key=lambda i: tuple(codes[i, prefix]))
        assert state.permutation.tolist() == expected


def test_core_does_not_sort_or_factorize_each_candidate(monkeypatch):
    rows = [[str(i % 3), str(i % 5), str(i % 11)] for i in range(300)]
    codes = encode_columns(rows)
    expected = _reference_greedy(codes, [3.0, 2.0, 5.0], 0.05)

    def forbidden(*args, **kwargs):
        raise AssertionError("canonical integer planning must not call np.unique")

    monkeypatch.setattr(np, "unique", forbidden)
    actual = _greedy(codes, [3.0, 2.0, 5.0], 0.05)
    assert actual == expected
    assert prefix_group_counts(codes, actual) == [
        _groups(codes, actual[:k]) for k in range(1, len(actual) + 1)
    ]


def test_sparse_and_negative_integer_inputs_use_bounded_workspace():
    codes = np.array([[-2, 2**62], [2**61, 2**62 + 1], [-2, 2**62 + 1],
                      [-2, 2**62]], dtype=np.int64)
    state = PrefixGroups(codes)
    assert state.workspace_bytes <= 40 * len(codes) + 8
    assert prefix_group_counts(codes, [1, 0]) == [_groups(codes, [1]), _groups(codes, [1, 0])]
    assert _greedy(codes, None) == _reference_greedy(codes)


def test_auxiliary_arrays_do_not_grow_with_candidate_column_count():
    narrow = np.tile(np.arange(300, dtype=np.uint16)[:, None] % 7, (1, 2))
    wide = np.tile(narrow[:, :1], (1, 200))
    assert PrefixGroups(narrow).workspace_bytes == PrefixGroups(wide).workspace_bytes


@pytest.mark.parametrize("cardinality,dtype", [
    (256, np.uint8), (257, np.uint16), (65536, np.uint16), (65537, np.uint32),
])
def test_encoding_width_boundaries_preserve_lexical_rank(cardinality, dtype):
    rows = [[str(i), "constant"] for i in range(cardinality)]
    codes = encode_columns(rows)
    assert codes.dtype == dtype
    assert codes.flags.f_contiguous
    ranks = {value: i for i, value in enumerate(sorted({row[0] for row in rows}))}
    assert codes[:, 0].tolist() == [ranks[row[0]] for row in rows]
    assert prefix_group_counts(codes, [1, 0]) == [1, cardinality]


def test_integer_sort_preserves_string_order_original_cells_and_duplicate_stability():
    rows = [["北京", "10"], ["a", "2"], ["a", "10"], [1, "x"],
            ["1", "x"], ["", "line\ncomma,"], [None, "-2"], ["a", "10"]]
    expected = sorted((list(row) for row in rows), key=lambda row: tuple(str(v) for v in row))
    assert lex_sort_rows(rows) == expected
    assert isinstance(lex_sort_rows(rows)[1][0], int)


def test_public_sort_helper_retains_ragged_sequence_support():
    rows = [["a", "x"], ["a"], [], ["a", "x", "y"], ["a", "w"]]
    expected = sorted(rows, key=lambda row: tuple(str(v) for v in row))
    assert lex_sort_rows(rows) == expected


@pytest.mark.parametrize("planner", PLANNERS)
@pytest.mark.parametrize("sample", [0, 11, 100])
def test_full_and_sampled_plans_keep_exact_order_and_serialized_input(planner, sample):
    rng = np.random.default_rng(3)
    rows = [[str(rng.integers(0, k)) for k in (2, 3, 5, 40)] for _ in range(80)]
    rows += [["rare", "北京, 中国", "", "line\nbreak"]] * 2
    header = ["a", "b", "c", "d"]
    counter = get_counter("chars4")
    sample_rows = rows
    if sample and sample < len(rows):
        sample_rows = [rows[i] for i in random.Random(9).sample(range(len(rows)), sample)]
    codes = encode_columns(sample_rows)
    if planner == "default":
        expected_order = list(range(4))
    elif planner == "random":
        expected_order = list(range(4))
        random.Random(9).shuffle(expected_order)
    elif planner == "ndv":
        expected_order = sorted(range(4), key=lambda c: _groups(codes, [c]))
    else:
        weights = None if planner == "solo_greedy" else rendered_field_weights(
            sample_rows, header, counter, sample=min(4000, len(sample_rows)), seed=9
        )
        eps = 0.05 if planner == "token_greedy_eps" else None
        expected_order = _reference_greedy(codes, weights, eps)
    ordered, order, meta = plan(rows, planner, counter, seed=9, plan_sample=sample, header=header)
    assert order == expected_order
    expected_rows = sorted(apply_order(rows, order), key=lambda r: tuple(str(v) for v in r))
    assert ordered == expected_rows
    assert meta["plan_rows"] == len(sample_rows)
    assert plan_columns(rows, planner, counter, seed=9, plan_sample=sample, header=header)[0] == order
    for encoding in ENCODINGS:
        planned_header = [header[c] for c in order]
        assert encode(planned_header, ordered, encoding) == encode(planned_header, expected_rows, encoding)


@pytest.mark.parametrize("planner", PLANNERS)
def test_empty_zero_width_and_unsorted_plans(planner):
    counter = get_counter("chars4")
    assert plan([], planner, counter)[0] == []
    assert plan([[], []], planner, counter)[0] == [[], []]
    rows = [["b", "2"], ["a", "1"]]
    ordered, order, _ = plan(rows, planner, counter, sort_rows=False)
    assert ordered == apply_order(rows, order)


def test_long_prefixes_never_pack_products_into_integer_keys():
    rng = np.random.default_rng(4)
    codes = rng.integers(0, 2, (40, 100), dtype=np.uint8)
    order = list(range(100))
    assert prefix_group_counts(codes, order) == [_groups(codes, order[:k]) for k in range(1, 101)]


@pytest.mark.parametrize("project", [False, True])
def test_scan_and_scanner_keep_pinned_fields_and_truth_aligned(project):
    from jev_solo.api import Scan
    from jev_solo.datasets import Predicate, TableSpec
    from jev_solo.pipeline import Scanner

    header = ["category", "amount", "threshold", "unused"]
    rows = [["b", "12", "10", "x"], ["a", "2", "10", "y"],
            ["b", "1", "0", "z"], ["a", "2", "10", "y"]]
    predicate = Predicate("compare", ["amount", "threshold"], "numeric",
                          "row r{rid}: amount > threshold", "r{rid}",
                          lambda values: float(values[0]) > float(values[1]))
    table = TableSpec("test", "unused.csv", {"compare": predicate},
                      use_cols=["category", "amount", "threshold"])
    counter = get_counter("chars4")
    scan = Scan(table, ["compare"], client=object(), project=project, counter=counter)
    planned, ordered, pins = scan._layout(header, rows, predicate)
    keep = [1, 2] if project else [0, 1, 2]
    sub = [[row[c] for c in keep] for row in rows]
    order = _reference_greedy(encode_columns(sub))
    expected = sorted(apply_order(sub, order), key=lambda r: tuple(str(v) for v in r))
    assert ordered == expected
    assert [planned[p] for p in pins] == predicate.cols
    truth = [int(predicate.truth([r[p] for p in pins])) for r in ordered]
    assert sum(truth) == 2

    scanner = Scanner(table, object(), counter=counter)
    planned, ordered, pins, truth = scanner._prepare(header, rows, predicate)
    assert [planned[p] for p in pins] == predicate.cols
    assert truth == [int(predicate.truth([r[p] for p in pins])) for r in ordered]
    assert sum(truth) == 2

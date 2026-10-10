import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solo_layout import LAYOUTS, LayoutOptimizer
from solo_layout._integer import PrefixGroups
from solo_layout.layout import encode_columns
from solo_layout._table import as_table


def oracle(rows):
    order, remaining = [], list(range(rows.shape[1]))
    while remaining:
        chosen = min(remaining, key=lambda c: len({tuple(row[j] for j in order + [c]) for row in rows}))
        order.append(chosen)
        remaining.remove(chosen)
    permutation = sorted(range(len(rows)), key=lambda i: tuple(rows[i, c] for c in order))
    return order, permutation


@pytest.mark.parametrize("n,m", [(0, 3), (1, 1), (40, 6), (300, 12), (20, 0)])
def test_exact_solo_matches_independent_joint_set_oracle(n, m):
    rng = np.random.default_rng(41)
    values = rng.integers(0, 8, size=(n, m)).astype(str)
    expected_cols, expected_rows = oracle(values)
    result = LayoutOptimizer().plan(values)
    assert result.column_order.tolist() == expected_cols
    assert result.row_order.tolist() == expected_rows
    assert not result.row_order.flags.writeable


def test_duplicate_rows_indices_and_cell_types_are_preserved():
    frame = pd.DataFrame({"id": [2, 1, 1, 3], "text": ["北京", "a", "a", ""]},
                         index=pd.Index(["x", "x", "z", "a"], name="source"))
    original = frame.copy(deep=True)
    plan = LayoutOptimizer().plan(frame)
    pd.testing.assert_frame_equal(plan.apply(frame), frame.iloc[plan.row_order, plan.column_order])
    pd.testing.assert_frame_equal(frame, original)
    np.testing.assert_array_equal(plan.restore(plan.row_order), np.arange(len(frame)))
    duplicate_positions = [int(i) for i in plan.row_order if i in (1, 2)]
    assert duplicate_positions == [1, 2]


def test_baselines_are_distinct_and_random_is_repeatable():
    data = np.array([["z", "1", "b"], ["a", "2", "b"], ["a", "1", "b"], ["z", "0", "b"]])
    original = LayoutOptimizer("original").plan(data)
    assert original.row_order.tolist() == list(range(4))
    lex = LayoutOptimizer("lexicographic").plan(data)
    assert lex.row_order.tolist() == [2, 1, 3, 0]
    ndv = LayoutOptimizer("cardinality").plan(data)
    assert ndv.column_order.tolist() == [2, 0, 1]
    for method in ("random", "random_columns"):
        a, b = [LayoutOptimizer(method, seed=14).plan(data) for _ in range(2)]
        np.testing.assert_array_equal(a.row_order, b.row_order)
        np.testing.assert_array_equal(a.column_order, b.column_order)


def test_equal_ndv_correlation_changes_conditional_counts():
    rows = np.array([[str(a), str(b), str(c), str((a+1)%4), str((a+2)%4)]
                     for a in range(4) for b in range(4) for c in range(4)])
    assert [len(set(rows[:, c])) for c in range(5)] == [4] * 5
    ndv = LayoutOptimizer("cardinality").plan(rows)
    solo = LayoutOptimizer().plan(rows)
    assert ndv.column_order.tolist() == [0, 1, 2, 3, 4]
    assert solo.column_order.tolist() == [0, 3, 4, 1, 2]
    state = PrefixGroups(encode_columns(rows))
    assert [state.refine(c) for c in solo.column_order] == [4, 4, 4, 16, 64]


def test_sampled_plan_uses_exact_sample_counts_then_sorts_every_row():
    import random
    rng = np.random.default_rng(31)
    rows = rng.integers(0, 200, size=(400, 5)).astype(str)
    sample = rows[random.Random(11).sample(range(400), 17)]
    order, _ = oracle(sample)
    plan = LayoutOptimizer(seed=11, sample_size=17).plan(rows)
    assert plan.column_order.tolist() == order
    assert plan.row_order.tolist() == sorted(range(400), key=lambda i: tuple(rows[i, c] for c in order))
    assert plan.plan_rows == 17


@pytest.mark.parametrize("size,dtype", [(256, np.uint8), (257, np.uint16), (65537, np.uint32)])
def test_compact_codes_retain_lexical_order_at_dtype_boundaries(size, dtype):
    rows = np.array([[str(i)] for i in range(size)], dtype=object)
    codes = encode_columns(rows)
    assert codes.dtype == dtype
    assert codes.flags.f_contiguous
    assert codes[np.argsort(rows[:, 0]), 0].tolist() == list(range(size))


def test_no_candidate_unique_or_sort(monkeypatch):
    data = np.array([[str(i%3), str(i%5), str(i%7)] for i in range(40)])
    def fail(*args, **kwargs):
        raise AssertionError("candidate refactorization regressed")
    monkeypatch.setattr(np, "unique", fail)
    LayoutOptimizer().plan(data)


@pytest.mark.parametrize("bad,columns", [([1, 2], None), ([[1], [2, 3]], None),
                                       ([[1, 2]], ["same", "same"]), ([[1, 2]], ["x"])])
def test_bad_shapes_and_names_fail_before_planning(bad, columns):
    with pytest.raises(ValueError):
        LayoutOptimizer().plan(bad, columns=columns)


def test_structured_array_and_missing_cells():
    a = np.array([(2, "b"), (1, "a")], dtype=[("id", "i4"), ("text", "U1")])
    plan = LayoutOptimizer().plan(a)
    assert plan.apply(a).dtype.names == plan.ordered_columns
    table = as_table(pd.DataFrame({"value": [None, np.nan, pd.NA, ""]}, dtype=object))
    assert table.rows[:, 0].tolist() == ["None", "nan", "<NA>", ""]


def test_record_layout_preserves_non_string_keys_and_rejects_mismatched_schema():
    records = [{1: "b", 2: "a"}, {1: "a", 2: "a"}]
    plan = LayoutOptimizer().plan(records)
    assert list(plan.apply(records)[0]) == [2, 1]
    with pytest.raises(ValueError):
        plan.apply([{"wrong": "b"}, {"wrong": "a"}])
    with pytest.raises(ValueError):
        LayoutOptimizer().plan(pd.DataFrame({"a": [1]}), columns=["renamed"])


def test_apply_accepts_the_same_mapping_records_as_plan():
    from types import MappingProxyType
    records = [MappingProxyType({"a": "z", "b": "1"}), MappingProxyType({"a": "a", "b": "2"})]
    plan = LayoutOptimizer().plan(records)
    applied = plan.apply(records)
    assert [list(row) for row in applied] == [list(plan.ordered_columns)] * 2
    assert [row["a"] for row in applied] == [records[int(i)]["a"] for i in plan.row_order]


@pytest.mark.parametrize("sample_size", [True, False, 0, -1, 2.0])
def test_sample_size_rejects_booleans_and_non_positive_values(sample_size):
    with pytest.raises(ValueError, match="sample_size"):
        LayoutOptimizer(sample_size=sample_size)

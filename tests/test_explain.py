import json
from os.path import commonprefix

import numpy as np
import pytest

from solo_layout import DecisionEngine, LayoutOptimizer, read_json


def test_prefix_group_curve_captures_correlation_beyond_marginal_ndv():
    records = [{"a": str(a), "b": str(b), "copy_a": str((a + 1) % 4)}
               for a in range(4) for b in range(4)]
    result = LayoutOptimizer().explain(records, methods=("original", "cardinality", "solo"))
    assert result.column_ndv == (4, 4, 4)
    reports = {row["method"]: row for row in result.layouts}
    assert reports["original"]["prefix_group_counts"] == [4, 16, 16]
    assert reports["cardinality"]["prefix_group_counts"] == [4, 16, 16]
    assert reports["solo"]["prefix_group_counts"] == [4, 4, 16]
    assert result.to_pandas().index.tolist() == ["original", "cardinality", "solo"]
    assert "not tokenizer output" in result.to_dict()["interpretation"]
    json.dumps(result.to_dict())


def test_byte_diagnostic_matches_independent_full_serialization_with_unicode():
    records = [{"id": 2, "name": "张三", "nested": {"x": [1, None]}},
               {"id": 1, "name": "张三", "nested": {"x": [1, None]}},
               {"id": 3, "name": "李四", "nested": {"x": [0, True]}}]
    source = read_json(records)
    methods = ("original", "lexicographic", "random", "solo")
    result = LayoutOptimizer(seed=31).explain(source, methods=methods)
    for report in result.layouts:
        plan = LayoutOptimizer(report["method"], seed=31).plan(source)
        serialized = [json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                      for row in plan.apply(source)]
        expected = sum(len(commonprefix([left, right])) for left, right in zip(serialized, serialized[1:]))
        assert report["serialized_bytes"] == sum(map(len, serialized))
        assert report["adjacent_prefix_bytes"] == expected
        assert report["adjacent_prefix_fraction"] == expected / sum(map(len, serialized))
    assert len({row["serialized_bytes"] for row in result.layouts}) == 1


def test_explain_makes_no_backend_calls():
    class NoNetwork:
        def prepare(self, *args, **kwargs):
            raise AssertionError("offline explanation prepared backend")

        def decide(self, *args, **kwargs):
            raise AssertionError("offline explanation called inference")

    with DecisionEngine(backend=NoNetwork()) as engine:
        result = engine.explain(np.array([["1", "same"], ["2", "same"]]), columns=["id", "value"])
        assert len(result.layouts) == 4


@pytest.mark.parametrize("data", [[], [[], []], [{}, {}]])
def test_empty_dimensions_have_finite_diagnostics(data):
    result = LayoutOptimizer().explain(data)
    assert all(np.isfinite(row["adjacent_prefix_fraction"]) for row in result.layouts)
    assert result.column_ndv == ()


def test_invalid_diagnostic_method_sets_fail_early():
    for methods in ((), ("solo", "solo"), ("absent",)):
        with pytest.raises(ValueError):
            LayoutOptimizer().explain([], methods=methods)

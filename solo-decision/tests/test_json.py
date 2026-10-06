import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solo_decision import DecisionEngine, DecisionResponse, JSONInput, LayoutOptimizer, read_json
from solo_decision._table import Serializer, as_table


def render(data):
    table = as_table(data)
    serializer = Serializer(table, range(table.shape[1]))
    return [json.loads(serializer(i)) for i in range(len(table))]


class JSONBackend:
    supports_cache_salt = True

    def __init__(self):
        self.rows = []

    def decide(self, state, spec, *, cache_salt=None):
        row = json.loads(state)
        self.rows.append(row)
        assert isinstance(row["amount"], int)
        return DecisionResponse((0.1, 0.9) if row["amount"] > 1 else (0.9, 0.1))


def test_json_types_nested_arrays_and_key_paths_are_preserved():
    records = [{"id": 2, "nested": {"z": None, "a": [True, 3, "4", {"x": 0}]},
                "nested.a": "literal path"},
               {"id": 1, "nested": {"a": [False, 2, "1", {"x": 9}], "z": None},
                "nested.a": "another literal"}]
    text = json.dumps(records)
    for source in (text, read_json(records), JSONInput(records)):
        assert render(source) == records
        plan = LayoutOptimizer().plan(source)
        reordered = plan.apply(source)
        assert isinstance(reordered, JSONInput)
        assert list(reordered) == [records[int(i)] for i in plan.row_order]
        # A materialized layout remains native JSON when fed back to scan.
        assert render(reordered) == list(reordered)
        np.testing.assert_array_equal(plan.restore(plan.row_order), np.arange(2))
        assert all(list(row) == list(plan.ordered_columns) for row in reordered)
    assert json.dumps(records) == text


def test_nested_object_key_order_is_canonical_but_array_order_is_not():
    source = read_json([{"cell": {"z": 1, "a": [2, 1]}},
                        {"cell": {"a": [2, 1], "z": 1}},
                        {"cell": {"a": [1, 2], "z": 1}}])
    table = as_table(source)
    assert table.rows[0, 0] == table.rows[1, 0]
    assert table.rows[0, 0] != table.rows[2, 0]
    assert render(source)[0]["cell"]["a"] == [2, 1]


def test_flat_legacy_cells_keep_existing_text_semantics():
    records = [{"amount": 2, "optional": None, "flag": True}]
    expected = [{"amount": "2", "optional": "None", "flag": "True"}]
    assert render(records) == expected
    assert render(pd.DataFrame(records)) == expected
    assert render(read_json(records)) == records
    assert render(records[0]) == records


def test_nested_cells_in_tabular_records_are_native_and_not_numpy_dimensions():
    records = [{"amount": 2, "items": [1, 2], "details": {"b": 1, "a": None}},
               {"amount": 3, "items": [2, 1], "details": {"a": True, "b": 3}}]
    expected = [{**row, "amount": str(row["amount"])} for row in records]
    assert render(records) == expected
    assert render(pd.DataFrame(records)) == expected
    assert render([{"items": [1, 2]}, {"items": [3, 4]}]) == [
        {"items": [1, 2]}, {"items": [3, 4]}]


def test_json_and_jsonl_paths_and_explicit_jsonl_text(tmp_path):
    records = [{"amount": 2, "payload": [False, None]}, {"amount": 1, "payload": [True, 4]}]
    ordinary = tmp_path / "rows.json"
    ordinary.write_text(json.dumps(records), encoding="utf-8")
    lines = "\n" + "\n\n".join(json.dumps(row) for row in records) + "\n"
    line_path = tmp_path / "rows.jsonl"
    line_path.write_text(lines, encoding="utf-8")
    assert render(ordinary) == render(line_path) == render(read_json(lines, lines=True)) == records
    assert len(LayoutOptimizer().plan(ordinary).apply(ordinary)) == 2
    with pytest.raises(ValueError):
        read_json(str(ordinary))  # filenames must be explicit Path objects
    with pytest.raises(ValueError, match="line 2"):
        read_json('{"a":1}\n[1]', lines=True)


@pytest.mark.parametrize("source", ['{"x":1,"x":2}', '{"x":{"a":1,"a":2}}',
                                     '[{"x":NaN}]', '[{"x":Infinity}]', '[{"x":1e999}]',
                                     [{"x": float("inf")}], [{"x": {1: "bad key"}}]])
def test_invalid_json_is_rejected_before_any_inference(source):
    with pytest.raises((TypeError, ValueError)):
        read_json(source)


@pytest.mark.parametrize("source", ['[1,2]', '[[1,2]]', '"text"', 'null',
                                     '[{"a":1},{"b":2}]', '[{"a":null},{}]'])
def test_json_schema_is_explicit_and_missing_is_not_null(source):
    with pytest.raises(ValueError):
        read_json(source)
    assert render('{"a":null}') == [{"a": None}]


def test_json_input_does_not_alias_caller_nested_values():
    records = [{"a": [1, {"b": 2}]}]
    source = read_json(records)
    records[0]["a"][1]["b"] = 100
    assert render(source) == [{"a": [1, {"b": 2}]}]


def test_scan_and_compare_json_restore_original_order(tmp_path):
    records = [{"amount": 3, "meta": {"kind": "same"}},
               {"amount": 1, "meta": {"kind": "same"}},
               {"amount": 2, "meta": {"kind": "other"}}]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(records))
    backend = JSONBackend()
    with DecisionEngine(backend=backend, concurrency=2) as engine:
        result = engine.scan(path, "amount exceeds one?")
        assert result.decisions.tolist() == [True, False, True]
        compared = engine.compare(path, "amount exceeds one?", methods=("original", "solo"),
                                  repeats=2, truth=[True, False, True])
        assert all(row["rows"] == 3 and row["accuracy"] == 1 for row in compared.summary)
        assert engine.scan(records[0], "amount exceeds one?").decisions.tolist() == [True]


def test_empty_json_and_empty_objects():
    assert as_table("[]").shape == (0, 0)
    assert as_table("[]", columns=["a"]).shape == (0, 1)
    assert render("[{},{}]") == [{}, {}]
    assert isinstance(LayoutOptimizer().plan("[]").apply("[]"), JSONInput)
    assert len(LayoutOptimizer().plan("[]").apply("[]")) == 0
    assert len(read_json("\n", lines=True)) == 0

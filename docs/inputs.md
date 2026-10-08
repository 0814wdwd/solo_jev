# Input formats

Every scan produces one decision per record. The engine reorders top-level fields
and rows, then restores the results to their original positions. It sends every
field; field selection and lossy compression are outside this interface.

| Input | Example | Record interpretation |
| --- | --- | --- |
| Pandas DataFrame | `engine.scan(df, question)` | One row per decision; original index retained |
| 2D NumPy array | `engine.scan(a, question, columns=["policy", "request"])` | One array row per decision |
| Structured NumPy array | `engine.scan(a, question)` | Named fields supply column names |
| List of record dictionaries | `engine.scan(records, question)` | Shared, strict top-level schema |
| JSON text | `engine.scan('[{"id":1,"request":"refund"}]', question)` | Array of objects, or one object |
| JSON file | `engine.scan(Path("tickets.json"), question)` | Array of objects, or one object |
| JSONL file | `engine.scan(Path("tickets.jsonl"), question)` | One object per nonblank line |
| Parsed JSON | `engine.scan(read_json(records), question)` | Preserve native JSON scalar types |

Use `pathlib.Path` for a file path. A string is interpreted as JSON text, not a
filename. `read_json(source, lines=True)` explicitly parses JSONL text. The helper
also accepts parsed JSON objects or lists of objects and returns a reusable
`JSONInput` value.

## JSON and nested values

```python
from solo_layout import DecisionEngine, read_json

records = read_json([
    {"id": 104, "customer": {"tier": "gold", "region": "north"},
     "tags": ["delivery", "refund"], "request": "Please refund this order.",
     "approved": None},
    {"id": 105, "customer": {"tier": "gold", "region": "north"},
     "tags": ["exchange"], "request": "I need another size.",
     "approved": None},
])
with DecisionEngine() as engine:
    result = engine.scan(records, "Does the request ask for a monetary refund?")
```

JSON numbers, booleans and null remain JSON numbers, booleans and null in the
model input. Nested objects and arrays stay nested: the planner treats each
whole nested value as one field. Nested object keys are serialized canonically;
array element order is preserved. The engine does not flatten objects into
potentially colliding dotted names or sort array elements.

Each record must have the same set of top-level keys. Missing keys raise an
error; explicit `null` is a value. Duplicate JSON object keys, non-string object
keys and non-finite numbers (`NaN`, `Infinity`) are rejected. These rules make
ambiguities visible before inference.

## Pandas, NumPy and the original record interface

For compatibility with the measured table benchmarks, scalar cells in
DataFrames, NumPy arrays and direct lists of dictionaries use `str(value)` and
are serialized as JSON strings. For example, an integer `1` becomes `"1"` and
`None` becomes `"None"`. Nested dict/list cells are emitted as nested canonical
JSON instead of Python representations. Use `read_json(parsed_records)` when
native JSON scalar types are required.

Column names must be unique after string conversion. Explicit `columns=` for a
named input must match its existing names; it is not a projection or renaming
operation. An unnamed 2D array defaults to `c0`, `c1`, and so on. Arrays must be
rectangular. Structured arrays must be one-dimensional.

The input is not mutated. Results preserve row position even when values or
DataFrame index labels repeat. `result.to_pandas()` retains the original index;
`result.decisions` and `result.probabilities` are NumPy arrays in input order.

## Files are materialized

`plan.apply` returns a `JSONInput` wrapper for native JSON sources, so scanning
the reordered input again preserves its numbers, booleans and nulls. Convert it
with `list(...)` when you need plain records for display.

JSON and JSONL inputs are loaded into memory to build a global row/column plan.
JSONL support does not imply streaming inference. For datasets larger than
memory, partition them explicitly and measure how partitioning changes available
reuse. The `sample_size` planning option limits rows used to choose columns, but
still retains and sorts the complete input table.

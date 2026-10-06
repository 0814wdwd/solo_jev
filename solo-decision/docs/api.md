# API reference

## DecisionEngine

```python
DecisionEngine(
    base_url="http://127.0.0.1:8000", model="jev-decision",
    model_dir=None, concurrency=4, api_key=None, timeout=180, backend=None,
)
```

Use `with DecisionEngine(...) as engine:` or call `engine.close()`. An engine
reuses its thread pool and HTTP sessions across scans. Scans on one engine are
serialized; requests within a scan run concurrently.

`model_dir` points to a directory containing `adapter_vllm/decision_head.json`
and `calibration.json`. With the `hub` extra installed, omitting it
fetches only these files from the pinned AutoTrust JEV-9B revision on first use.

### scan

```python
result = engine.scan(
    data, question, columns=None, method="solo", kind="noul",
    options=None, seed=0, sample_size=None, cache_salt=None,
)
```

| Decision kind | Output | Invocation |
| --- | --- | --- |
| `noul` (default) | `False` or `True` | `engine.scan(data, "Is the request a refund?")` |
| `choice` | One option string, 2–16 options | `engine.scan(data, "Which queue?", kind="choice", options=["billing", "technical", "sales"])` |
| `score` | Integer from 0 to 5 | `engine.scan(data, "Rate urgency from 0 to 5", kind="score")` |

`result.decisions` is a NumPy array in original row order.
`result.probabilities` has shape `(rows, len(result.options))` and the same row
order. `result.to_pandas()` includes the decision and a probability column for
each option, with the original DataFrame index where present. `result.metrics()`
reports wall time, planning/preparation/inference time, throughput, latencies,
input tokens and cache usage. Unknown token/cache metadata is `None`.

### compare

```python
comparison = engine.compare(
    data, question,
    methods=("original", "lexicographic", "random", "cardinality", "solo"),
    repeats=2, truth=None, columns=None, kind="noul", options=None,
    seed=42, sample_size=None,
)
print(comparison.to_pandas())
```

JSON/JSONL parsing occurs once before trials; each timed trial includes table
normalization, planning and request serialization. Each trial receives a fresh
cache namespace. Order reverses on alternating repeats. All methods use the same records and backend. `truth`, when supplied,
contains one label per original input row; it is used only for evaluation.
`comparison.runs` contains per-trial metrics; `comparison.summary` uses median
elapsed time. Throughput is row count divided by median elapsed time.

`speedup_vs_original` compares each method with original order.
`solo_speedup_vs_this_method` compares SOLO with that method. Accuracy against
`truth` and agreement with original-order predictions are distinct metrics.
Comparison requires a backend that supports cache namespace isolation and an
otherwise idle server; namespaces do not isolate GPU contention.

## LayoutOptimizer

```python
optimizer = LayoutOptimizer("solo", seed=42, sample_size=None)
plan = optimizer.plan(data, columns=None)
ordered_data = plan.apply(data)
original_order_outputs = plan.restore(outputs_in_planned_order)
```

| Method | Column order | Row order |
| --- | --- | --- |
| `original` | Input order | Input order |
| `lexicographic` | Input order | Stable string-lexicographic sort |
| `random` | Seeded random permutation | Seeded random permutation |
| `random_columns` | Seeded random permutation | Stable lexicographic sort |
| `cardinality` | Ascending marginal NDV; input order breaks ties | Stable lexicographic sort |
| `solo` | Greedy minimum distinct prefix combinations | Stable lexicographic sort |

`plan.apply` retains Pandas and NumPy containers. For native JSON input it
returns a `JSONInput` wrapper, preserving scalar types if scanned again; use
`list(ordered_data)` to display plain records.

`plan.row_order` and `plan.column_order` are immutable permutation arrays.
`plan.ordered_columns` contains the resulting field names. Sampling limits the
rows used to select a column order; all rows still appear in the final plan.

### explain

```python
report = LayoutOptimizer().explain(data)
print(report.to_pandas())
print(report.to_dict())
```

`engine.explain(data, methods=(...))` provides the same offline diagnostics.
Reports contain field NDVs, prefix-group counts and adjacent shared-prefix bytes
for each requested layout. Bytes describe serialization structure; they are not
cached-token counts or predicted speedups. No model request is required.

## Custom backends

Pass `backend=your_backend` to `DecisionEngine`. Implement the following method:

```python
from solo_decision import DecisionResponse

class MyBackend:
    def decide(self, state, spec, *, cache_salt=None):
        # state: the complete serialized record; spec: question and output options
        probabilities = call_your_service(state, spec)
        return DecisionResponse(tuple(probabilities))
```

Return probabilities in `spec.options` order. Calls to `decide` can run
concurrently, so the backend must be thread-safe. Optional `prepare(spec)` and
`close()` hooks support setup and cleanup. Set `supports_cache_salt=True` only if
the service actually implements cache-namespace isolation; otherwise `compare`
rejects the backend.

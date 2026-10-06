# Reproduce the measurements

Two synthetic workloads demonstrate different sources of reusable input
structure. Both run complete records through a real decision model. Their
results should be interpreted within the measured distributions and serving
configuration.

## Common protocol

| Setting | Value |
| --- | --- |
| GPU | One NVIDIA RTX 4090, 24 GB |
| Model | AutoTrust JEV-9B, BF16 backbone plus decision adapter |
| Model revision | `b63f651ce8ed64481d3f5e73ecdb05f740042f01` |
| vLLM | 0.31.0, eager execution |
| Prefix cache | Enabled, hybrid cache mode `align` |
| Context limit | 16,384 tokens |
| Max server sequences / client concurrency | 4 / 4 |
| Max batched tokens | 2,048 |
| Measured prefix-cache block size | 528 tokens |
| Records per case | 128 |
| Trials per method | 2; reverse method order on second trial |
| Random seed | 42 |
| Task | Does `customer_request` ask for a monetary refund? |

The server configuration is recorded in
[validation/server-configuration.json](../validation/server-configuration.json).
The separate deployment helper reproduces the chosen serving flags. Its
installation/runtime lifecycle checks are distinct from model benchmark results.

The throughput runs used the initial packaged client before JSON support was
extended. The release [compatibility audit](../validation/layout-compatibility.json)
checks all four generated cases across five layouts: row/column permutations and
serialized full-record requests are byte-identical to that measured version.
This verifies input compatibility; it is not another set of throughput trials.
[Benchmark provenance](../validation/benchmark-provenance.json) records the
measured client and the release compatibility checks.

For each case, the demo constructs one fixed dataset and evaluates layouts over
those same complete records. It warms the planner signature and model separately,
then assigns a new UUID `cache_salt` to each measured scan. Thus each scan pays its
first cache fills, and one method cannot reuse another method's cached prefixes.
This isolates cache matches, not contention: use an otherwise idle GPU server.

End-to-end wall time includes normalization, planning, serialization, HTTP and
inference. Per-method summary throughput is `rows / median(wall_seconds)`, not the
median of per-run throughputs. Individual trials are retained in each report and
shown as points on throughput bars. Two repeats show variation but do not provide
a broad statistical confidence claim.

Each report includes accuracy against generated labels and agreement with
original-order predictions. These labels are direct, unambiguous refund/exchange
messages; 100% on these tasks is not evidence of general semantic accuracy.
Changing field order can change predictions on another task.

## Demo 1: repeated context

Each row has an ID, a customer request and eight reusable fields. The original
serialization puts the unique ID first. Six reusable fields have 132 tokens each
and NDVs 4, 8, 16, 32, 64 and 64. Two fields are global constants whose lengths
increase across the sweep. The ID plus customer request occupy 264 value tokens.
Neutral padding is intentional and retained in every request.

The reusable fraction is:

```text
sum of cell-value tokens in columns with NDV < number of rows
------------------------------------------------------------
                 sum of all cell-value tokens
```

It excludes JSON labels and prompt formatting. It is neither the proportion of
duplicate rows nor the server's cache-hit rate. The 97% and 98.3% cases represent
especially long shared-context workloads, rather than ordinary 80/20 data.

```bash
python -m pip install ".[demo]"
python demo.py --model-dir /path/to/JEV-9B --workload shared \
  --rows 128 --repeat-ratios .80 .97 .983 \
  --methods original solo --repeats 2 --out validation/shared-live.json
```

![Repeated-context results](../assets/shared-benchmark.svg)

[Raw observations](../validation/shared-live.json) · [Figure PDF](../assets/shared-benchmark.pdf)

The measured speedups over original order are **1.49×, 7.90× and 11.71×**
for the 80%, 97% and 98.3% cases. Their maximum prompts contain 1,401, 8,881
and 15,611 tokens respectively. All six measured method/case combinations
reach 100% accuracy on the generated task.

The checked-in sweep compares original order with SOLO. To measure all baselines
on the same workload, replace `--methods original solo` with
`--methods original lexicographic random cardinality solo`. This takes longer,
especially on the longest contexts. The correlated demo below supplies the
five-method comparison already measured.

An earlier standalone harness measured a 9.26× result on a different long-context
dataset. The current README figures use the packaged demo's own measurements;
the earlier result is not relabeled or combined with this sweep.

## Demo 2: equal marginal statistics, correlated columns

This dataset contains three balanced latent variables. Twelve columns are
bijective encodings of the first variable; `customer_request` and one other field
use the second and third. Correlated columns are interleaved with the others in
the input. All fields have eight distinct values and exactly 128 value tokens;
all 128 complete records are distinct.

Marginal NDV cannot distinguish these columns. In this implementation, ties
retain the input column order. With that order, distinct prefix counts become
`8, 56, 128, ...`. SOLO holds the first twelve prefix counts at `8`, then increases
to `56` and `128`. All five layouts send 244,352 input tokens per scan; the server
can reuse more of those tokens under the SOLO layout.

```bash
python demo.py --model-dir /path/to/JEV-9B --workload correlated \
  --rows 128 --cardinality 8 --correlated-columns 12 --field-tokens 128 \
  --repeats 2 --out validation/correlated-live.json
```

![Correlated-field results](../assets/correlated-benchmark.svg)

[Raw observations](../validation/correlated-live.json)

This distribution deliberately highlights the difference between marginal and
joint statistics. It does not establish that NDV is always weak. Its tie rule
matters: the optional `random_columns` baseline randomizes column order and then
sorts rows, while `random` shuffles both rows and columns. Use additional seeds
and distributions for conclusions beyond this demonstration.

## Inspect without inference

```bash
python demo.py --model-dir /path/to/JEV-9B --workload correlated \
  --field-tokens 128 --dry-run --out /tmp/correlated-dry-run.json
```

Dry runs need the local tokenizer and installed dependencies, but make no model
API calls. They report dataset hashes, field lengths, NDVs, prefix counts and
maximum prompt length. Actual requests fail early if the demo detects a prompt
above its configured context limit. A higher context setting must also be
supported by the server and available GPU memory.

For a completely model-free example, run `python examples/offline_layout.py`.
Its byte-level layout diagnostics do not need a tokenizer.

## Regenerate figures

```bash
python -m pip install ".[plot]"
python scripts/render_benchmarks.py
```

The renderer reads the checked-in JSON reports and writes SVG, PNG and PDF
versions of both benchmark figures to `assets/`. It does not rerun inference.
Raw reports retain unrounded values; README tables round only for presentation.

## Where results may differ

Benefits depend on field repetition and correlation, field lengths, row count,
cache capacity and eviction, server block size, concurrency, warmup and model
behavior. Synthetic padding emphasizes prefill cost. Whole-field NDV also misses
partial text prefixes shared by otherwise distinct values. Report quality and
full serving conditions alongside speedups, and measure complete task time on
your own data before extrapolating to production.

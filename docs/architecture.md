# Architecture and planner provenance

The package has three boundaries: input normalization, layout planning, and a
concurrent decision client. A custom model backend can reuse the first two.

```text
DataFrame / NumPy / records / JSON / JSONL
                    │
            normalized full records
                    │
      LayoutOptimizer → row + column permutations
                    │
       bounded serialization and HTTP requests
                    │
        JEV decision head on a vLLM server
                    │
      decisions + probabilities in original order
```

## What the planner optimizes

For a column order, let G(k) be the number of distinct value combinations in its
first k columns. SOLO greedily chooses the next column that minimizes the new
prefix count, then stably groups rows under the chosen order. Correlated columns
can extend a prefix without increasing its group count, even when every column
has the same marginal number of distinct values.

A field's original value is still present in every request. Prefix caching is
performed by the server against matching token prefixes; the planner creates
layouts with more opportunities for those matches. This does not require a
compressed representation or a batch of many records in one model prompt.

The objective uses prefix-combination counts. It is not a full simulation of the
server's tokenizer, token blocks, eviction or GPU scheduler. Different field
lengths and serving conditions can affect which layout is fastest. Use
`explain()` for structural inspection and `compare()` for actual execution.

## Performance implementation

The integer grouping kernel is copied unchanged from the supplied SOLO source
snapshot's **“Optimize SOLO planning with reusable integer stamps”** PR #1. See
[planner-provenance.json](../planner-provenance.json) for its SHA256 hashes and the
reported upstream commit. The snapshot itself contained no Git history.

After reading and lexically encoding each column's distinct strings:

- Greedy column planning takes O(NM²) time.
- Fixed-order row partitioning takes O(NM) time.
- Grouping uses O(N + M) auxiliary space, beyond the O(NM) encoded table.
- Integer encoding chooses the smallest shared unsigned type needed by column
  cardinalities. Candidate scans reuse a value-indexed stamp array.

These bounds cover integer grouping. Normalization and dictionary encoding also
read the original input and sort unique values. The first Numba invocation may
compile or load a cached signature; benchmark warmup is explicit.

The engine consumes permutations directly, instead of materializing a reordered
table for every scan. It submits at most `concurrency` in-flight requests, reuses
HTTP connections and bounds the JSON escaping cache. Original row positions are
carried through execution, so identical rows and duplicate DataFrame index
labels do not require identity reconstruction.

## Decision backend

JEV-9B's adapter exposes constrained decision slots. `JevBackend` sends one
completion request per complete record, requests one output token and the
required decision-slot log probabilities, and applies the pinned model's bias
and temperature calibration. Binary, categorical and score modes share this
transport.

The client needs only `decision_head.json` and `calibration.json`; it does not
load the backbone or tokenizer. The model server uses the pinned backbone and
adapter together. A compatible OpenAI-style endpoint alone is not sufficient to
claim compatibility with these JEV-specific decision semantics.

Missing decision slots, malformed probabilities or HTTP errors raise an error.
A failed request reports its original row position and is not silently turned
into a negative decision. The client drains active requests before returning an
error. Use the engine as a context manager to close its executor and connections.

## Scope

This release implements a client library and a single tested model backend. The
standalone layout plan is reusable with other services, but each backend's
prefix-cache behavior and decision quality need separate validation. The original
research probes and calibration experiments remain outside the client wheel.

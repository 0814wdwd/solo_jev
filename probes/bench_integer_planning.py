"""Bounded, single-process comparison against the pre-optimization planner.

Run: python probes/bench_integer_planning.py --out results/integer_planning.json
The default 20,000 x 20 tables need no API or external dataset. This command
limits its address space to 2 GiB and refuses tables larger than 20,000 x 40.
Reported steady-state times exclude the separately recorded first new call.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path


def _legacy_columns(rows, weights, eps):
    """Sorting-based implementation from commit 65b0a21, including inverse copies."""
    import numpy as np

    n, m = len(rows), len(rows[0])
    codes = np.empty((n, m), dtype=np.int64)
    for c in range(m):
        _, inverse = np.unique(np.asarray([str(row[c]) for row in rows], dtype=object),
                               return_inverse=True)
        codes[:, c] = inverse
    groups = np.zeros(n, dtype=np.int64)
    remaining = list(range(m))
    order = []
    while remaining:
        best_col = best_score = best_ids = None
        scored = []
        for c in remaining:
            combined = groups * (int(codes[:, c].max()) + 1) + codes[:, c]
            unique, inverse = np.unique(combined, return_inverse=True)
            count = int(unique.size)
            ids = inverse.astype(np.int64)
            score = count if weights is None else count * float(weights[c])
            if eps is not None:
                scored.append((c, count, ids))
            if best_score is None or score < best_score:
                best_col, best_score, best_ids = c, score, ids
        if eps is not None:
            threshold = min(g for _, g, _ in scored) * (1.0 + eps)
            best_col, _, best_ids = max(
                (entry for entry in scored if entry[1] <= threshold),
                key=lambda entry: float(weights[entry[0]]),
            )
        order.append(best_col)
        remaining.remove(best_col)
        groups = best_ids
    return order


def _legacy_plan(rows, planner, counter, header):
    from jev_solo.tokens import rendered_field_weights

    weights = None if planner == "solo_greedy" else rendered_field_weights(
        rows, header, counter, sample=min(4000, len(rows)), seed=0
    )
    order = _legacy_columns(rows, weights, 0.05 if planner == "token_greedy_eps" else None)
    ordered = [[row[c] for c in order] for row in rows]
    ordered = sorted((list(row) for row in ordered),
                     key=lambda row: tuple(str(value) for value in row))
    return ordered, order


def _measure(fn, repeats):
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - started)
        del result
    gc.collect()
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return {"median_seconds": statistics.median(times), "peak_traced_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-limit-mib", type=int, default=2048)
    parser.add_argument("--out")
    args = parser.parse_args()
    if not 1 <= args.rows <= 20000 or not 1 <= args.cols <= 40:
        parser.error("use 1..20000 rows and 1..40 columns")
    if not 1 <= args.repeats <= 10 or not 64 <= args.memory_limit_mib <= 2048:
        parser.error("use 1..10 repeats and a 64..2048 MiB memory cap")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    _, inherited_hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = args.memory_limit_mib * 1024**2
    if inherited_hard != resource.RLIM_INFINITY:
        limit = min(limit, inherited_hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numba
    import numpy as np
    from jev_solo._integer import PrefixGroups
    from jev_solo.objective import encode_columns
    from jev_solo.plan import plan
    from jev_solo.tokens import get_counter

    rng = np.random.default_rng(20261002)
    header = [f"c{c}" for c in range(args.cols)]
    counter = get_counter("chars4")
    report = {"baseline_commit": "65b0a210866302d8c1c68738a14f222804d80bee",
              "numpy": np.__version__, "numba": numba.__version__,
              "rows": args.rows, "cols": args.cols, "repeats": args.repeats,
              "address_space_limit_bytes": limit, "cases": []}
    for kind in ("mixed_cardinality", "correlated"):
        latent = rng.integers(0, 1000, args.rows)
        values = np.empty((args.rows, args.cols), dtype=np.int64)
        for c in range(args.cols):
            cardinality = (2, 3, 5, 10, 100)[c % 5]
            values[:, c] = (rng.integers(0, cardinality, args.rows) if kind == "mixed_cardinality"
                            else latent % cardinality)
        rows = [[str(value) for value in row] for row in values]
        del values
        for planner in ("solo_greedy", "token_greedy_eps"):
            old_fn = lambda: _legacy_plan(rows, planner, counter, header)
            new_fn = lambda: plan(rows, planner, counter, plan_sample=0, header=header)[:2]
            expected = old_fn()
            started = time.perf_counter()
            actual = new_fn()
            first_seconds = time.perf_counter() - started
            assert actual == expected, "column order or stable row order changed"
            del expected, actual
            old = _measure(old_fn, args.repeats)
            new = _measure(new_fn, args.repeats)
            codes = encode_columns(rows)
            case = {"kind": kind, "planner": planner, "outputs_identical": True,
                    "legacy": old, "integer": new, "first_integer_call_seconds": first_seconds,
                    "speedup": old["median_seconds"] / new["median_seconds"],
                    "encoded_table_bytes": codes.nbytes, "encoded_dtype": str(codes.dtype),
                    "legacy_encoded_table_bytes": len(rows) * len(header) * 8,
                    "integer_workspace_bytes": PrefixGroups(codes).workspace_bytes}
            del codes
            report["cases"].append(case)
            print(json.dumps(case), flush=True)
    # Linux reports ru_maxrss in KiB. The test machine and this benchmark are Unix.
    report["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"process_peak_rss_kib": report["process_peak_rss_kib"]}), flush=True)


if __name__ == "__main__":
    main()

"""Permutation-only layouts using SOLO's optimized integer grouping kernel."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from os import PathLike
import random
import time

import numpy as np

from ._integer import PrefixGroups, code_dtype
from ._table import JSONInput, Serializer, Table, as_table, read_json

LAYOUTS = ("original", "lexicographic", "random", "random_columns", "cardinality", "solo")


def encode_columns(rows: np.ndarray) -> np.ndarray:
    """One lexical encoding per column; compact shared dtype, column-major."""
    n, m = rows.shape
    codes = np.empty((n, m), dtype=np.uint8, order="F")
    for c in range(m):
        values = sorted(set(rows[:, c]))
        dtype = code_dtype(len(values))
        if dtype.itemsize > codes.dtype.itemsize:
            expanded = np.empty((n, m), dtype=dtype, order="F")
            expanded[:, :c] = codes[:, :c]
            codes = expanded
        lookup = {value: i for i, value in enumerate(values)}
        codes[:, c] = np.fromiter((lookup[v] for v in rows[:, c]), dtype=codes.dtype, count=n)
    return codes


@dataclass(frozen=True)
class LayoutPlan:
    method: str
    row_order: np.ndarray
    column_order: np.ndarray
    columns: tuple[str, ...]
    planning_seconds: float
    plan_rows: int
    encoded_bytes: int = 0
    workspace_bytes: int = 0

    @property
    def ordered_columns(self):
        return tuple(self.columns[int(c)] for c in self.column_order)

    def restore(self, values):
        """Restore outputs supplied in planned row order to input row order."""
        values = np.asarray(values)
        if values.ndim == 0 or len(values) != len(self.row_order):
            raise ValueError("one output is required per input row")
        result = np.empty_like(values)
        result[self.row_order] = values
        return result

    def apply(self, data):
        """Materialize a layout when needed; scans use permutations directly."""
        native_json = isinstance(data, (JSONInput, str, PathLike, Mapping))
        if native_json:
            data = read_json(data)
        if len(data) != len(self.row_order):
            raise ValueError("plan and input must have the same row count")
        if hasattr(data, "iloc"):
            if tuple(map(str, data.columns)) != self.columns:
                raise ValueError("plan and DataFrame columns differ")
            return data.iloc[self.row_order, self.column_order].copy()
        if isinstance(data, np.ndarray) and data.dtype.names:
            if tuple(data.dtype.names) != self.columns:
                raise ValueError("plan and structured-array columns differ")
            return data[self.row_order][list(self.ordered_columns)].copy()
        if len(data) and isinstance(data[0], dict):
            keys = list(data[0])
            if tuple(map(str, keys)) != self.columns or any(set(row) != set(keys) for row in data):
                raise ValueError("plan and record fields differ")
            records = [{keys[int(c)]: data[int(i)][keys[int(c)]]
                        for c in self.column_order} for i in self.row_order]
            return JSONInput(records) if native_json else records
        if not len(data) and not isinstance(data, np.ndarray):
            return JSONInput([]) if native_json else []
        values = np.asarray(data)
        if values.ndim != 2 or values.shape[1] != len(self.columns):
            raise ValueError("plan and array shape differ")
        return values[np.ix_(self.row_order, self.column_order)]


@dataclass
class LayoutExplanation:
    """Offline structural evidence; byte reuse is not a cache-hit prediction."""
    rows: int
    columns: tuple[str, ...]
    column_ndv: tuple[int, ...]
    layouts: list[dict]

    def to_dict(self):
        return {
            "rows": self.rows,
            "columns": list(self.columns),
            "column_ndv": dict(zip(self.columns, self.column_ndv)),
            "layouts": self.layouts,
            "measurement": "UTF-8 bytes in serialized row states, excluding question/template",
            "interpretation": (
                "Adjacent shared prefixes describe input structure, not tokenizer output, "
                "vLLM cache hits, or a throughput prediction. Server block size, eviction "
                "and concurrency affect realized reuse."
            ),
        }

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self.layouts).set_index("method")


class LayoutOptimizer:
    def __init__(self, method="solo", *, seed=0, sample_size=None):
        if method not in LAYOUTS:
            raise ValueError(f"unknown layout {method!r}; choose from {LAYOUTS}")
        if sample_size is not None and (not isinstance(sample_size, int) or sample_size < 1):
            raise ValueError("sample_size must be a positive integer or None")
        self.method, self.seed, self.sample_size = method, seed, sample_size

    def plan(self, data, *, columns=None) -> LayoutPlan:
        return self._plan(as_table(data, columns))

    def explain(self, data, *, columns=None, methods=None) -> LayoutExplanation:
        """Compare complete serialized inputs without calling a model.

        By default compare original order with this optimizer's method. The
        prefix-group curve is G^(k), independent of row order; adjacent-prefix
        bytes also reflect the actual request order. Both include full rows.
        Beyond planning, grouping uses O(NM) work per method, and rendering
        reads each serialized byte once plus adjacent-prefix comparisons.
        """
        methods = tuple(dict.fromkeys(("original", self.method))) if methods is None else tuple(methods)
        if not methods or len(set(methods)) != len(methods):
            raise ValueError("methods must be nonempty and unique")
        optimizers = [LayoutOptimizer(method, seed=self.seed, sample_size=self.sample_size)
                      for method in methods]
        table = as_table(data, columns)
        codes = encode_columns(table.rows)
        ndv = tuple(int(codes[:, c].max()) + 1 if len(codes) else 0 for c in range(codes.shape[1]))
        reports = []
        for optimizer in optimizers:
            plan = optimizer._plan(table)
            groups = PrefixGroups(codes)
            counts = [groups.refine(int(c)) for c in plan.column_order]
            render = Serializer(table, plan.column_order)
            previous, total, shared = b"", 0, 0
            for row in plan.row_order:
                current = render(int(row)).encode("utf-8")
                total += len(current)
                common = 0
                for left, right in zip(previous, current):
                    if left != right:
                        break
                    common += 1
                shared += common
                previous = current
            reports.append({
                "method": plan.method,
                "column_order": list(plan.ordered_columns),
                "prefix_group_counts": counts,
                "serialized_bytes": total,
                "adjacent_prefix_bytes": shared,
                "adjacent_prefix_fraction": shared / total if total else 0.0,
                "planning_seconds": plan.planning_seconds,
                "plan_rows": plan.plan_rows,
            })
        return LayoutExplanation(len(table), table.columns, ndv, reports)

    def _plan(self, table: Table) -> LayoutPlan:
        started = time.perf_counter()
        n, m = table.shape
        order = list(range(m))
        permutation = np.arange(n, dtype=np.intp)
        planned_rows, encoded_bytes, workspace_bytes = n, 0, 0
        rng = random.Random(self.seed)
        if self.method == "random":
            rng.shuffle(order)
            rows = list(range(n))
            rng.shuffle(rows)
            permutation = np.asarray(rows, dtype=np.intp)
        elif self.method != "original" and n and m:
            codes = encode_columns(table.rows)
            encoded_bytes = codes.nbytes
            planned = codes
            if self.sample_size and self.sample_size < n and self.method in ("solo", "cardinality"):
                planned = np.asfortranarray(codes[rng.sample(range(n), self.sample_size)])
            planned_rows = len(planned)
            state = None
            if self.method == "solo":
                state = PrefixGroups(planned)
                remaining, order = list(range(m)), []
                while remaining:
                    # Python's stable min preserves the old column tie break.
                    chosen = min(remaining, key=state.count)
                    order.append(chosen)
                    remaining.remove(chosen)
                    state.refine(chosen)
            elif self.method == "cardinality":
                # A row sample can omit integer IDs, so max+1 is not its NDV.
                ndv = [len(set(planned[:, c])) for c in range(m)]
                order.sort(key=lambda c: ndv[c])
            elif self.method == "random_columns":
                rng.shuffle(order)
            if state is None or planned_rows != n:
                state = PrefixGroups(codes)
                for c in order:
                    state.refine(c)
            permutation = state.permutation
            workspace_bytes = state.workspace_bytes
        column_order = np.asarray(order, dtype=np.intp)
        permutation.setflags(write=False)
        column_order.setflags(write=False)
        return LayoutPlan(self.method, permutation, column_order, table.columns,
                          time.perf_counter() - started, planned_rows,
                          encoded_bytes, workspace_bytes)

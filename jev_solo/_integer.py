"""Exact prefix grouping with integer stamps and stable counting partitions.

Rows with the same current prefix are contiguous in ``permutation``. Scanning
them in that order makes ``seen[value] = prefix_group`` sufficient to count
distinct (group, value) pairs. No Cartesian-product table, composite-key
arithmetic, candidate sorting, or candidate inverse arrays are needed.

The compiled scans take O(N) time per candidate and use O(N) shared workspace,
in addition to the encoded table. Only a selected column refines the row order.
Two stable counting passes order its (prefix_group, value) pairs, preserving
exact lexicographic order, including the original order of duplicate rows.
"""
from __future__ import annotations

import numpy as np
from numba import njit


def code_dtype(cardinality: int) -> np.dtype:
    """Smallest unsigned dtype that can represent IDs 0..cardinality-1."""
    last_id = max(0, cardinality - 1)
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if last_id <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError("integer code exceeds uint64")


@njit(cache=True)
def _count(codes, permutation, groups, seen, column, cardinality):
    seen[:cardinality] = -1
    count = 0
    for position in range(len(permutation)):
        row = permutation[position]
        value = np.int64(codes[row, column])
        group = np.int64(groups[row])
        if seen[value] != group:
            seen[value] = group
            count += 1
    return count


@njit(cache=True)
def _refine(codes, permutation, groups, scratch, offsets,
            column, cardinality, group_count):
    n = len(permutation)
    # Stable pass on the secondary key (column value).
    offsets[:cardinality + 1] = 0
    for position in range(n):
        row = permutation[position]
        value = np.int64(codes[row, column])
        offsets[value + 1] += 1
    for value in range(cardinality):
        offsets[value + 1] += offsets[value]
    for position in range(n):
        row = permutation[position]
        value = np.int64(codes[row, column])
        scratch[offsets[value]] = row
        offsets[value] += 1

    # Stable pass on the primary key (current prefix group).
    offsets[:group_count + 1] = 0
    for position in range(n):
        group = np.int64(groups[scratch[position]])
        offsets[group + 1] += 1
    for group in range(group_count):
        offsets[group + 1] += offsets[group]
    for position in range(n):
        row = scratch[position]
        group = np.int64(groups[row])
        permutation[offsets[group]] = row
        offsets[group] += 1

    # Dense IDs in sorted pair order. Every row is visited once; retain its
    # old pair separately before updating its group ID in place.
    previous_group = -1
    previous_value = -1
    new_group = -1
    for position in range(n):
        row = permutation[position]
        group = np.int64(groups[row])
        value = np.int64(codes[row, column])
        if group != previous_group or value != previous_value:
            new_group += 1
        previous_group = group
        previous_value = value
        groups[row] = new_group
    return new_group + 1


class PrefixGroups:
    """Reusable grouping workspace for an integer-coded table.

    Canonical codes are nonnegative and smaller than N. Sparse/negative integer
    inputs are factorized once on entry, so direct addressing cannot allocate
    an array proportional to an arbitrary original value. Integer column order
    is preserved by that normalization.
    """

    def __init__(self, codes: np.ndarray):
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.dtype.kind not in "iu":
            raise ValueError("codes must be a two-dimensional integer array")
        n, m = codes.shape
        if n and m and (int(codes.min()) < 0 or int(codes.max()) >= n):
            normalized = np.empty((n, m), dtype=code_dtype(n), order="F")
            for column in range(m):
                _, inverse = np.unique(codes[:, column], return_inverse=True)
                normalized[:, column] = inverse
            codes = normalized
        self.codes = codes
        self.cardinalities = (
            codes.max(axis=0).astype(np.intp) + 1 if n else np.zeros(m, dtype=np.intp)
        )
        self.permutation = np.arange(n, dtype=np.intp)
        self.groups = np.zeros(n, dtype=code_dtype(n))
        self.group_count = 1 if n else 0
        self._seen = np.empty(int(self.cardinalities.max()) if m else 0, dtype=np.intp)
        self._scratch = np.empty(n, dtype=np.intp)
        self._offsets = np.empty(n + 1, dtype=np.intp)

    def count(self, column: int) -> int:
        """Count distinct (current prefix, candidate value) pairs exactly."""
        cardinality = int(self.cardinalities[column])
        if self.group_count == len(self.permutation) or cardinality <= 1:
            return self.group_count
        return int(_count(self.codes, self.permutation, self.groups, self._seen,
                          column, cardinality))

    def refine(self, column: int) -> int:
        """Commit one column and update the stable lexicographic permutation."""
        cardinality = int(self.cardinalities[column])
        if self.group_count != len(self.permutation) and cardinality > 1:
            self.group_count = int(_refine(
                self.codes, self.permutation, self.groups, self._scratch,
                self._offsets, column, cardinality, self.group_count,
            ))
        return self.group_count

    @property
    def workspace_bytes(self) -> int:
        """Row-grouping buffers; excludes input and O(M) cardinality metadata."""
        return sum(array.nbytes for array in (
            self.permutation, self.groups, self._seen, self._scratch, self._offsets,
        ))

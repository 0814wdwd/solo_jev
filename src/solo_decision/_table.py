"""Table normalization; pandas is optional and imported only for DataFrames."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from os import PathLike
from pathlib import Path

import numpy as np


def _json_value(value):
    """Validate and copy JSON values, retaining object and array semantics."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"non-finite JSON number: {value}")


class JSONInput:
    """Validated JSON records whose scalar types are retained in model input.

    Use :func:`read_json` for text or files. Objects become a one-row table;
    arrays must contain objects with exactly the same top-level fields.
    Nested objects remain atomic cells, and nested array order is preserved.
    """
    def __init__(self, records):
        if isinstance(records, Mapping):
            records = [records]
        if not isinstance(records, (list, tuple)):
            raise ValueError("JSON input must be an object or an array of objects")
        if any(not isinstance(record, Mapping) for record in records):
            raise ValueError("JSON input must contain objects, not scalar values or arrays")
        copied = tuple(_json_value(record) for record in records)
        fields = tuple(copied[0]) if copied else ()
        if any(set(record) != set(fields) for record in copied):
            raise ValueError("all JSON records must contain the same fields; use explicit null for missing values")
        self.records = copied

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        return self.records[key]


def read_json(source, *, lines=False) -> JSONInput:
    """Read JSON text, a Path, or parsed records without losing JSON types.

    String arguments are JSON text, never filenames; use ``Path`` for files.
    ``.jsonl``/``.ndjson`` Paths select JSON Lines automatically. For JSON Lines
    text, pass ``lines=True``. Blank lines are ignored; each other line must
    contain one object. Duplicate keys and NaN/Infinity are rejected.
    """
    if isinstance(source, JSONInput):
        if lines:
            raise TypeError("lines=True requires JSON Lines text or a file")
        return source
    if isinstance(source, PathLike):
        path = Path(source)
        lines = lines or path.suffix.lower() in (".jsonl", ".ndjson")
        source = path.read_text(encoding="utf-8-sig")
    if isinstance(source, str):
        def decode(text):
            return json.loads(text, object_pairs_hook=_object_pairs, parse_constant=_nonfinite)
        if lines:
            records = []
            for number, line in enumerate(source.splitlines(), 1):
                if line.strip():
                    try:
                        record = decode(line)
                        if not isinstance(record, dict):
                            raise ValueError("each JSON Lines line must contain an object")
                        records.append(record)
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"invalid JSON Lines input at line {number}: {exc}") from exc
            source = records
        else:
            source = decode(source)
    elif lines:
        raise TypeError("lines=True requires JSON Lines text or a file")
    return JSONInput(source)


@dataclass
class Table:
    rows: np.ndarray
    columns: tuple[str, ...]
    index: object = None
    json_values: bool = False

    @property
    def shape(self):
        return self.rows.shape

    def __len__(self):
        return len(self.rows)


def as_table(data, columns=None) -> Table:
    if isinstance(data, Table):
        if columns is not None and tuple(map(str, columns)) != data.columns:
            raise ValueError("columns must match named inputs")
        return data
    index = None
    inferred = None
    native_json = isinstance(data, (JSONInput, str, PathLike, Mapping))
    if native_json:
        data = read_json(data)
        inferred = list(data[0]) if len(data) else list(columns) if columns is not None else []
        values = np.empty((len(data), len(inferred)), dtype=object)
        for i, record in enumerate(data):
            for c, key in enumerate(inferred):
                values[i, c] = record[key]
    elif type(data).__module__.split(".")[0] == "pandas":
        import pandas as pd
        if not isinstance(data, pd.DataFrame):
            raise TypeError("expected a pandas DataFrame, not a Series")
        inferred = list(data.columns)
        index = data.index.copy()
        values = data.to_numpy(dtype=object, copy=True)
    elif isinstance(data, np.ndarray):
        if data.dtype.names:
            if data.ndim != 1:
                raise ValueError("structured arrays must be one-dimensional")
            inferred = list(data.dtype.names)
            values = np.empty((len(data), len(inferred)), dtype=object)
            for c, name in enumerate(inferred):
                values[:, c] = data[name]
        else:
            values = data.astype(object, copy=True)
    else:
        records = list(data)
        if records and isinstance(records[0], Mapping):
            inferred = list(records[0])
            if any(not isinstance(r, Mapping) or set(r) != set(inferred) for r in records):
                raise ValueError("all records must contain the same fields")
            # Explicit assignment prevents equal-length nested arrays becoming
            # an accidental third NumPy dimension.
            values = np.empty((len(records), len(inferred)), dtype=object)
            for i, record in enumerate(records):
                for c, key in enumerate(inferred):
                    values[i, c] = record[key]
        elif not records:
            values = np.empty((0, len(columns) if columns is not None else 0), dtype=object)
        else:
            try:
                values = np.asarray(records, dtype=object)
            except ValueError as exc:
                raise ValueError("expected a rectangular table") from exc
    if values.ndim != 2:
        raise ValueError("expected a two-dimensional rectangular table")
    names = list(columns) if columns is not None else inferred
    if inferred is not None and columns is not None and tuple(map(str, names)) != tuple(map(str, inferred)):
        raise ValueError("columns must match named inputs; use columns to name an unnamed array")
    if names is None:
        names = [f"c{i}" for i in range(values.shape[1])]
    names = tuple(str(name) for name in names)
    if len(names) != values.shape[1]:
        raise ValueError("columns must match the number of table columns")
    if len(set(names)) != len(names):
        raise ValueError("column names must be unique after conversion to strings")
    # Flat table inputs keep the existing str(cell) serialization byte for
    # byte. JSON sources retain scalar types. Nested cells in table inputs
    # remain JSON objects/arrays while their scalar siblings keep str(cell).
    json_values = native_json or any(isinstance(value, (Mapping, list)) for value in values.flat)
    for i in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[i, c]
            if json_values:
                value = _json_value(value) if native_json or isinstance(value, (Mapping, list)) else str(value)
                values[i, c] = json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False)
            else:
                values[i, c] = str(value)
    return Table(values, names, index, json_values)


class Serializer:
    """Render complete rows on demand, with bounded reuse of escaped values."""
    def __init__(self, table: Table, column_order):
        self.table = table
        self.order = tuple(int(c) for c in column_order)
        self.keys = tuple(json.dumps(k, ensure_ascii=False) for k in table.columns)
        self.escape = lru_cache(maxsize=4096)(lambda s: json.dumps(s, ensure_ascii=False))

    def __call__(self, row: int) -> str:
        values = self.table.rows[row]
        return "{" + ",".join(
            self.keys[c] + ":" + (values[c] if self.table.json_values else self.escape(values[c]))
            for c in self.order
        ) + "}"

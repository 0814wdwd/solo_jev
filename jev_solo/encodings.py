"""State encodings for a block of relational rows.

A Jev request is `state` + typed questions, billed on input tokens only, with a
hard 64k budget per request (32k for state + the longest question). When the
state holds a *block* of rows and we ask one question per row, the rows must be
addressable, so every block encoding pays explicit row ids. The per-row-request
baseline does not pay that, and we keep the asymmetry visible rather than
quietly dropping it.

FACTORED_PREAMBLE is the price of the ditto convention: paid once per request.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

FACTORED_PREAMBLE = (
    "Rows are sorted. A row starting with ^ repeats every field of the row "
    "above that it does not restate."
)

NULL_SENTINEL = "\\N"

CSV_RLE_PREAMBLE = (
    "CSV, rows sorted. An empty cell repeats the value from the row above; "
    "\\N means the value is genuinely missing."
)

ENCODINGS = ("row_kv", "row_json", "csv_block", "csv_rle", "factored_rle", "columnar_rle")


def _cell(value) -> str:
    return "" if value is None else str(value)


def _csv_cell(value) -> str:
    """RFC4180 quoting.

    Without this, a value containing a comma silently adds a field and shifts
    every later column out of alignment with the header. On the flight table
    `OriginCityName` is "Hartford, CT" on every row, so every csv-family state
    was misaligned until this was added. Values with commas are ordinary in real
    tables; a serializer that ignores them is simply wrong.
    """
    text = _cell(value)
    if any(ch in text for ch in (',', '"', '\n', '\r')):
        return '"' + text.replace('"', '""') + '"'
    return text


def encode_row_kv(header: Sequence[str], rows: Sequence[Sequence[str]], row_ids: bool = True) -> str:
    """SOLO's current per-row format. Column names repeat on every row."""
    out = []
    for i, row in enumerate(rows):
        body = " | ".join(f"{h}={_cell(v)}" for h, v in zip(header, row))
        out.append(f"r{i + 1}: {body}" if row_ids else body)
    return "\n".join(out)


def encode_row_json(header: Sequence[str], rows: Sequence[Sequence[str]], row_ids: bool = True) -> str:
    """JSON array of objects. Keys repeat on every row -- the verbose baseline."""
    import json

    recs = []
    for i, row in enumerate(rows):
        rec = {h: _cell(v) for h, v in zip(header, row)}
        if row_ids:
            rec = {"id": f"r{i + 1}", **rec}
        recs.append(rec)
    return json.dumps(recs, ensure_ascii=False, separators=(",", ":"))


def encode_csv_block(header: Sequence[str], rows: Sequence[Sequence[str]], row_ids: bool = True) -> str:
    """Header once, then CSV rows. Column names are paid a single time."""
    head = ("id," if row_ids else "") + ",".join(_csv_cell(h) for h in header)
    out = [head]
    for i, row in enumerate(rows):
        line = ",".join(_csv_cell(v) for v in row)
        out.append(f"r{i + 1},{line}" if row_ids else line)
    return "\n".join(out)


def _rle_cell(value) -> str:
    """Quote for CSV, and make a genuinely empty value distinguishable from ditto.

    Empty-means-ditto collides with empty-means-NULL: without a sentinel, a NULL
    is silently read as the value of the row above. flight carries 34 empty
    DepDelay and 40 empty ArrDelay values per 2000 rows, so this is not a corner
    case.
    """
    text = _cell(value)
    return NULL_SENTINEL if text == "" else _csv_cell(text)


def encode_csv_rle(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    row_ids: bool = True,
    pin: Sequence[int] = (),
) -> str:
    """CSV with ditto: column names once, and unchanged leading cells left empty.

    This is the encoding the objective argues for. Column names are amortized
    like csv_block, but an emitted (non-empty) cell in column k still occurs
    exactly once per run, so the count is G^(k) and the cost per run collapses
    to the value plus one comma -- no "Name=" label. Reordering therefore pays
    here, which it does not in plain csv_block.

    `pin` names column indices that are ALWAYS restated, never dittoed. This
    exists because of a measured failure: accuracy collapses exactly on the
    columns a question references when those columns are the compressible ones.
    A low-NDV column sorts early, gets dittoed away for most rows, and the model
    then has to recover it from a distant earlier row -- so the better a column
    compresses, the less often its value sits near the row that needs it.
    Pinning the predicate's columns spends tokens precisely where comprehension
    needs them.
    """
    pinned = set(pin)
    out = [CSV_RLE_PREAMBLE,
           ("id," if row_ids else "") + ",".join(_csv_cell(h) for h in header)]
    prev: Sequence[str] | None = None
    for i, row in enumerate(rows):
        if prev is None:
            cells = [_rle_cell(v) for v in row]
        else:
            start = 0
            while start < len(row) and _cell(row[start]) == _cell(prev[start]):
                start += 1
            if start == len(row):
                start = len(row) - 1
            cells = [""] * start + [_rle_cell(row[k]) for k in range(start, len(row))]
            for k in pinned:  # restate the predicate's columns unconditionally
                if k < len(row):
                    cells[k] = _rle_cell(row[k])
        line = ",".join(cells)
        out.append(f"r{i + 1},{line}" if row_ids else line)
        prev = row
    return "\n".join(out)


def encode_factored_rle(header: Sequence[str], rows: Sequence[Sequence[str]], row_ids: bool = True) -> str:
    """Header once; each row restates only the suffix that changed.

    Emitted values in column position k equal the number of runs in that
    column, which for lexicographically sorted rows is exactly G^(k). This is
    the encoding the token objective in objective.py is derived for.
    """
    out = [FACTORED_PREAMBLE]
    prev: Sequence[str] | None = None
    for i, row in enumerate(rows):
        if prev is None:
            start = 0
        else:
            start = 0
            while start < len(row) and _cell(row[start]) == _cell(prev[start]):
                start += 1
            if start == len(row):  # fully duplicate row: still needs its id
                start = len(row) - 1
        body = " | ".join(f"{header[k]}={_cell(row[k])}" for k in range(start, len(row)))
        prefix = f"r{i + 1}: " if row_ids else ""
        out.append(f"{prefix}{'^ ' if start > 0 else ''}{body}")
        prev = row
    return "\n".join(out)


def encode_columnar_rle(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    row_ids: bool = True,
    pin: Sequence[int] = (),
) -> str:
    """Column-major with run lengths: pays run counts instead of repeating values.

    `pin` names columns emitted value-by-value with no run collapsing, the
    columnar analogue of csv_rle's pinning: a run-length-collapsed column forces
    the model to count positions to find the value for a given row id, which is
    exactly what it cannot do reliably.
    """
    if not rows:
        return ""
    pinned = set(pin)
    n_cols = len(rows[0])
    out = [f"{len(rows)} rows, ids r1..r{len(rows)}, column-major with run lengths:"]
    for c in range(n_cols):
        if c in pinned:
            body = ", ".join(f"r{i + 1}={_cell(row[c])}" for i, row in enumerate(rows))
            out.append(f"{header[c]}: {body}")
            continue
        runs: List[Tuple[str, int]] = []
        for row in rows:
            v = _cell(row[c])
            if runs and runs[-1][0] == v:
                runs[-1] = (v, runs[-1][1] + 1)
            else:
                runs.append((v, 1))
        body = ", ".join(v if n == 1 else f"{v}x{n}" for v, n in runs)
        out.append(f"{header[c]}: {body}")
    return "\n".join(out)


_DISPATCH = {
    "row_kv": encode_row_kv,
    "row_json": encode_row_json,
    "csv_block": encode_csv_block,
    "csv_rle": encode_csv_rle,
    "factored_rle": encode_factored_rle,
    "columnar_rle": encode_columnar_rle,
}


def encode(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    encoding: str = "factored_rle",
    row_ids: bool = True,
) -> str:
    try:
        fn = _DISPATCH[encoding]
    except KeyError:
        raise ValueError(f"unknown encoding: {encoding!r}; known: {sorted(_DISPATCH)}") from None
    return fn(header, rows, row_ids=row_ids)

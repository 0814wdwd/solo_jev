"""Tables and predicates, as the library's public interface.

Everything measured so far came from one table, so the encoding-selection rule the
results imply could be an artifact of that schema. This module makes the table and
its predicates the caller's input rather than a constant in a probe script, which
is also how a user of the library would bring their own workload.

Predicate shapes are deliberately mirrored across tables so the rule can be tested
rather than merely restated:

    shape                          flight           movies
    numeric compare, 2 cols        delay_gt         profit
    numeric threshold, 1 col       dist_gt          budget_gt, vote_high
    categorical equality, 2 cols   state_eq         --
    skewed categorical, 1 col      carrier_in       lang_en
    binary / very low NDV          weekend          is_action
    arithmetic over 2 cols         taxi_sum         --
    conjunction                    conj             hit

`truth` is evaluated on the raw cell strings, so ground truth comes from the table
and never from a model.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

# Where to look for the example tables. The built-in specs name a filename, not a
# machine-specific path, so a clone works anywhere the files are placed:
#   JEV_SOLO_DATA_DIR=/path/to/data python3 -m jev_solo.datasets
# `fallback` records where the file sat on the machine the measurements were taken
# on, purely so those runs stay reproducible; it is tried last.
DATA_DIRS = [d for d in (os.environ.get("JEV_SOLO_DATA_DIR"), "data", "./data") if d]


def resolve(filename: str, fallback: str = "") -> str:
    """First existing of $JEV_SOLO_DATA_DIR/<file>, ./data/<file>, then fallback."""
    for d in DATA_DIRS:
        cand = Path(d) / filename
        if cand.exists():
            return str(cand)
    if fallback and Path(fallback).exists():
        return fallback
    return str(Path(DATA_DIRS[0] if DATA_DIRS else ".") / filename)


def _num(v: str) -> float:
    return float(str(v).strip().strip('"'))


def _s(v: str) -> str:
    return str(v).strip().strip('"')


@dataclass
class Predicate:
    name: str
    cols: List[str]          # columns the model is shown, and pinned
    shape: str
    ask_verbose: str         # must contain {rid}
    ask_terse: str
    truth: Callable[[Sequence[str]], bool]
    truth_cols: Optional[List[str]] = None
    semantic: bool = False

    @property
    def label_cols(self) -> List[str]:
        """Columns ground truth is computed from — withheld from the model when
        they differ from `cols`.

        Every predicate that ships with a table below except the semantic ones is a
        deterministic function of the columns the model is shown. That made them
        ideal for evaluation and useless as a value case: a gradient-boosted tree on
        those same columns scores 97-100% on all of them, for free. A decision model
        earns its place only where the shown columns do NOT determine the answer, so
        `truth_cols` lets a predicate read its label from a column the model never
        sees.
        """
        return self.truth_cols or self.cols


@dataclass
class TableSpec:
    name: str
    path: str  # use resolve(filename, fallback) for the built-in specs
    predicates: Dict[str, Predicate]
    delimiter: str = ","
    use_cols: Optional[List[str]] = None  # explicit projection; None = first `width`
    width: int = 20                       # columns used when use_cols is None
    exclude_cols: List[str] = field(default_factory=list)
    note: str = ""

    def load(self, limit: int = 40000) -> tuple[List[str], List[List[str]]]:
        out: List[List[str]] = []
        with open(self.path, newline="", encoding="utf-8", errors="replace") as f:
            r = csv.reader(f, delimiter=self.delimiter)
            header = [_s(h) for h in next(r)]
            for row in r:
                if len(row) == len(header):
                    out.append([_s(cell) for cell in row])
                if len(out) >= limit:
                    break
        return header, out

    def projection(self, header: Sequence[str], predicate: Predicate) -> List[int]:
        """Column indices to serialize: the projection plus the predicate's columns."""
        if self.use_cols is not None:
            base = [header.index(c) for c in self.use_cols if c in header]
        else:
            base = [i for i, h in enumerate(header) if h not in self.exclude_cols][: self.width]
        need = [header.index(c) for c in predicate.cols]
        return sorted(set(base + need))


def _p(name, cols, shape, verbose, terse, truth, truth_cols=None,
       semantic=False) -> Predicate:
    return Predicate(name=name, cols=cols, shape=shape, ask_verbose=verbose,
                     ask_terse=terse, truth=truth, truth_cols=truth_cols,
                     semantic=semantic)


def _genre(flag: str, phrase: str) -> Predicate:
    """Show the title; ask about the film; grade against a withheld genre flag."""
    return _p(f"is_{flag.lower()}", ["title"], "semantic: title -> genre",
              "For row r{rid}: " + phrase,
              "r{rid}: " + phrase,
              lambda v: str(v[0]).strip() in ("1", "1.0", "True"),
              truth_cols=[flag], semantic=True)


FLIGHT = TableSpec(
    name="flight",
    path=resolve("flight_100k.csv", "/home/ubuntu/lyz/fd_datasets/flight_100k.csv"),
    width=20,
    note="110 columns of airline on-time data, sorted by carrier and route in the "
         "file, so consecutive rows are highly redundant.",
    predicates={
        "delay_gt": _p("delay_gt", ["ArrDelay", "DepDelay"], "numeric compare, 2 cols",
                       "For row r{rid}: the arrival delay is strictly greater than the "
                       "departure delay.",
                       "r{rid}: ArrDelay > DepDelay.",
                       lambda v: _num(v[0]) > _num(v[1])),
        "state_eq": _p("state_eq", ["OriginState", "DestState"],
                       "categorical equality, 2 cols",
                       "For row r{rid}: the origin state and the destination state are "
                       "the same.",
                       "r{rid}: OriginState equals DestState.",
                       lambda v: _s(v[0]) == _s(v[1])),
        "dist_gt": _p("dist_gt", ["Distance"], "numeric threshold, 1 col",
                      "For row r{rid}: the distance is greater than 1000.",
                      "r{rid}: Distance > 1000.",
                      lambda v: _num(v[0]) > 1000),
        "weekend": _p("weekend", ["DayOfWeek"], "set membership, 1 low-NDV col",
                      "For row r{rid}: the day of week is 6 or 7.",
                      "r{rid}: DayOfWeek is 6 or 7.",
                      lambda v: _s(v[0]) in ("6", "7")),
        "carrier_in": _p("carrier_in", ["Reporting_Airline"],
                         "set membership, 1 low-NDV col",
                         "For row r{rid}: the reporting airline is AA, DL or UA.",
                         "r{rid}: Reporting_Airline is AA, DL or UA.",
                         lambda v: _s(v[0]) in ("AA", "DL", "UA")),
        "taxi_sum": _p("taxi_sum", ["TaxiOut", "TaxiIn"], "arithmetic over 2 cols",
                       "For row r{rid}: taxi out plus taxi in, added together, is "
                       "greater than 30.",
                       "r{rid}: TaxiOut + TaxiIn > 30.",
                       lambda v: _num(v[0]) + _num(v[1]) > 30),
        "conj": _p("conj", ["DepDelay", "ArrDelay"], "conjunction, 2 cols",
                   "For row r{rid}: the departure delay is greater than 0 and the "
                   "arrival delay is also greater than 0.",
                   "r{rid}: DepDelay > 0 and ArrDelay > 0.",
                   lambda v: _num(v[0]) > 0 and _num(v[1]) > 0),
    },
)

MOVIES = TableSpec(
    name="movies",
    path=resolve("movies_after_preprocess.csv",
                 "/home/ubuntu/gym/datamine/lab1/output/movies_after_preprocess.csv"),
    # `crew` is a multi-kilobyte JSON blob per row: it would dominate every state and
    # shrink blocks to a handful of rows, which is a separate stress case, not this one.
    exclude_cols=["crew"],
    width=20,
    note="36 columns of film metadata: a few high-cardinality numerics, one skewed "
         "language column, and ~20 binary genre flags whose values collapse into long "
         "runs once sorted -- the low-emit regime by construction.",
    predicates={
        "profit": _p("profit", ["revenue", "budget"], "numeric compare, 2 cols",
                     "For row r{rid}: the revenue is strictly greater than the budget.",
                     "r{rid}: revenue > budget.",
                     lambda v: _num(v[0]) > _num(v[1])),
        "budget_gt": _p("budget_gt", ["budget"], "numeric threshold, 1 col",
                        "For row r{rid}: the budget is greater than 50 million.",
                        "r{rid}: budget > 50000000.",
                        lambda v: _num(v[0]) > 50_000_000),
        "vote_high": _p("vote_high", ["vote_average"], "numeric threshold, 1 col",
                        "For row r{rid}: the average vote is greater than 6.5.",
                        "r{rid}: vote_average > 6.5.",
                        lambda v: _num(v[0]) > 6.5),
        "lang_en": _p("lang_en", ["original_language"],
                      "skewed categorical, 1 low-NDV col",
                      "For row r{rid}: the original language is English.",
                      "r{rid}: original_language is en.",
                      lambda v: _s(v[0]) == "en"),
        "action_flag": _p("action_flag", ["Action"], "binary flag, 1 very low-NDV col",
                        "For row r{rid}: the Action flag is set to 1.",
                        "r{rid}: Action is 1.",
                        lambda v: _s(v[0]) in ("1", "1.0", "True")),
        "long_film": _p("long_film", ["runtime"], "numeric threshold, 1 col",
                        "For row r{rid}: the runtime is greater than 120 minutes.",
                        "r{rid}: runtime > 120.",
                        lambda v: _num(v[0]) > 120),
        # Semantic: the model is shown only the title and must know the film.
        "is_action": _genre("Action", "this film is an action film."),
        "is_comedy": _genre("Comedy", "this film is a comedy."),
        "is_drama": _genre("Drama", "this film is a drama."),
        "is_horror": _genre("Horror", "this film is a horror film."),
        "hit": _p("hit", ["vote_average", "vote_count"], "conjunction, 2 cols",
                  "For row r{rid}: the average vote is above 7 and the vote count is "
                  "above 1000.",
                  "r{rid}: vote_average > 7 and vote_count > 1000.",
                  lambda v: _num(v[0]) > 7 and _num(v[1]) > 1000),
    },
)

def make_synthetic(path: str = "data/synthetic.csv", rows: int = 4000,
                   seed: int = 0) -> str:
    """A table with a controlled NDV profile, for testing the rule's inputs directly.

    flight and movies agree on the policy but they are two points in a large space.
    Here the cardinality of each column is set explicitly, from 2 to `rows`, so a
    predicate can be attached to a column of known cardinality and the emit fraction
    it produces is a property we chose rather than one we found. That is the only way
    to test "emit fraction predicts the damage" as a relationship rather than as two
    anecdotes.

    Columns: k2, k4, k16, k64, k256, k4000 (cardinality in the name), plus two
    numerics with a known relationship (`a`, `b`).
    """
    import csv as _csv
    import random as _random

    rng = _random.Random(seed)
    cards = [2, 4, 16, 64, 256, rows]
    header = [f"k{c}" for c in cards] + ["a", "b"]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(header)
        for _ in range(rows):
            row = [f"v{rng.randrange(c)}" for c in cards]
            a = rng.randint(-40, 200)
            row += [str(a), str(a + rng.randint(-60, 60))]
            w.writerow(row)
    return str(out)


SYNTHETIC = TableSpec(
    name="synthetic",
    path=resolve("synthetic.csv"),
    width=8,
    note="Generated by make_synthetic(): column cardinality is set explicitly (2, 4, "
         "16, 64, 256, n), so emit fraction is a controlled variable rather than an "
         "observed one.",
    predicates={
        "a_gt_b": _p("a_gt_b", ["a", "b"], "numeric compare, 2 cols",
                     "For row r{rid}: a is strictly greater than b.",
                     "r{rid}: a > b.",
                     lambda v: _num(v[0]) > _num(v[1])),
        "low_card": _p("low_card", ["k4"], "membership on a cardinality-4 column",
                       "For row r{rid}: k4 is v0 or v1.",
                       "r{rid}: k4 in (v0, v1).",
                       lambda v: _s(v[0]) in ("v0", "v1")),
        "mid_card": _p("mid_card", ["k64"], "threshold on a cardinality-64 column",
                       "For row r{rid}: the number in k64 is below 32.",
                       "r{rid}: k64 < v32.",
                       lambda v: int(_s(v[0])[1:]) < 32),
        "high_card": _p("high_card", ["k256"], "threshold on a cardinality-256 column",
                        "For row r{rid}: the number in k256 is below 128.",
                        "r{rid}: k256 < v128.",
                        lambda v: int(_s(v[0])[1:]) < 128),
    },
)

TABLES: Dict[str, TableSpec] = {t.name: t for t in (FLIGHT, MOVIES, SYNTHETIC)}


def get(name: str) -> TableSpec:
    try:
        return TABLES[name]
    except KeyError:
        raise SystemExit(f"unknown table {name!r}; known: {sorted(TABLES)}") from None


def validate(spec: TableSpec, limit: int = 5000) -> None:
    """Check the file parses, the columns exist, and each predicate is non-degenerate."""
    if not Path(spec.path).exists():
        print(f"!! {spec.name}: no file at {spec.path}. "
              f"Put it in ./data/ or set JEV_SOLO_DATA_DIR.")
        return
    header, rows = spec.load(limit)
    print(f"# {spec.name}: {len(rows)} rows x {len(header)} cols  ({spec.path})")
    for name, pred in spec.predicates.items():
        missing = [c for c in pred.cols if c not in header]
        if missing:
            print(f"  !! {name}: missing columns {missing}")
            continue
        idx = [header.index(c) for c in pred.cols]
        ok = bad = pos = 0
        for row in rows:
            vals = [row[i] for i in idx]
            if any(v in ("", "NA", "NULL") for v in vals):
                bad += 1
                continue
            try:
                pos += int(bool(pred.truth(vals)))
                ok += 1
            except Exception:
                bad += 1
        rate = pos / ok if ok else float("nan")
        flag = "  <- DEGENERATE" if ok and (rate < 0.02 or rate > 0.98) else ""
        print(f"  {name:12s} {pred.shape:32s} usable={ok:5d} skipped={bad:5d} "
              f"base_rate={rate:.3f}{flag}")


if __name__ == "__main__":
    for t in TABLES.values():
        validate(t)
        print()

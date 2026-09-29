#!/usr/bin/env python3
"""Probe 2: how does Jev count the tokens it bills?

The offline objective in jev_solo/ is built on a proxy tokenizer (cl100k_base)
because Jev's tokenizer is unpublished. That proxy decides which state encoding
looks cheapest, so it has to be checked against real billed tokens before any
cost number goes in a paper.

Three things this measures:
  1. calibration -- fit billed_tokens ~ a * proxy_tokens + b over many payload
     sizes, and report the residuals. `a` is then the correction factor for
     bench_offline.py.
  2. encoding ranking -- the same rows serialized five ways. Does the billed
     ranking match the proxy ranking? (If not, the offline conclusions move.)
  3. container effects -- state as a plain string vs a JSON object vs an array
     of strings, holding content constant, and whether questions are billed as
     input tokens at all.

Usage:
  TYPESAFE_API_KEY=... python3 probes/probe_tokenize.py --out results/tok.json
  python3 probes/probe_tokenize.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_client import JevClient, add_common_args, noul, save  # noqa: E402
from jev_solo.encodings import ENCODINGS, encode  # noqa: E402
from jev_solo.plan import lex_sort_rows  # noqa: E402
from jev_solo.tokens import get_counter  # noqa: E402
from jev_solo import datasets  # noqa: E402

DEFAULT_CSV = datasets.FLIGHT.path  # portable: see jev_solo/datasets.py


def load(path: str, rows: int, cols: int):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        header = [h.strip().strip('"') for h in next(r)][:cols]
        out = []
        for row in r:
            out.append(row[:cols])
            if len(out) >= rows:
                break
    return header, out


def main() -> None:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--cols", type=int, default=20)
    ap.add_argument("--sizes", default="5,10,20,40,80,160",
                    help="row counts to sweep for calibration")
    ap.add_argument("--tokenizer", default="cl100k_base")
    args = ap.parse_args()

    client = JevClient(api_key=args.api_key, base_url=args.base_url,
                       model=args.model, dry_run=args.dry_run)
    counter = get_counter(args.tokenizer)
    header, rows = load(args.csv, 400, args.cols)
    rows = lex_sort_rows(rows)
    q = [noul("q1", "All rows are internally consistent.")]
    records: List[Dict[str, Any]] = []

    print(f"# probe_tokenize  proxy={args.tokenizer}  table={Path(args.csv).name}"
          f"  cols={args.cols}  base_url={args.base_url}")

    # 1 + 2: sweep sizes x encodings
    print("\n## calibration + encoding ranking")
    print(f"{'encoding':14s} {'rows':>5s} {'proxy_tok':>10s} {'billed_tok':>11s} {'ratio':>7s} {'latency_ms':>11s}")
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    for enc in ENCODINGS:
        for n in sizes:
            block = rows[:n]
            state = encode(header, block, enc, row_ids=True)
            proxy = counter(state)
            r = client.ask(state, q)
            billed = r.input_tokens
            ratio = (billed / proxy) if (billed and proxy) else None
            records.append({"phase": "calibration", "encoding": enc, "rows": n,
                            "proxy_tokens": proxy, "billed_tokens": billed,
                            "state_chars": len(state), "latency_s": r.latency_s,
                            "ok": r.ok, "status": r.status, "error": r.error})
            print(f"{enc:14s} {n:5d} {proxy:10d} "
                  f"{(billed if billed is not None else -1):11d} "
                  f"{(f'{ratio:.3f}' if ratio else 'n/a'):>7s} {r.latency_s*1000:11.1f}")

    # 3: container shape, content held constant
    print("\n## container shape (same content, three JSON shapes)")
    block = rows[:40]
    flat = encode(header, block, "csv_block", row_ids=True)
    containers = {
        "string": flat,
        "array_of_lines": flat.split("\n"),
        "json_object": {"table": [dict(zip(header, r)) for r in block]},
    }
    for name, state in containers.items():
        r = client.ask(state, q)
        records.append({"phase": "container", "container": name,
                        "billed_tokens": r.input_tokens, "latency_s": r.latency_s,
                        "ok": r.ok, "status": r.status, "error": r.error})
        print(f"  {name:16s} billed={r.input_tokens}  latency={r.latency_s*1000:.1f}ms")

    # 3b: are questions billed as input?
    print("\n## are questions billed? (same state, 1 vs 16 questions)")
    many = [noul(f"q{i}", f"Row r{i+1} has a plausible departure delay.") for i in range(16)]
    for label, qs in (("1_question", q), ("16_questions", many)):
        r = client.ask(flat, qs)
        records.append({"phase": "question_billing", "label": label,
                        "n_questions": len(qs), "billed_tokens": r.input_tokens,
                        "latency_s": r.latency_s, "ok": r.ok, "status": r.status})
        print(f"  {label:14s} billed={r.input_tokens}  latency={r.latency_s*1000:.1f}ms")

    ok = [r for r in records if r.get("phase") == "calibration" and r.get("billed_tokens")]
    if len(ok) >= 3:
        import numpy as np

        x = np.array([r["proxy_tokens"] for r in ok], dtype=float)
        y = np.array([r["billed_tokens"] for r in ok], dtype=float)
        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        ss = 1 - ((y - pred) ** 2).sum() / max(1e-9, ((y - y.mean()) ** 2).sum())
        print(f"\n# calibration: billed ~ {a:.4f} * proxy + {b:.1f}   R^2={ss:.4f}")
        print(f"# feed this to bench_offline.py as the correction factor for {args.tokenizer}")
    elif not args.dry_run:
        print("\n# not enough billed-token readings to calibrate; "
              "check whether the response reports usage at all")

    save(args.out, {"args": vars(args) | {"api_key": None}, "records": records})


if __name__ == "__main__":
    main()

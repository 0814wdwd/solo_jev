#!/usr/bin/env python3
"""Compare original and SOLO layouts against a live vllm-jev server."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from solo_layout import DecisionEngine, VllmJevBackend


def workload(rows: int, policy_repeats: int):
    paragraph = (
        "Refund policy: approve only when the purchase is verified and all "
        "required evidence is present. Reject requests with missing evidence. "
    )
    policy = paragraph * policy_repeats
    records, truth = [], []
    for index in range(rows):
        approved = index % 2 == 0
        records.append([
            f"request-{index:04d}",
            policy,
            f"customer-segment-{index % 4}",
            f"purchase-{index:04d}",
            ("purchase verified; all required evidence is present"
             if approved else
             "purchase verified; required receipt evidence is missing"),
        ])
        truth.append(approved)
    columns = ["request_id", "policy", "customer_context", "purchase", "evidence"]
    return records, columns, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8795")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--policy-repeats", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.policy_repeats < 1 or args.repeats < 1:
        parser.error("rows, policy-repeats and repeats must be positive")

    records, columns, truth = workload(args.rows, args.policy_repeats)
    backend = VllmJevBackend(
        args.url, use_prefix_cache=not args.disable_prefix_cache, timeout=300)
    with DecisionEngine(backend=backend, concurrency=args.concurrency) as engine:
        comparison = engine.compare(
            records,
            "According to the policy, should this refund be approved?",
            columns=columns,
            methods=("original", "solo"),
            repeats=args.repeats,
            truth=truth,
        )

    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": "vllm-jev",
        "model": "ZefanCai/Open-Jev-2B",
        "rows": args.rows,
        "policy_repeats": args.policy_repeats,
        "concurrency": args.concurrency,
        "prefix_cache": not args.disable_prefix_cache,
        "question": "According to the policy, should this refund be approved?",
        "columns": columns,
        "summary": comparison.summary,
        "runs": comparison.runs,
    }
    rendered = json.dumps(document, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
